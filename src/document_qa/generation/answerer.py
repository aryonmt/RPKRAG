import re
import time
import logging
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from document_qa.core.config import settings
from document_qa.generation.service_pool import ServicePool, load_services_from_file
from document_qa.generation.prompts import build_system_prompt, format_evidence
from document_qa.generation.validator import validate_and_build

logger = logging.getLogger(__name__)

_service_pool_instance = None

def get_service_pool() -> ServicePool:
    global _service_pool_instance
    if _service_pool_instance is None:
        services = load_services_from_file("llm_services.json")
        _service_pool_instance = ServicePool(services)
        logger.info(f"Service pool initialized with {len(services)} services.")
    return _service_pool_instance

class LLMService:
    def __init__(self):
        if not settings.LLM_DATA_POLICY_ACK:
            raise ValueError("llm_not_configured")
        self.service_pool = get_service_pool()
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_tokens = settings.LLM_MAX_OUTPUT_TOKENS
        self.last_used_service = None

    def _get_llm(self, service: dict) -> OpenAILike:
        return OpenAILike(
            api_base=service["base_url"],
            api_key=service["api_key"],
            model=service["model"],
            temperature=0.0,
            max_tokens=self.max_tokens,
            is_chat_model=True,
            timeout=self.timeout,
            max_retries=0
        )

    def chat(self, messages: list[ChatMessage], task_name: str = "unknown", exclude_services: list[str] = None) -> str:
        """One network retry for connection or 5xx. No retry on timeout or HTTP 429."""
        retried_network = False
        rate_limited = set(exclude_services or [])
        while True:
            try:
                service = self.service_pool.get_service(exclude=rate_limited)
            except RuntimeError as e:
                if "llm_rate_limited" in str(e) or "no_llm_service" in str(e):
                    raise RuntimeError("llm_rate_limited") from e
                raise

            self.last_used_service = service["name"]
            llm = self._get_llm(service)

            start = time.time()
            try:
                response = llm.chat(messages)
                duration = time.time() - start
                content = response.message.content or ""
                logger.info(f"LLM OK | task={task_name:12s} | service={service['name']:12s} | model={service['model']:30s} | {duration:.2f}s | {len(content)} chars")
                return content
            except Exception as e:
                duration = time.time() - start
                error_str = str(e).lower()
                if re.search(r"(?<!\d)429(?!\d)", error_str) or "quota" in error_str or "rate limit" in error_str:
                    self.service_pool.mark_rate_limited(service["name"])
                    rate_limited.add(service["name"])
                    logger.warning(f"LLM RATE-LIMIT | task={task_name} | service={service['name']} | {duration:.2f}s")
                    continue
                timed_out = "timeout" in error_str or "timed out" in error_str
                server_error = re.search(r"(?<!\d)(500|502|503|504)(?!\d)", error_str) is not None
                network = (not timed_out) and ("connection" in error_str or "connect" in error_str or server_error)
                if network and not retried_network:
                    retried_network = True
                    logger.warning(f"LLM NET-ERR | task={task_name} | service={service['name']} | {duration:.2f}s | {e}")
                    time.sleep(1)
                    continue
                logger.error(f"LLM FAILED | task={task_name} | service={service['name']} | {duration:.2f}s | {e}")
                raise

    def generate_answer(self, question: str, evidence_chunks: list[dict]) -> dict:
        if not evidence_chunks:
            logger.warning("generate_answer called with no evidence chunks.")
            return {"status": "abstained", "claims": [], "missing": [], "citations": []}

        sys_prompt = build_system_prompt()
        ev_text = format_evidence(evidence_chunks)
        user_prompt = f"Question: {question}\n\nEvidence:\n{ev_text}"

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]

        raw_output = self.chat(messages, task_name="generation")
        return validate_and_build(raw_output, evidence_chunks)