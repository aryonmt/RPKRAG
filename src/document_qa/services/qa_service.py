import uuid
import json
import os
import re
import time
import logging
from datetime import datetime, timezone
from qdrant_client.http import models
from document_qa.api.errors import AppError
from document_qa.core.config import settings
from document_qa.storage.sqlite import get_db_connection
from document_qa.retrieval.embedding import LocalOnnxEmbedding
from document_qa.retrieval.qdrant_index import get_qdrant_client
from document_qa.retrieval.fts_retriever import FtsRetriever
from document_qa.retrieval.fusion import rrf_fusion
from document_qa.retrieval.fts_query import build_fts_query
from document_qa.generation.answerer import LLMService
from document_qa.generation.prompts import build_system_prompt, format_evidence, build_query_expansion_prompt, build_rerank_prompt
from document_qa.generation.validator import validate_and_build
from llama_index.core.schema import NodeWithScore, TextNode
from document_qa.ingestion.normalizer import fold_digits
from document_qa.domain.policies import ABSTENTION_TEXT, CONFLICT_TEXT, MISSING_PREFIX
from llama_index.core.llms import ChatMessage

logger = logging.getLogger(__name__)

def _get_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _save_trace(trace: dict, final_res: dict):
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/latest_trace.json", "w", encoding="utf-8") as f:
            json.dump({"trace": trace, "result": final_res}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save trace: {e}")

def _calculate_math_override(question: str) -> str:
    normalized = fold_digits(question)
    avail_match = re.search(r"(\d+(?:\.\d+)?)\s*%", normalized)
    days_match = re.search(r"(\d+(?:\.\d+)?)\s*روز", normalized)
    if not avail_match or not days_match: return ""
    availability = float(avail_match.group(1))
    days = float(days_match.group(1))
    if not 0 <= availability <= 100 or days <= 0: return ""
    total_minutes = days * 24 * 60
    downtime = total_minutes * (100 - availability) / 100
    return (
        "\n\n[سیستمی - محاسبه دقیق]: "
        f"{days:g} روز برابر با {total_minutes:g} دقیقه است. "
        f"زمان عدم دسترس‌پذیری متناظر با دسترس‌پذیری {availability:g}% "
        f"برابر با {downtime:.1f} دقیقه است. "
        "این مقدار فقط محاسبه ریاضی است و به‌تنهایی واجد شرایط بودن اعتبار خدمت قراردادی را ثابت نمی‌کند."
    )

def _generate_multi_queries(question: str, llm_service: LLMService, trace: dict) -> list[str]:
    try:
        prompt = build_query_expansion_prompt(question)
        raw = llm_service.chat([ChatMessage(role="user", content=prompt)], task_name="multi_query").strip()
        trace["multi_query_model"] = llm_service.last_used_service
        trace["multi_query_prompt"] = prompt
        trace["multi_query_raw"] = raw
        if raw.startswith("```json"): raw = raw[7:-3].strip()
        elif raw.startswith("```"): raw = raw[3:-3].strip()
        parsed = json.loads(raw)
        
        if isinstance(parsed, dict) and "queries" in parsed:
            queries = parsed["queries"]
        elif isinstance(parsed, list):
            queries = parsed
        else:
            queries = []
            
        valid_queries = []
        if isinstance(queries, list):
            for q in queries:
                q_str = str(q).strip()
                if len(q_str.split()) < 20 and len(q_str) > 0:
                    valid_queries.append(q_str)
                
        if valid_queries:
            return [question] + valid_queries
    except Exception as e:
        logger.warning(f"Multi-query failed: {e}")
    return [question]

def _load_json_object(raw: str):
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _rerank_with_llm(question: str, chunks: list[dict], llm_service: LLMService, trace: dict) -> list[dict]:
    if not chunks:
        return []
    rerank_pool = chunks[:50]
    chunk_text = ""
    for i, c in enumerate(rerank_pool):
        chunk_text += f"ID: E{i+1}\nText: {c['original_text']}\n---\n"
    prompt = build_rerank_prompt(question, chunk_text)
    
    try:
        raw = llm_service.chat([ChatMessage(role="user", content=prompt)], task_name="rerank").strip()
        trace["rerank_model"] = llm_service.last_used_service
        trace["rerank_prompt"] = prompt
        trace["rerank_raw"] = raw
        scores = _load_json_object(raw)
        
        if not isinstance(scores, dict):
            return chunks
        
        scored_chunks = []
        for i, c in enumerate(rerank_pool):
            eid = f"E{i+1}"
            raw_score = scores.get(eid, 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0
            scored_chunks.append({"chunk": c, "score": score})
            
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in scored_chunks] + chunks[len(rerank_pool):]
    except Exception as e:
        logger.warning(f"Rerank failed: {e}")
        return chunks

def ask_question(conversation_id: str, question: str, return_trace: bool = False) -> dict:
    t_total = time.time()
    deadline = t_total + settings.QUESTION_DEADLINE_SECONDS
    logger.info(f"=== New Question | conv={conversation_id[:8]}... ===")
    
    conn = get_db_connection()
    trace = {"question": question}

    def _abort(status_code: int, code: str, message: str):
        err = AppError(status_code, code, message)
        err.trace = trace
        raise err
    
    conv = conn.execute("SELECT id FROM conversations WHERE id = ? AND status = 'active'", (conversation_id,)).fetchone()
    if not conv:
        conn.close()
        _abort(404, "conversation_not_found", "این گفتگو پیدا نشد.")

    docs = conn.execute("SELECT id FROM documents WHERE conversation_id = ? AND status IN ('indexed', 'partial_warning')", (conversation_id,)).fetchall()
    ready_docs = [d["id"] for d in docs]
    trace["ready_docs"] = ready_docs

    if not ready_docs:
        processing = conn.execute("SELECT id FROM documents WHERE conversation_id = ? AND status = 'processing'", (conversation_id,)).fetchall()
        conn.close()
        if processing:
            _abort(409, "ingest_in_progress", "هنوز سندی برای جست‌وجو آماده نیست.")
        else:
            res = _create_abstained_response(conversation_id, question)
            _save_trace(trace, res)
            if return_trace: return {"result": res, "trace": trace}
            return res

    try:
        llm_service = LLMService()
    except ValueError as e:
        conn.close()
        if str(e) == "llm_not_configured":
            _abort(503, "llm_not_configured", "ارسال متن سند به مدل زبانی هنوز تأیید نشده است. LLM_DATA_POLICY_ACK را روشن کنید.")
        raise

    def _past_deadline() -> bool:
        return time.time() > deadline

    if _past_deadline():
        conn.close()
        _abort(503, "llm_unavailable", "مهلت پاسخ به این سؤال تمام شد.")

    multi_queries = _generate_multi_queries(question, llm_service, trace)
    trace["multi_queries"] = multi_queries

    embed_model = LocalOnnxEmbedding()
    client = get_qdrant_client()
    
    all_dense_results = []
    try:
        for q in multi_queries:
            q_emb = embed_model._get_query_embedding(q)
            response = client.query_points(
                collection_name=settings.QDRANT_COLLECTION, query=q_emb,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="conversation_id", match=models.MatchValue(value=conversation_id)),
                    models.FieldCondition(key="document_id", match=models.MatchAny(any=ready_docs))
                ]),
                limit=settings.RETRIEVAL_DENSE_TOP_K
            )
            all_dense_results.extend([NodeWithScore(node=TextNode(id_=str(p.id)), score=p.score) for p in response.points])
            
        best_dense = {}
        for res in all_dense_results:
            prev = best_dense.get(res.node.id_)
            if prev is None or (res.score or 0) > (prev.score or 0):
                best_dense[res.node.id_] = res
        dense_results = sorted(best_dense.values(), key=lambda r: -(r.score or 0))
    except Exception as e:
        conn.close()
        logger.error(f"Dense retrieval FAILED: {e}")
        _abort(503, "retrieval_unavailable", "جست‌وجوی برداری در دسترس نیست.")
    
    fts = FtsRetriever(conversation_id=conversation_id, ready_document_ids=ready_docs)
    best_lexical = {}
    for q in multi_queries:
        for res in fts.retrieve(q):
            prev = best_lexical.get(res.node.id_)
            if prev is None or (res.score or 0) > (prev.score or 0):
                best_lexical[res.node.id_] = res
    lexical_results = sorted(best_lexical.values(), key=lambda r: -(r.score or 0))

    trace["retrieval"] = {"dense_count": len(dense_results), "lexical_count": len(lexical_results)}

    fused_results = rrf_fusion(dense_results, lexical_results, top_n=60)
    
    candidate_chunks = []
    seen_texts = []
    for res in fused_results:
        row = conn.execute("SELECT * FROM chunks WHERE node_id = ?", (res.node.id_,)).fetchone()
        if row:
            chunk_dict = dict(row)
            text = chunk_dict["original_text"].strip()
            if text in seen_texts:
                continue
            candidate_chunks.append(chunk_dict)
            seen_texts.append(text)

    trace["fusion"] = {
        "candidate_count": len(candidate_chunks),
        "candidates": [
            {
                "rank": i,
                "node_id": c["node_id"],
                "filename": c["filename"],
                "title": c["title"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "breadcrumb": c["breadcrumb"],
                "original_text": c["original_text"],
            }
            for i, c in enumerate(candidate_chunks, start=1)
        ],
    }
    
    if _past_deadline():
        conn.close()
        _abort(503, "llm_unavailable", "مهلت پاسخ به این سؤال تمام شد.")
    reranked_chunks = _rerank_with_llm(question, candidate_chunks, llm_service, trace)
    evidence_chunks = reranked_chunks[:settings.EVIDENCE_MAX]
    if not evidence_chunks:
        conn.close()
        res = _create_abstained_response(conversation_id, question)
        _save_trace(trace, res)
        if return_trace:
            return {"result": res, "trace": trace}
        return res
    
    trace["fusion"]["final_evidence"] = [
        {
            "eid": f"E{i}",
            "node_id": c["node_id"],
            "filename": c["filename"],
            "title": c["title"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "breadcrumb": c["breadcrumb"],
            "original_text": c["original_text"],
        }
        for i, c in enumerate(evidence_chunks, start=1)
    ]

    math_override = _calculate_math_override(question)
    trace["math_override"] = {"matched": bool(math_override), "value": math_override}

    sys_prompt = build_system_prompt()
    ev_text = format_evidence(evidence_chunks)
    user_prompt = f"Question: {question}\n{math_override}\n\nEvidence:\n{ev_text}"
    
    trace["llm_input"] = {"system_prompt": sys_prompt, "user_prompt": user_prompt}

    messages = [ChatMessage(role="system", content=sys_prompt), ChatMessage(role="user", content=user_prompt)]

    if _past_deadline():
        conn.close()
        _abort(503, "llm_unavailable", "مهلت پاسخ به این سؤال تمام شد.")

    try:
        raw_output = llm_service.chat(messages, task_name="generation")
        trace["used_model"] = llm_service.last_used_service
    except Exception as e:
        conn.close()
        logger.error(f"Generation FAILED: {e}")
        _save_trace(trace, {"status": "error", "error": str(e)})
        if "llm_rate_limited" in str(e):
            _abort(429, "llm_rate_limited", "سهمیه یا محدودیت درخواست مدل زبانی پر شده است. بعداً دوباره تلاش کنید.")
        _abort(503, "llm_unavailable", "سرویس مدل زبانی در دسترس نیست.")

    try:
        result = validate_and_build(raw_output, evidence_chunks, trace)
    except Exception as e:
        conn.close()
        logger.error(f"Validation FAILED: {type(e).__name__}: {e}")
        _save_trace(trace, {"status": "llm_output_invalid", "error": str(e)})
        _abort(503, "llm_output_invalid", "مدل پاسخ قابل‌پردازش برنگرداند؛ دوباره تلاش کنید.")

    cards = []
    if result["status"] in ("answered", "conflict"):
        evidence_map = {f"E{i}": chunk for i, chunk in enumerate(evidence_chunks, start=1)}
        for eid in result.get("citations", []):
            if eid in evidence_map:
                chunk = evidence_map[eid]
                quote = chunk["original_text"]
                truncated = len(quote) > settings.QUOTE_MAX_CHARS
                if truncated: quote = quote[:settings.QUOTE_MAX_CHARS].rsplit(' ', 1)[0]
                cards.append({
                    "evidence_id": eid, "filename": chunk["filename"], "title": chunk["title"],
                    "page_start": chunk["page_start"], "page_end": chunk["page_end"],
                    "breadcrumb": chunk["breadcrumb"], "quote": quote, "quote_truncated": truncated
                })

    if result["status"] == "abstained":
        answer_text = ABSTENTION_TEXT
    elif result["status"] == "conflict":
        answer_text = CONFLICT_TEXT
    else:
        answer_text = " ".join(result["claims"])
        if result.get("missing"):
            answer_text += "\n\n" + MISSING_PREFIX + "\n" + "\n".join(result["missing"])

    msg_id = str(uuid.uuid4())
    now = _get_now()
    conn.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)", (str(uuid.uuid4()), conversation_id, question, now))
    conn.execute("""INSERT INTO messages (id, conversation_id, role, content, status, answer_text, missing, citations, created_at) 
                    VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?)""", 
                 (msg_id, conversation_id, answer_text, result["status"], answer_text, json.dumps(result.get("missing", []), ensure_ascii=False), json.dumps(cards, ensure_ascii=False), now))
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()

    final_res = {
        "message_id": msg_id, "question": question, "standalone_question": None,
        "status": result["status"], "answer_text": answer_text,
        "missing": result.get("missing", []), "citations": cards
    }
    
    _save_trace(trace, final_res)
    logger.info(f"=== Done | status={result['status']} | gen_model={trace.get('used_model')} | total={time.time()-t_total:.2f}s ===")
    
    if return_trace: return {"result": final_res, "trace": trace}
    return final_res

def _create_abstained_response(conversation_id: str, question: str) -> dict:
    msg_id = str(uuid.uuid4())
    now = _get_now()
    answer_text = ABSTENTION_TEXT
    conn = get_db_connection()
    conn.execute("INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)", (str(uuid.uuid4()), conversation_id, question, now))
    conn.execute("""INSERT INTO messages (id, conversation_id, role, content, status, answer_text, missing, citations, created_at) 
                    VALUES (?, ?, 'assistant', ?, 'abstained', ?, '[]', '[]', ?)""", 
                 (msg_id, conversation_id, answer_text, answer_text, now))
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()
    return {"message_id": msg_id, "question": question, "standalone_question": None, "status": "abstained", "answer_text": answer_text, "missing": [], "citations": []}