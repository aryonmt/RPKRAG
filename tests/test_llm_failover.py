from document_qa.generation.answerer import LLMService
from document_qa.generation.service_pool import ServicePool


def _service(name: str) -> dict:
    return {"name": name, "base_url": "http://example.test", "api_key": "k", "model": "m"}


def _llm(pool: ServicePool):
    service = LLMService.__new__(LLMService)
    service.service_pool = pool
    service.timeout = 1
    service.max_tokens = 8
    service.last_used_service = None
    return service


def _reply(text: str):
    return type("R", (), {"message": type("M", (), {"content": text})()})()


def test_connection_error_uses_the_next_service(monkeypatch):
    monkeypatch.setattr("document_qa.generation.service_pool.random.choice", lambda items: items[0])
    called = []

    def fake_get_llm(service):
        class Client:
            def chat(self, messages):
                called.append(service["name"])
                if service["name"] == "a":
                    raise ConnectionError("Connection error.")
                return _reply("ok")

        return Client()

    service = _llm(ServicePool([_service("a"), _service("b")]))
    monkeypatch.setattr(service, "_get_llm", fake_get_llm)
    assert service.chat([], task_name="generation") == "ok"
    assert called == ["a", "b"]


def test_timeout_fails_over_only_once(monkeypatch):
    monkeypatch.setattr("document_qa.generation.service_pool.random.choice", lambda items: items[0])
    called = []

    def fake_get_llm(service):
        class Client:
            def chat(self, messages):
                called.append(service["name"])
                raise TimeoutError("Request timed out.")

        return Client()

    service = _llm(ServicePool([_service("a"), _service("b"), _service("c")]))
    monkeypatch.setattr(service, "_get_llm", fake_get_llm)
    try:
        service.chat([], task_name="generation")
    except TimeoutError:
        pass
    else:
        raise AssertionError("timeout should surface after one failover")
    assert called == ["a", "b"]
