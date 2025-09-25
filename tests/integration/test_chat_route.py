import pytest


def test_chat_invalid_session_returns_400(client, clear_sessions, stub_rag):
    body = {"session_id": "nope", "message": "hi"}
    resp = client.post("/chat", json=body)
    assert resp.status_code == 400
    assert "Invalid or expired" in resp.json()["detail"]


def test_chat_empty_message_returns_400(client, clear_sessions, stub_rag):
    sid = "sess_test"
    import main
    main.SESSIONS[sid] = []
    body = {"session_id": sid, "message": "   "}
    resp = client.post("/chat", json=body)
    assert resp.status_code == 400
    assert "Message cannot be empty" in resp.json()["detail"]


def test_chat_success_returns_answer_and_appends_history(client, clear_sessions, stub_rag):
    sid = "sess_test"
    import main
    main.SESSIONS[sid] = []
    body = {"session_id": sid, "message": "Hello"}
    resp = client.post("/chat", json=body)
    assert resp.status_code == 200
    assert resp.json()["answer"] == "stubbed answer"
    assert len(main.SESSIONS[sid]) == 2


def test_chat_failure_returns_500(client, clear_sessions, monkeypatch):
    sid = "sess_test"
    import main
    main.SESSIONS[sid] = []

    import main

    class BoomRAG:
        def __init__(self, session_id=None):
            pass
        def load_retriever_from_faiss(self, *a, **k):
            from multi_doc_chat.exception.custom_exception import DocumentPortalException
            raise DocumentPortalException("fail load", None)

    monkeypatch.setattr(main, "ConversationalRAG", BoomRAG)
    resp = client.post("/chat", json={"session_id": sid, "message": "hi"})
    assert resp.status_code == 500
    assert "fail load" in resp.json()["detail"].lower()