from document_qa.retrieval.fts_query import build_fts_query


def test_fts_query_quotes_tokens_and_drops_stopwords():
    query = build_fts_query("این درخواست مرخصی چیست")
    assert "مرخصی" in query
    assert " OR " in query or query.startswith('"')
    assert "این" not in query.split(" OR ")
