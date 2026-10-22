import pytest

from document_qa.retrieval.embedding import FakeEmbedding


def test_fake_embedding_is_deterministic_and_384d():
    model = FakeEmbedding()
    first = model._get_text_embedding("تست مدل تقلبی")
    second = model._get_text_embedding("تست مدل تقلبی")
    assert len(first) == 384
    assert first == second


@pytest.mark.model
def test_onnx_embedding_ranks_related_sentences_closer():
    import numpy as np

    from document_qa.retrieval.embedding import LocalOnnxEmbedding

    model = LocalOnnxEmbedding()
    leave = model._get_text_embedding("درخواست مرخصی باید سه روز قبل داده شود.")
    paraphrase = model._get_text_embedding("برای گرفتن مرخصی، سه روز زودتر درخواست بدهید.")
    weather = model._get_text_embedding("امروز هوا بسیار آفتابی و خوب است.")

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert len(leave) == 384
    assert cosine(leave, paraphrase) > cosine(leave, weather)
