import pytest

from document_qa.generation.validator import validate_and_build


def _chunk(text: str, document_id: str = "d1") -> dict:
    return {
        "document_id": document_id,
        "original_text": text,
        "breadcrumb": "",
        "page_start": 1,
        "page_end": 1,
        "char_start": 0,
        "char_end": len(text),
        "filename": "a.pdf",
        "title": "a",
    }


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError):
        validate_and_build("این پاسخ JSON نیست {broken", [_chunk("متن")])


def test_abstained_without_claims_stays_abstained():
    raw = '{"status":"abstained","claims":[]}'
    result = validate_and_build(raw, [_chunk("متن سند")])
    assert result["status"] == "abstained"
    assert result["claims"] == []


def test_abstained_with_supported_claim_is_kept():
    raw = '{"status":"abstained","claims":[{"text":"درخواست مرخصی باید سه روز قبل باشد.","evidence_ids":["E1"]}]}'
    result = validate_and_build(raw, [_chunk("درخواست مرخصی باید سه روز قبل باشد.")])
    assert result["status"] == "answered"


def test_claim_without_evidence_id_matches_only_a_close_chunk():
    raw = '{"status":"answered","claims":[{"text":"درخواست مرخصی باید سه روز قبل باشد.","evidence_ids":[]}]}'
    result = validate_and_build(raw, [_chunk("درخواست مرخصی باید سه روز قبل باشد.")])
    assert result["status"] == "answered"
    assert result["citations"] == ["E1"]


def test_unrelated_claim_without_evidence_id_is_removed():
    raw = '{"status":"answered","claims":[{"text":"حقوق مدیرعامل علنی است","evidence_ids":[]}]}'
    result = validate_and_build(raw, [_chunk("درخواست مرخصی باید سه روز قبل باشد.")])
    assert result["status"] == "abstained"


def test_number_copied_from_the_question_is_allowed():
    raw = '{"status":"answered","claims":[{"text":"سقف خرید ۴۰ میلیون ریال نیاز به تأیید مدیر دارد","evidence_ids":["E1"]}]}'
    result = validate_and_build(
        raw,
        [_chunk("سقف خرید میلیون ریال نیاز به تأیید مدیر دارد.")],
        trace={"question": "خرید ۴۰ میلیون ریال را چه کسی تأیید می‌کند؟"},
    )
    assert result["status"] == "answered"


def test_invented_number_is_still_rejected():
    raw = '{"status":"answered","claims":[{"text":"مهلت ۱۲ روز است","evidence_ids":["E1"]}]}'
    result = validate_and_build(
        raw,
        [_chunk("مهلت ۱۲۳ روز است.")],
        trace={"question": "مهلت چند روز است؟"},
    )
    assert result["status"] == "abstained"


def test_missing_number_is_taken_from_another_supporting_chunk():
    raw = '{"status":"answered","claims":[{"text":"داده‌های سطح C4 با AES-256 نگهداری می‌شوند","evidence_ids":["E1"]}]}'
    result = validate_and_build(raw, [
        _chunk("داده‌های سطح C4 با ثبت لاگ کامل نگهداری می‌شوند."),
        _chunk("داده‌های سطح C4 با AES-256 نگهداری می‌شوند."),
    ])
    assert result["status"] == "answered"
    assert "E2" in result["citations"]


def test_number_in_an_unrelated_chunk_does_not_save_the_claim():
    raw = '{"status":"answered","claims":[{"text":"داده‌های سطح C4 با AES-256 نگهداری می‌شوند","evidence_ids":["E1"]}]}'
    result = validate_and_build(raw, [
        _chunk("داده‌های سطح C4 با ثبت لاگ کامل نگهداری می‌شوند."),
        _chunk("مهلت مرخصی ۲۵۶ روز است."),
    ])
    assert result["status"] == "abstained"


def test_number_must_appear_exactly():
    raw = '{"status":"answered","claims":[{"text":"مهلت ۱۲ روز است","evidence_ids":["E1"]}]}'
    result = validate_and_build(raw, [_chunk("مهلت ۱۲۳ روز است.")])
    assert result["status"] == "abstained"


def test_supported_claim_keeps_citation():
    raw = '{"status":"answered","claims":[{"text":"درخواست مرخصی باید سه روز قبل باشد.","evidence_ids":["E1"]}]}'
    result = validate_and_build(raw, [_chunk("درخواست مرخصی باید سه روز قبل باشد.")])
    assert result["status"] == "answered"
    assert result["citations"] == ["E1"]
