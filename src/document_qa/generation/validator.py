import re
import json
from pathlib import Path
from document_qa.ingestion.normalizer import tokenize, remove_stopwords, load_stopwords, fold_digits
from document_qa.core.config import settings

_STOPWORDS_PATH = Path(__file__).resolve().parents[3] / "resources" / "fa_stopwords.txt"

def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", fold_digits(text))

def extract_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    if not (t.startswith("{") and t.endswith("}")):
        raise ValueError("llm_output_invalid")
    return t

def _best_evidence(claim_text: str, evidence_chunks: list[dict], stopwords: set) -> tuple[str | None, float]:
    claim_tokens = [t for t in remove_stopwords(tokenize(claim_text), stopwords) if len(t) >= 2 and not t.replace(".", "").isdigit()]
    if not claim_tokens:
        return None, 0.0
    best_eid, best_ratio = None, 0.0
    claim_set = set(claim_tokens)
    for i, chunk in enumerate(evidence_chunks, start=1):
        ev_tokens = set(tokenize(chunk["original_text"]))
        if not ev_tokens:
            continue
        ratio = sum(1 for token in claim_set if token in ev_tokens) / len(claim_set)
        if ratio > best_ratio:
            best_eid, best_ratio = f"E{i}", ratio
    return best_eid, best_ratio


def validate_and_build(raw_output: str, evidence_chunks: list[dict], trace: dict = None) -> dict:
    if trace is None: trace = {}
    trace["llm_raw_output"] = raw_output
    
    try:
        clean_json = extract_json(raw_output)
        data = json.loads(clean_json)
        if not isinstance(data, dict):
            trace["json_parse_success"] = False
            trace["json_parse_error"] = f"Expected JSON object, got {type(data).__name__}"
            raise ValueError("llm_output_invalid")
        trace["json_parse_success"] = True
    except (json.JSONDecodeError, ValueError) as e:
        trace["json_parse_success"] = False
        trace["json_parse_error"] = str(e)
        raise ValueError("llm_output_invalid")

    status = data.get("status", "abstained")
    if status not in ("answered", "abstained", "conflict"):
        trace["json_parse_success"] = False
        trace["json_parse_error"] = f"Invalid status: {status}"
        raise ValueError("llm_output_invalid")
    evidence_map = {f"E{i}": chunk for i, chunk in enumerate(evidence_chunks, start=1)}
    
    trace["rejected_claims"] = []
    trace["validated_claims"] = []

    if status == "abstained" and not data.get("claims"):
        return {"status": "abstained", "claims": [], "missing": [], "citations": []}
    if status == "abstained":
        trace["rejected_claims"].append({"reason": "status was abstained but claims were present; validating them"})
        status = "answered"

    if status == "conflict":
        c_ids = data.get("conflict_evidence_ids", [])
        valid_pair = [eid for eid in c_ids if eid in evidence_map]
        
        if len(valid_pair) >= 2:
            c1, c2 = evidence_map[valid_pair[0]], evidence_map[valid_pair[1]]
            is_distinct = False
            if c1["document_id"] != c2["document_id"]: is_distinct = True
            elif c1["page_start"] != c2["page_start"]: is_distinct = True
            elif c1["char_start"] >= c2["char_end"] or c1["char_end"] <= c2["char_start"]: is_distinct = True
                
            if is_distinct and c1["original_text"] != c2["original_text"]:
                return {"status": "conflict", "claims": [], "missing": [], "citations": [valid_pair[0], valid_pair[1]]}
        
        trace["rejected_claims"].append({"reason": "Conflict validation failed"})
        return {"status": "abstained", "claims": [], "missing": [], "citations": []}

    if status == "answered":
        valid_claims = []
        raw_missing = data.get("missing", [])
        if not isinstance(raw_missing, list):
            raw_missing = [str(raw_missing)]
        missing_parts = []
        for item in raw_missing:
            label = str(item).strip()
            if not label or len(label) > settings.MISSING_MAX_CHARS:
                continue
            if re.search(r"\d", fold_digits(label)):
                continue
            missing_parts.append(label)
            
        stopwords = load_stopwords(str(_STOPWORDS_PATH))
        math_override_text = trace.get("math_override", {}).get("value", "")
        allowed_math_nums = set(extract_numbers(math_override_text))
        question_nums = set(extract_numbers(trace.get("question") or ""))
        
        top_level_eids = data.get("evidence_ids", [])
        if not isinstance(top_level_eids, list):
            top_level_eids = []
        top_level_eids = [e for e in top_level_eids if isinstance(e, str) and e in evidence_map]

        claims_list = data.get("claims", [])
        if not isinstance(claims_list, list):
            claims_list = [claims_list]
        
        for claim in claims_list:
            # 1. Extract text and eids from claim
            if isinstance(claim, str):
                text = claim.strip()
                eids = []
            elif isinstance(claim, dict):
                text = claim.get("text", "").strip()
                eids = claim.get("evidence_ids", [])
                if not isinstance(eids, list):
                    eids = []
            else:
                continue
            
            if not text or len(text) > settings.CLAIM_MAX_CHARS:
                trace["rejected_claims"].append({"text": text[:80], "reason": "empty or too long"})
                continue
            
            if not eids:
                found_markers = re.findall(r'\((E\d+)\)|\[(E\d+)\]', text)
                eids = [m[0] or m[1] for m in found_markers]
            
            if not eids and top_level_eids:
                eids = top_level_eids
            valid_eids = [e for e in eids if isinstance(e, str) and e in evidence_map]
            if not valid_eids:
                best_eid, best_ratio = _best_evidence(text, evidence_chunks, stopwords)
                if best_eid and best_ratio >= 0.5:
                    valid_eids = [best_eid]
                else:
                    trace["rejected_claims"].append({"text": text[:80], "reason": "no evidence id in this bundle"})
                    continue

            def _chunk_text(eid: str) -> str:
                chunk = evidence_map[eid]
                return chunk["original_text"] + " " + (chunk.get("breadcrumb") or "")

            claim_tokens = [t for t in remove_stopwords(tokenize(text), stopwords)
                           if not t.replace('.', '').isdigit() and len(t) >= 2]
            if not claim_tokens:
                trace["rejected_claims"].append({"text": text[:80], "reason": "no content tokens"})
                continue

            def _overlap(eid: str) -> float:
                tokens = set(tokenize(_chunk_text(eid)))
                return sum(1 for token in claim_tokens if token in tokens) / len(claim_tokens)

            text_for_num_check = re.sub(r'\(E\d+\)|\[E\d+\]', '', text)
            text_for_num_check = re.sub(r'(?m)(^|\s)\d+(?:\.\d+)*[\.\)\-]\s+', ' ', text_for_num_check)
            claim_nums = set(extract_numbers(text_for_num_check))

            def _covered() -> set[str]:
                found = set()
                for eid in valid_eids:
                    found.update(extract_numbers(_chunk_text(eid)))
                return found

            missing_nums = [
                num for num in claim_nums
                if num not in _covered() and num not in allowed_math_nums and num not in question_nums
            ]
            for num in list(missing_nums):
                if num not in missing_nums:
                    continue
                candidates = [
                    eid for eid in evidence_map
                    if eid not in valid_eids and num in set(extract_numbers(_chunk_text(eid)))
                ]
                if not candidates:
                    continue
                best = max(candidates, key=_overlap)
                if _overlap(best) < settings.LEXICAL_SUPPORT_MIN_RATIO:
                    continue
                valid_eids.append(best)
                added_nums = set(extract_numbers(_chunk_text(best)))
                missing_nums = [item for item in missing_nums if item not in added_nums]

            if missing_nums:
                trace["rejected_claims"].append({
                    "text": text[:80],
                    "reason": f"Number '{missing_nums[0]}' not found in cited evidence"
                })
                continue

            check_text = " ".join(_chunk_text(eid) for eid in valid_eids)
            
            check_tokens = set(tokenize(check_text + " " + math_override_text))
            matches = sum(1 for t in claim_tokens if t in check_tokens)
            ratio = matches / len(claim_tokens)
            
            if ratio >= settings.LEXICAL_SUPPORT_MIN_RATIO:
                valid_claims.append({"text": text, "evidence_ids": valid_eids})
                trace["validated_claims"].append({"text": text, "eids": valid_eids, "lexical_ratio": ratio})
            else:
                trace["rejected_claims"].append({
                    "text": text[:80],
                    "reason": f"Lexical support too low ({ratio:.2f} < {settings.LEXICAL_SUPPORT_MIN_RATIO})"
                })
                
        if valid_claims:
            citations = []
            for c in valid_claims:
                for eid in c["evidence_ids"]:
                    if eid not in citations: citations.append(eid)
            return {"status": "answered", "claims": [c["text"] for c in valid_claims], "missing": missing_parts, "citations": citations}
        else:
            return {"status": "abstained", "claims": [], "missing": [], "citations": []}

    return {"status": "abstained", "claims": [], "missing": data.get("missing", []), "citations": []}