"""
Report Analysis Agent: extracts diabetes-relevant lab values from a report's raw
text (via the LLM) and classifies each value against standard reference ranges
using a deterministic rule table (so classification never depends on the LLM's
arithmetic).

Safeguards:
  * values the LLM returns are dropped if the number does not actually appear in
    the report text (hallucination guard);
  * common unit variants (mmol/L, umol/L, mmol/mol, mg/mmol) are converted to the
    canonical unit before classification;
  * ranges use exclusive upper bounds, so decimal values never fall into gaps.

Labels say "..._range" on purpose: a single lab value is not a diagnosis.
"""
import math
import re
from src import llm_client
from src.agents.json_utils import extract_json

INF = math.inf

# Ordered bands per parameter: a value belongs to the first band whose (exclusive)
# upper bound is greater than the value.
REFERENCE_BANDS = {
    "HbA1c": {"unit": "%", "bands": [(4.0, "low"), (5.7, "normal"), (6.5, "prediabetes_range"), (INF, "diabetes_range")]},
    "FBG": {"unit": "mg/dL", "bands": [(70, "low"), (100, "normal"), (126, "prediabetes_range"), (INF, "diabetes_range")]},
    # Random glucose has no prediabetes category; >=200 mg/dL counts only together with classic symptoms.
    "RBG": {"unit": "mg/dL", "bands": [(70, "low"), (200, "below_diagnostic_threshold"), (INF, "diabetes_range_if_symptomatic")]},
    "OGTT_2hr": {"unit": "mg/dL", "bands": [(140, "normal"), (200, "prediabetes_range"), (INF, "diabetes_range")]},
    # Creatinine ranges differ by sex/lab: use the lab's own reference interval when available.
    "Serum_Creatinine": {"unit": "mg/dL", "bands": [(1.3, "normal"), (INF, "elevated")]},
    # ADA: UACR 30-299 mg/g moderately increased, >=300 mg/g severely increased.
    "Urine_Albumin": {"unit": "mg/g", "bands": [(30, "normal"), (300, "microalbuminuria"), (INF, "macroalbuminuria")]},
    "LDL": {"unit": "mg/dL", "bands": [(100, "optimal"), (160, "borderline"), (INF, "high")]},
    # HDL goal is sex-specific (men >=40, women >=50 mg/dL); 40 is used as the general cut-off.
    "HDL": {"unit": "mg/dL", "bands": [(40, "low"), (INF, "acceptable")]},
    "Triglycerides": {"unit": "mg/dL", "bands": [(150, "normal"), (200, "borderline"), (500, "high"), (INF, "very_high")]},
}


def _same(v):
    return v


_GLUCOSE = {"mg/dl": _same, "mmol/l": lambda v: v * 18.016}
_CHOL = {"mg/dl": _same, "mmol/l": lambda v: v * 38.67}

# unit (normalised: lowercase, no spaces) -> converter into the canonical unit
CONVERSIONS = {
    "HbA1c": {"%": _same, "mmol/mol": lambda v: v / 10.929 + 2.15},
    "FBG": _GLUCOSE,
    "RBG": _GLUCOSE,
    "OGTT_2hr": _GLUCOSE,
    "Serum_Creatinine": {"mg/dl": _same, "umol/l": lambda v: v / 88.4},
    "Urine_Albumin": {"mg/g": _same, "mg/gcreat": _same, "mg/gcr": _same, "ug/mg": _same,
                      "mg/mmol": lambda v: v * 8.84},
    "LDL": _CHOL,
    "HDL": _CHOL,
    "Triglycerides": {"mg/dl": _same, "mmol/l": lambda v: v * 88.57},
}

EXTRACTION_SYSTEM_PROMPT = """You are the Report Analysis Agent for diabetes laboratory reports.
Extract these parameters if present in the report text:
HbA1c, FBG, RBG, OGTT_2hr, Serum_Creatinine, Urine_Albumin, LDL, HDL, Triglycerides
Return a flat JSON object where each key maps to {"value": <number>, "unit": "<unit exactly as printed in the report, or null>"}.
Copy numbers exactly as printed. Never estimate, infer or calculate a value that is not printed.
Omit any parameter that is not present. Return ONLY valid JSON, no markdown fences, no explanation."""


def _norm_unit(unit):
    if not unit:
        return None
    return str(unit).strip().lower().replace(" ", "").replace("\u00b5", "u").replace("\u03bc", "u")


def _appears_in_text(value: float, text: str) -> bool:
    """True if the number is literally present in the report text (hallucination guard)."""
    candidates = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    candidates |= {c.replace(".", ",") for c in candidates}
    for c in candidates:
        if re.search(rf"(?<![\d.,]){re.escape(c)}(?!\d|[.,]\d)", text):
            return True
    return False


def _split_value_unit(raw):
    """Accepts {"value": 7.2, "unit": "%"} or a bare number/string. Returns (float|None, unit|None)."""
    unit = None
    if isinstance(raw, dict):
        unit = raw.get("unit")
        raw = raw.get("value")
    try:
        return float(str(raw).replace(",", ".").strip().rstrip("%")), unit
    except (TypeError, ValueError):
        return None, unit


def convert_to_canonical(param: str, value: float, unit):
    """Returns the value in the canonical unit, or None if the unit is not supported."""
    unit_n = _norm_unit(unit)
    if unit_n is None:
        return value  # no unit printed: assume the canonical unit
    converter = CONVERSIONS.get(param, {}).get(unit_n)
    return converter(value) if converter else None


def classify_value(param: str, value: float) -> str:
    spec = REFERENCE_BANDS.get(param)
    if not spec:
        return "unknown"
    for upper, label in spec["bands"]:
        if value < upper:
            return label
    return spec["bands"][-1][1]


def extract_values(report_text: str) -> dict:
    try:
        result = llm_client.generate(report_text, system_instruction=EXTRACTION_SYSTEM_PROMPT, temperature=0.0)
    except Exception:  # noqa: BLE001
        return {}
    return extract_json(result)


def analyze_report(report_text: str) -> dict:
    """Returns {param: {value, unit, classification[, reported_as]}} for every trustworthy parameter."""
    interpretation = {}
    for param, raw in extract_values(report_text).items():
        if param not in REFERENCE_BANDS:
            continue
        value, unit = _split_value_unit(raw)
        if value is None or not _appears_in_text(value, report_text):
            continue  # missing, unparseable, or not actually printed in the report
        canonical_unit = REFERENCE_BANDS[param]["unit"]
        converted = convert_to_canonical(param, value, unit)
        if converted is None:
            interpretation[param] = {"value": value, "unit": unit, "classification": "unit_not_supported"}
            continue
        entry = {
            "value": round(converted, 2),
            "unit": canonical_unit,
            "classification": classify_value(param, converted),
        }
        if _norm_unit(unit) not in (None, _norm_unit(canonical_unit)):
            entry["reported_as"] = f"{value:g} {unit}"
        interpretation[param] = entry
    return interpretation
