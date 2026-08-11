"""LLM threat-report generation (Qwen2.5-Instruct, 4-bit NF4). GPU-gated; falls back to a
cached report when run locally without CUDA."""
from __future__ import annotations
import json
import os
import re
from functools import lru_cache
from typing import Dict, Any

import torch

from . import config


GROUP_HINTS = {
    "byte_histogram":         "byte-frequency distribution of the raw file",
    "byte_entropy_histogram": "per-region Shannon entropy (packing / encryption indicator)",
    "strings":                "statistics over printable ASCII/UTF-16 strings (URLs, paths, MZ headers)",
    "general_file_info":      "top-level PE metadata (size, has_debug, has_signature, vsize)",
    "header_info":            "COFF + optional-header fields (timestamp, machine, subsystem, characteristics)",
    "section_info":           "PE section table properties (name, size, entropy, characteristics)",
    "imports":                "hashed Windows API imports (which DLLs / functions the file calls)",
    "exports":                "hashed exported symbol names",
    "data_directories":       "size / VA of PE data-directory entries (TLS, IAT, resources, etc.)",
}

BEHAVIOUR_CLUSTERS = [
    "packing-or-compression", "import-heavy", "import-light-or-obfuscated",
    "section-layout-anomaly", "resource-or-data-directory-anomaly",
    "header-metadata-anomaly", "string-profile-anomaly", "insufficient-evidence",
]

# An earlier iteration of this prompt produced reports naming specific malware
# families, inventing antivirus labels, and citing features absent from the
# attribution set (5 of 8 sampled reports were affected). The constraints below
# plus the post-generation validator reduced that to 0 of 8.
SYSTEM_INSTR = (
    "You are a senior malware analyst writing a concise triage report for a SOC.\n"
    "You must obey these constraints without exception:\n"
    "1. NEVER name a specific malware family, strain, campaign or threat actor "
    "(e.g. Emotet, TrickBot, WannaCry). You have no evidence for attribution.\n"
    "2. NEVER invent an antivirus detection label (e.g. \"Trojan.Win32.X\").\n"
    "3. ONLY reference features that appear in the list provided. Never cite a "
    "feature index or name that is not listed.\n"
    "4. NEVER assert a specific behaviour (C2 communication, persistence, "
    "credential theft, lateral movement, exfiltration) unless a listed feature "
    "directly supports it. Say \"not determinable from these features\" instead.\n"
    "5. Features listed as REDUCING the malicious score are evidence AGAINST "
    "maliciousness. Never present them as incriminating.\n"
    "6. If the evidence is weak or ambiguous, say so plainly. An honest "
    "\"insufficient evidence\" is more useful to a SOC than a confident guess."
)

FAMILY_BLOCKLIST = [
    "emotet", "trickbot", "ramnit", "wannacry", "zeus", "conti", "lockbit",
    "qakbot", "qbot", "dridex", "ryuk", "mirai", "agenttesla", "agent tesla",
    "formbook", "redline", "njrat", "darkcomet", "cobalt strike", "metasploit",
    "revil", "maze", "petya", "notpetya", "gandcrab", "cerber", "locky",
    "dharma", "stuxnet", "zloader", "icedid", "bazarloader", "raccoon",
    "vidar", "azorult", "remcos", "nanocore", "asyncrat", "gh0st", "plugx",
]
_AV_LABEL_RE = re.compile(
    r"\b(?:Trojan|Worm|Backdoor|Packed|Ransom|Adware|Spyware|Dropper|Downloader|Virus)"
    r"[.:][A-Za-z0-9_.\-]{2,}", re.I)
_FEAT_CITE_RE = re.compile(r"\b([a-z_]+)\[(\d+)\]")


def validate(text: str, entry: Dict[str, Any]) -> list[str]:
    """Flag fabrications: named families, invented AV labels, features not in the
    file's own top-10, and a missing behaviour-cluster label."""
    issues, low = [], text.lower()
    for fam in FAMILY_BLOCKLIST:
        if re.search(rf"\b{re.escape(fam)}\b", low):
            issues.append(f"named malware family: {fam}")
    for m in _AV_LABEL_RE.findall(text):
        issues.append(f"invented AV label: {m}")
    allowed = {r["feature"] for r in entry["features"]}
    for g, i in _FEAT_CITE_RE.findall(text):
        if f"{g}[{i}]" not in allowed:
            issues.append(f"cited feature not in top-10: {g}[{i}]")
    if not any(c in low for c in BEHAVIOUR_CLUSTERS):
        issues.append("no behaviour-cluster label from the closed list")
    return issues


def build_messages(entry: Dict[str, Any]) -> list[dict]:
    pos = [r for r in entry["features"] if r["shap"] > 0]
    neg = [r for r in entry["features"] if r["shap"] <= 0]

    def block(rows):
        if not rows:
            return "  (none in the top 10)"
        return "\n".join(
            f'  {r["feature"]:28s} SHAP {r["shap"]:+.4f}   '
            f'[{r["group"]} — {GROUP_HINTS.get(r["group"], "")}]'
            for r in rows)

    user_msg = (
        f'A static-analysis classifier (FT-Transformer over EMBER PE features) '
        f'scored a Windows executable at probability {entry["probability"]:.4f} '
        f'of being malicious.\n\n'
        f'FEATURES INCREASING THE MALICIOUS SCORE (evidence FOR):\n'
        f'{block(pos)}\n\n'
        f'FEATURES REDUCING THE MALICIOUS SCORE (evidence AGAINST):\n'
        f'{block(neg)}\n\n'
        f'Write a triage report in three short paragraphs:\n'
        f'  1. What PE-level characteristics do the contributing features indicate? '
        f'Refer to feature groups, not raw indices. Note explicitly if the '
        f'evidence-against materially weakens the case.\n'
        f'  2. Classify the observed pattern using EXACTLY ONE label from this '
        f'closed list, and justify it from the listed features only:\n'
        f'     {", ".join(BEHAVIOUR_CLUSTERS)}\n'
        f'     Do NOT name a malware family.\n'
        f'  3. Recommended SOC triage actions proportionate to this evidence.\n'
        f'Under 250 words. Do not speculate beyond the features listed above.'
    )
    # Folded into the user turn: some chat templates reject a `system` role, and
    # it is semantically identical for single-turn generation.
    return [{"role": "user", "content": SYSTEM_INSTR + "\n\n" + user_msg}]


@lru_cache(maxsize=1)
def _load_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    hf_token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(config.GENERATOR_ID, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        config.GENERATOR_ID,
        quantization_config=bnb,
        device_map="auto",
        token=hf_token,
    )
    return tok, llm


def _cached_fallback(entry: Dict[str, Any]) -> str | None:
    """Return a cached report that matches this entry's top groups, if any."""
    path = config.ARTIFACTS_DIR / config.REPORTS_CACHE
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text())
    except Exception:
        return None
    sig = tuple(sorted({r["group"] for r in entry["features"][:3]}))
    for item in cache:
        other_sig = tuple(sorted({r["group"] for r in item["entry"]["features"][:3]}))
        if other_sig == sig:
            return (
                "**Cached report (matched by top-3 feature groups — "
                f"{config.GENERATOR_NAME} not loaded in this environment):**\n\n" + item["report"]
            )
    return None


def generate(entry: Dict[str, Any], max_new_tokens: int = 400) -> tuple[str, float]:
    """Return (report_text, seconds_elapsed). Uses cache when no GPU available."""
    import time
    if not torch.cuda.is_available():
        cached = _cached_fallback(entry)
        if cached is not None:
            return cached, 0.0
        return (
            f"_{config.GENERATOR_NAME} requires a CUDA GPU. Deploy to the HF Space (ZeroGPU) "
            "to generate reports live, or drop `reports_cache.json` into "
            "`assets/` for an offline preview._"
        ), 0.0
    tok, llm = _load_llm()
    msgs = build_messages(entry)
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(llm.device)
    prompt_len = inputs["input_ids"].shape[-1]
    t0 = time.time()
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.3, top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    # Surface any fabrication the prompt constraints failed to prevent, rather
    # than presenting an unchecked generation as authoritative.
    issues = validate(text, entry)
    if issues:
        text += ("\n\n---\n\n**⚠️ Automated validation flagged this report:**\n"
                 + "\n".join(f"- {i}" for i in issues)
                 + "\n\nTreat the flagged claims as unsupported by the SHAP evidence.")
    return text, time.time() - t0
