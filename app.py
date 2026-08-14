"""Mal-D3 — live FT-Transformer malware triage over the EMBER held-out test set.

The user selects any file from a stratified slice of EMBER 2018's
held-out test set; the detector scores it, SHAP attributes the decision, and an
LLM narrates the attribution as a SOC triage report.

Detection and attribution run live on CPU. Report generation needs a GPU, so it
falls back to pre-generated text where available — the eight files carrying a
cached report are always present in the subset.

Data is static and pre-approved: EMBER only, no live sample fetching.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config, data, model as det, shap_explain, llm_report

CACHE_PATH = Path(__file__).parent / "assets" / "reports_cache.json"

st.set_page_config(page_title="Mal-D3 Triage", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading FT-Transformer detector…")
def _detector():
    return det.load_detector(device="cpu")


@st.cache_resource(show_spinner="Loading EMBER test slice…")
def _dataset():
    d = config.ARTIFACTS_DIR
    X = np.load(d / config.DEMO_X).astype(np.float32, copy=False)
    y = np.load(d / config.DEMO_Y).astype(np.int64, copy=False)
    src = np.load(d / config.DEMO_SRCIDX).astype(np.int64, copy=False)
    return X, y, src


@st.cache_data
def _scores(variant: str) -> np.ndarray:
    """Precomputed scores for the whole slice, used only to populate the browser.
    The file the user actually selects is re-scored live below."""
    return np.load(config.ARTIFACTS_DIR
                   / config.DETECTORS[variant]["scores"]).astype(np.float32)


@st.cache_data
def _cache() -> dict[int, dict]:
    if not CACHE_PATH.exists():
        return {}
    return {it["entry"]["test_idx"]: it for it in json.loads(CACHE_PATH.read_text())}


# Confidence bands the triage queue samples from. Spread deliberately across the
# decision boundary rather than taking the top five by score — a queue sorted by
# probability makes "which is riskiest" a trivial read of the first row, which
# defeats the point of asking. The low band usually surfaces a false positive.
TRIAGE_BANDS = (0.55, 0.70, 0.85, 0.95, 0.995)


@st.cache_data(show_spinner="Building triage queue…")
def _triage_queue(_scores_arr: np.ndarray, _pred: np.ndarray,
                  variant_key: str) -> list[int]:
    """Row indices of the queue: nearest flagged file to each confidence band."""
    flagged = np.where(_pred == 1)[0]
    chosen: list[int] = []
    for target in TRIAGE_BANDS:
        order = flagged[np.argsort(np.abs(_scores_arr[flagged] - target))]
        for cand in order:
            if int(cand) not in chosen:
                chosen.append(int(cand))
                break
    return chosen


variant = st.session_state.get("variant", config.DEFAULT_DETECTOR)

try:
    X, y, src_idx = _dataset()
    scores = _scores(variant)
    cache = _cache()
except FileNotFoundError as e:
    st.error(f"Missing artifact: `{e.filename}`. See `assets/` in the repo README.")
    st.stop()


# ----------------------------------------------------------------- header
st.title("🛡️ Mal-D3 — FT-Transformer Malware Triage")
st.caption(
    "Live detection and SHAP attribution over the EMBER 2018 held-out test set · "
    "reports narrated by Qwen2.5-Instruct."
)

pred = (scores >= 0.5).astype(int)
acc = float((pred == y).mean())
h1, h2, h3, h4 = st.columns(4)
h1.metric("Files loaded", f"{len(y):,}")
h2.metric("Malicious / benign", f"{int((y==1).sum()):,} / {int((y==0).sum()):,}")
h3.metric("Accuracy on this slice", f"{acc:.4f}")
h4.metric("Reports pre-generated", f"{len(cache)}")


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Detector")
    picked = st.selectbox(
        "Model", list(config.DETECTORS),
        index=list(config.DETECTORS).index(variant),
        help="Same architecture, different preprocessing. Switching re-scores the "
             "slice and recomputes attributions.",
    )
    if picked != variant:
        # lru_cache(maxsize=1) evicts the previous detector, scaler and background,
        # but CPython only returns the pages once the refcount drops and the
        # allocator compacts. Collect explicitly: the host cap is 1 GB and a
        # resident second FT plus its attention buffers is most of the headroom.
        import gc
        st.session_state["variant"] = picked
        gc.collect()
        st.rerun()
    st.caption(config.DETECTORS[variant]["blurb"])

    st.divider()
    st.header("Select a file")

    view = st.radio(
        "Show",
        ["Flagged as malicious", "All files", "Only files with a cached report",
         "Model errors"],
        index=0,
    )
    if view == "Flagged as malicious":
        mask = pred == 1
    elif view == "Only files with a cached report":
        mask = np.isin(src_idx, list(cache))
    elif view == "Model errors":
        mask = pred != y
    else:
        mask = np.ones(len(y), bool)

    lo, hi = st.slider("P(malicious) range", 0.0, 1.0, (0.0, 1.0), 0.01)
    mask &= (scores >= lo) & (scores <= hi)
    pool = np.where(mask)[0]

    st.caption(f"**{len(pool):,}** files match.")
    if len(pool) == 0:
        st.stop()

    if st.button("🎲 Pick one at random", use_container_width=True):
        st.session_state["sel"] = int(np.random.choice(pool))

    default = st.session_state.get("sel", int(pool[0]))
    if default not in pool:
        default = int(pool[0])
    labels = {int(i): f"#{int(src_idx[i])} · p={scores[i]:.4f} · "
                       f"{'malicious' if y[i] == 1 else 'benign'}"
                       f"{' · report' if int(src_idx[i]) in cache else ''}"
              for i in pool[:2000]}
    if default not in labels:
        labels[default] = f"#{int(src_idx[default])} · p={scores[default]:.4f}"
    sel = st.selectbox("File", list(labels), format_func=lambda i: labels[i],
                       index=list(labels).index(default))
    st.session_state["sel"] = int(sel)

    st.divider()
    st.caption(
        f"Detector: {variant}, trained on EMBER "
        "(540K) · Attribution: `shap.GradientExplainer` on the logit · "
        f"Generator: {config.GENERATOR_NAME} (4-bit NF4)."
    )


# ----------------------------------------------------------------- scoring
i = int(st.session_state["sel"])
orig = int(src_idx[i])
prob = float(scores[i])
truth = int(y[i])
verdict = "MALICIOUS" if prob >= 0.5 else "BENIGN"
correct = (prob >= 0.5) == bool(truth)

# Re-score the selected file live. The browser table uses precomputed values so
# the page loads instantly; this is the detector actually running, on one sample.
row_scaled = det.prepare(X[i:i + 1], variant)
live_logit, live_prob = det.score(row_scaled, variant=variant, device="cpu")

c1, c2, c3, c4 = st.columns(4)
c1.metric("EMBER test index", f"#{orig}")
c2.metric("Prediction", verdict)
c3.metric("Ground truth", "malicious" if truth else "benign")
c4.metric("Correct?", "✅ yes" if correct else "❌ no")
st.progress(min(max(prob, 0.0), 1.0), text=f"P(malicious) = {prob:.4f}")
st.caption(
    f"Re-scored live just now: logit {live_logit:+.4f} → p = {live_prob:.6f}. "
    "The table above is precomputed so the browser loads instantly; this value "
    "is the detector running on the selected file."
)


# ----------------------------------------------------------------- tabs
tab_queue, tab_shap, tab_report, tab_raw = st.tabs(
    ["🚨 Triage queue", "🧭 SHAP attribution", "📝 Threat report", "🔎 Raw details"])


with tab_queue:
    st.subheader("Open alerts awaiting triage")

    # variant_key is what st.cache_data hashes: the array args are underscored
    # and therefore excluded from the key, so without it the queue would be
    # computed once and reused for whichever detector was selected first.
    queue = _triage_queue(scores, pred, variant)

    @st.cache_data(show_spinner="Attributing queue entries…")
    def _queue_evidence(rows: tuple[int, ...], variant_key: str) -> list[dict]:
        """Top driver and evidence balance per queue entry. Cached: this is five
        SHAP passes, and they only need doing once per session."""
        out = []
        for r in rows:
            top = shap_explain.top_k_for_row(
                det.prepare(X[r:r + 1], variant), k=10, variant=variant)
            out.append({
                "top_driver": top[0]["feature"],
                "driver_group": top[0]["group"],
                "toward_malicious": sum(t["direction"] == "malicious" for t in top),
            })
        return out

    ev = _queue_evidence(tuple(queue), variant)
    qdf = pd.DataFrame({
        "alert": [f"A{n + 1}" for n in range(len(queue))],
        "file": [f"#{int(src_idx[r])}" for r in queue],
        "P(malicious)": [f"{scores[r]:.4f}" for r in queue],
        "strongest driver": [e["top_driver"] for e in ev],
        "driver group": [e["driver_group"] for e in ev],
        "evidence toward malicious": [f"{e['toward_malicious']}/10" for e in ev],
        "report": ["yes" if int(src_idx[r]) in cache else "—" for r in queue],
    })
    st.dataframe(qdf, hide_index=True, use_container_width=True)

    st.write("**Open an alert**")
    cols = st.columns(len(queue))
    for n, (col, r) in enumerate(zip(cols, queue)):
        if col.button(f"A{n + 1} · #{int(src_idx[r])}", use_container_width=True,
                      key=f"q{r}"):
            st.session_state["sel"] = int(r)
            st.rerun()



with tab_shap:
    st.subheader("Top-10 attributions (signed SHAP, logit space)")
    with st.spinner("Computing SHAP attributions…"):
        top = shap_explain.top_k_for_row(row_scaled, k=10, variant=variant)
    df = pd.DataFrame(top)
    shown = df[["rank", "feature", "group", "shap", "direction"]].copy()
    shown["shap"] = shown["shap"].map(lambda v: f"{v:+.4f}")
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.bar_chart(df[["feature", "shap"]].iloc[::-1], x="feature", y="shap",
                 horizontal=True, height=380)
    st.caption(
        "Positive SHAP pushed the detector toward *malicious*, negative toward "
        "*benign*. Computed live for this file against a balanced background "
        "drawn from the same slice."
    )

with tab_report:
    st.subheader(f"{config.GENERATOR_NAME} triage report")
    cached = cache.get(orig)
    if cached and variant != "FT-SS":
        st.info(
            f"The cached reports were generated from **FT-SS** attributions. "
            f"You are viewing **{variant}**, so the report below describes a "
            "different model's reasoning about this file."
        )
    if cached:
        st.caption(f"Pre-generated · {float(cached.get('gen_seconds', 0)):.1f} s")
        st.markdown(cached.get("report", "").strip())
        if cached.get("validation"):
            st.warning("Validator flagged: " + "; ".join(cached["validation"]))
        else:
            st.success("Passed automated fabrication checks.")
    else:
        st.info(
            "No pre-generated report for this file. Generation requires a GPU; "
            "on a CPU deployment only the eight pre-generated examples are "
            "available. Detection and attribution above are live."
        )
        if st.button("Attempt live generation"):
            with st.spinner("Generating…"):
                entry = {"probability": prob, "test_idx": orig, "features": top}
                text, secs = llm_report.generate(entry)
            st.markdown(text)
            st.caption(f"{secs:.1f} s")

with tab_raw:
    st.write({
        "dataset": "EMBER 2018 held-out test set (stratified slice)",
        "ember_test_index": orig,
        "probability": prob,
        "predicted": verdict,
        "ground_truth": "malicious" if truth else "benign",
        "correct": correct,
        "has_cached_report": orig in cache,
    })
    with st.expander("Top-10 SHAP (raw)"):
        st.json(top)


