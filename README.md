---
title: Mal-D3 Malware Triage
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Mal-D3 — FT-Transformer + SHAP + LLM malware triage

Viva-demonstration prototype for the dissertation *LLM-based Malware Detection:
An Approach to Cybersecurity Threat Identification* (Aryan Bardhan).

Select any file from a 5,000-sample stratified slice of the EMBER 2018 held-out
test set. The detector scores it, SHAP attributes the decision to individual PE
features, and an LLM narrates that attribution as a SOC triage report.

**Detection and attribution run live.** The FT-Transformer scores the selected
file on CPU in milliseconds, and `shap.GradientExplainer` computes attributions
against a balanced background drawn from the same slice.

**Report generation requires a GPU.** On a CPU deployment the eight
pre-generated reports are served from cache; those files are always present in
the slice. With a GPU attached, reports generate live.

## Pipeline

| Stage | Component |
|---|---|
| Features | EMBER 2018 static PE features, 2,381 → 500 (variance threshold + RF Gini) |
| Preprocessing | StandardScaler fitted on the EMBER 540K training fold, frozen |
| Detector | FT-Transformer (Gorishniy et al., 2021), 30 epochs, `87a377e2…` |
| Attribution | `shap.GradientExplainer` on the logit, background from the training distribution |
| Generator | Qwen2.5-Instruct, 4-bit NF4 |

## Scope

Data is static and pre-approved: the EMBER held-out test set only. There is no
live sample fetching, no file upload, and no third-party feed. SOREL-20M is used
for the cross-dataset evaluation reported in the dissertation but is deliberately
excluded here.

Reports are constrained against fabrication: the prompt separates evidence for
and against the classification, replaces open-ended family attribution with a
closed behaviour-cluster vocabulary, and every generation is checked by a
validator for invented family names, fabricated AV labels, and citations of
features outside the attribution set.

## Artifacts

`assets/` must contain: `ft_ss_best.pt`, `standard_scaler.joblib`,
`selected_feature_indices.npy`, `feature_names.json`, `ember_demo_X500.npy`,
`ember_demo_y.npy`, `ember_demo_srcidx.npy`, `reports_cache.json`.
