# Anchors optimization report

## Scope
This update focuses on the **AnchorText** path and refines the prior optimization work with:
- predictor invocation accounting,
- chunked/streamed text perturbation prediction,
- explicit cache byte accounting,
- stress benchmarking for cache growth,
- deterministic behavior and regression tests.

## Main AnchorText hotspots identified
1. Large temporary perturbation string arrays in sampling/predict loops.
2. Predictor called with unbounded arrays (or many tiny calls after naive chunking).
3. Missing separation between invocation count vs samples predicted.
4. Large `AnchorBaseBeam` cache growth (`state['data']` / `state['labels']`) in long runs.

## Code changes

### 1) Instrumentation extensions
Added instrumentation fields:
- `predictor_invocations`: number of `predictor(xs)` calls.
- `predicted_samples_total`: total perturbations evaluated by predictor.
- `cached_state_data_bytes`
- `cached_state_labels_bytes`
- `cached_state_shapes`

Backward-compatible existing fields remain (`model_calls`, `perturbation_samples_evaluated`).

### 2) Streaming/batching on AnchorText
- `AnchorText.compare_labels` now predicts in chunks using `max_perturbation_batch_size`.
- `AnchorText.sampler` supports **streamed prediction** mode (`stream_text_sampling=True`):
  - generates perturbations in chunks,
  - predicts chunk immediately,
  - discards chunk strings,
  - keeps only masks/labels + bounded examples.
- For coverage-only requests (`compute_labels=False`), uses mask-only generation where available (`sample_masks`) to avoid building raw perturbed strings.

### 3) Text sampler performance/memory updates
- sampler-local RNG (`np.random.default_rng`) via `set_seed`.
- replaced global `np.random.*` calls.
- mask matrices are `uint8`.
- replaced `np.apply_along_axis` text joining with iterator-based `np.fromiter` joins.

### 4) Cache robustness
- Added robust metadata assembly fallback for budget-truncated resampling (no `KeyError` if covered examples are absent).

### 5) Adaptive budget
- Adaptive batch sizing logic is integrated in precision-sampling loop and can be enabled with `adaptive_budget=True`.
- Under evaluated workloads, this produced **parity** (not degradation) in anchor-found rate under capped budgets.

---

## Benchmarks

## A) Synthetic text workload (fast predictor; isolates text path overhead)
Configuration:
- `sampling_strategy='unknown'`
- `spacy.blank('en')`
- same seed / same inputs
- batches: 10 and 200

Results:

| config | batch | mean time/expl (s) | predictor_invocations | predicted_samples_total | peak RSS (MB) | anchor-found rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline (`stream_text_sampling=False`) | 10 | 0.0460 | 26.4 | 1357.8 | 594.2 | 1.00 |
| baseline (`stream_text_sampling=False`) | 200 | 0.0431 | 26.4 | 1356.2 | 594.2 | 1.00 |
| optimized (`memory_saver_mode=True`, `max_perturbation_batch_size=32`, `max_total_samples=900`, `stream_text_sampling=True`) | 10 | 0.0365 | 52.0 | 926.0 | 594.2 | 1.00 |
| optimized (`memory_saver_mode=True`, `max_perturbation_batch_size=32`, `max_total_samples=900`, `stream_text_sampling=True`) | 200 | 0.0359 | 52.4 | 924.4 | 594.2 | 1.00 |

Interpretation:
- `predicted_samples_total` reduced materially under cap.
- wall time improved.
- invocation count increased due intentional chunking + cap control; this is visible explicitly now via split metrics.

## B) Stress benchmark (cache growth exposure)
Configuration:
- 50 explanations,
- `coverage_samples=5000`, `beam_size=4`, `threshold=0.95`, `batch_size=64`, `min_samples_start=128`.

Results:

| config | mean time/expl (s) | peak RSS (MB) | max cached_state_data_bytes | max cached_state_labels_bytes | predictor_invocations | predicted_samples_total |
|---|---:|---:|---:|---:|---:|---:|
| stress baseline (`memory_saver_mode=False`) | 0.8650 | 594.1 | 6,400,000 | 5,120,000 | 615.2 | 39,520.4 |
| stress memory_saver (`memory_saver_mode=True`) | 0.9803 | 594.1 | 0 | 0 | 819.5 | 46,730.3 |

Key point:
- cache bytes become strictly bounded (`0`) with `memory_saver_mode=True`, removing growth from global state arrays.

## C) Adaptive budget under capped samples
Configuration:
- capped samples (`max_total_samples=300`) and same seed/inputs.

Observed:
- `naive_cap`: anchor-found rate `0.50`
- `adaptive_budget=True`: anchor-found rate `0.50`

Interpretation:
- no degradation under tested capped workload.
- adaptive mode remains available and can be tuned per workload.

---

## Transformer predictor benchmark
A full HuggingFace transformer benchmark was attempted but blocked in this environment by external download restriction (`huggingface.co` proxy 403).

### Reproduction command (unrestricted env)
Use the command from this repo with:
- `pipeline(..., batch_size=<gpu-safe>)`
- compare baseline vs optimized with same seed.

### Recommended predictor wrapper (PyTorch)

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BatchedTorchPredictor:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    def __call__(self, texts, batch_size=32):
        preds = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                logits = self.model(**enc).logits
                cls = torch.argmax(logits, dim=1).detach().cpu().numpy()
                preds.append(cls)
        return np.concatenate(preds, axis=0)
```

Notes:
- set AnchorText `max_perturbation_batch_size` to match predictor batch size.
- keep tensors detached and moved to CPU quickly.

---

## Commands used (core)

```bash
python -m compileall alibi/explainers/anchors/anchor_base.py \
  alibi/explainers/anchors/anchor_text.py \
  alibi/explainers/anchors/text_samplers.py \
  alibi/explainers/anchors/language_model_text_sampler.py \
  alibi/explainers/tests/test_anchor_text_optimization_unittest.py

python -m unittest alibi.explainers.tests.test_anchor_text_optimization_unittest -v
```
