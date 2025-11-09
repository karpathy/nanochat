**product requirements document: interactive logit-lens explorer for nanochat d32 (augmented)**

**1. overview**
real-time, inference-only demo exposing *emergent reasoning crystallization* across all 33 layers of nanochat d32. users scrub through layers to see token-level decisions unfold, tracing how safety, style, and factuality coalesce. supports live comparison vs. final outputs, emphasizing hidden dynamics often masked in static outputs. optionally, per-head or residual trajectory overlays can be toggled to inspect micro-structures.

**2. problem statement**
traditional llm demos obfuscate the *process*, only showing the outcome. this prevents:

* mapping emergent behaviors to transformer depth
* discovering latent alignment or hallucination triggers
* tracing creative vs. deterministic token stabilization
  researchers need intuitive, interactive tools to isolate *when*, *where*, and *how* a model commits to a prediction without costly retraining or probing.

**3. solution**
capture hidden states per layer in a single forward pass, project via `lm_head`, greedily decode, and stream results to an interactive timeline. layer scrubbing dynamically updates output text, with:

* diff highlighting vs. final layer
* token confidence stripes showing rank volatility
* hover tooltips for per-layer top-k token insights
* optional residual trajectory & per-head decomposition overlays

**4. core features**

*4.1 layer scrubber & live diff*

* vertical slider (0-32) updates decoded text in <50ms
* stable tokens gray, changing tokens red
* example: "the bomb" → "the device" as safety alignment activates

*4.2 token confidence stripe*

* 32-segment horizontal bar beneath each token showing rank stabilization per layer
* color gradient yellow→blue; creative/fuzzy tokens wobble longer
* enables quick identification of depth where decisions lock

*4.3 per-layer top-3 token tooltip*

* hover token to reveal top-3 predictions at that layer
* visualize semantic drift: e.g., "cat" → "feline" → "creature" → "cat"

*4.4 optional enhancements (toggleable)*

* **per-head logit lens**: view each of 8 heads per layer via heatmap
* **residual trajectory pca**: animate 1280-d trajectory in 2D
* **causal patching sandbox**: test interventions on intermediate states

**5. technical requirements**

backend (fastapi + nanochat/gpt.py)

hook forward() to return list of hidden states

project residuals through lm_head per layer

greedy decode 33 sequences concurrently; stream via websocket

latency: <200ms end-to-end for 128 tokens on rtx 4090

caching optional per session to reduce repeated computation

frontend (raw html + alpine.js + plotly.js)

vertical slider built with alpine.js to scrub layers

token diff highlighting implemented in raw html/css

incremental websocket layer updates bound via alpine.js

hover tooltips for per-layer top-3 token predictions using alpine directives

optional overlays (per-head/residual trajectory) toggleable

responsive ≥768px; mobile optional

**6. non-functional requirements**

* inference-only; no finetuning or model downloads
* deterministic: same seed → identical layer traces
* stateless; server memory only, no persistent storage
* target 5 concurrent users per rtx 4090

**7. success metrics**

* engagement: avg session >3 min, >20 layers scrubbed
* insight: ability to pinpoint layer where factual answer stabilizes (e.g., “paris” at layer 12 ±2)
* latency: 95th percentile <100ms
* optional: user-triggered per-head/residual exploration increases engagement by +15%

**8. engineering effort** (~8-10h)

* backend hooks & caching: 2h
* fastapi streaming: 1.5h
* react ui + diff + tooltip: 3h
* polish/latency optimizations: 1.5h
* optional overlays (per-head/pca/causal): 1-2h

**9. future enhancements**

* per-head logit lens heatmaps
* residual trajectory visualization
* causal patching sandbox
* token influence ranking & interactive rollback
* integration with multi-turn trace playback

**10. out-of-scope**

* training, fine-tuning, or lens optimization
* multi-turn conversation state management
* model distillation/compression
* external datasets or embedding lookups

---

i can also draft a **visual mockup + interaction flow diagram** that would make this PRD *actually legible for a designer*—would you want me to do that next?
