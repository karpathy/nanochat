# CORE boundary fixtures

These small excerpts come from the evaluation bundle used by `scripts/base_eval.py`:
[eval_bundle.zip](https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip),
SHA-256 `90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200`.
The fixture JSON records dataset paths, zero-based row indices, and the sampling
metadata needed to reproduce the configured task without downloading data in tests.

- `core_arc_easy_prefix.json` contains ARC-Easy row 254 and its ten selected
  few-shot examples. The AI2 Reasoning Challenge dataset is credited to Peter
  Clark and colleagues at the Allen Institute for Artificial Intelligence.
  The [publisher's dataset card](https://huggingface.co/datasets/allenai/ai2_arc)
  identifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  This fixture's dataset text remains under that license.
- `core_schema_shared_context.json` contains Winograd Schema Challenge rows 26
  and 40 in the bundle's context/continuation representation. The collection is
  credited to Ernest Davis, Leora Morgenstern, and Charles Ortiz. The
  [authors' collection page](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html)
  licenses it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  This fixture's dataset text remains under that license.

The excerpts retain the bundle's rendered question/context and answer fields;
they are selected and wrapped with provenance metadata for regression testing.
They are not new benchmark results or evidence of pretrained-model quality.
The surrounding test code is separate from this third-party dataset text.
