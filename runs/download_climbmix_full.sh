#!/bin/bash
set -euo pipefail

DEST="${DEST:-/u201/l39gu/projects/climbmix-400b-shuffle}"
SRC="${SRC:-/u201/l39gu/nanoknow-climbmix/corpus/climbmix-400b-shuffle}"
HF_BIN="${HF_BIN:-/tmp/${USER:-nanochat}/nanochat-smoke-venv/bin/hf}"

mkdir -p "$DEST"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Destination: $DEST"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing shards from: $SRC"

python - <<'PY'
import os
from pathlib import Path

dest = Path(os.environ["DEST"])
src = Path(os.environ["SRC"])
dest.mkdir(parents=True, exist_ok=True)

linked = 0
existing = 0
for source in sorted(src.glob("*")):
    if source.name.startswith(".cache"):
        continue
    target = dest / source.name
    if target.exists():
        existing += 1
        continue
    try:
        os.link(source, target)
    except OSError:
        os.symlink(source, target)
    linked += 1

print(f"Reused files: linked={linked}, already_present={existing}")
PY

if [ ! -x "$HF_BIN" ]; then
    echo "hf CLI not found at $HF_BIN"
    echo "Set HF_BIN to an executable hf CLI path."
    exit 1
fi

export HF_HOME="${HF_HOME:-/tmp/${USER:-nanochat}/hf-home}"
mkdir -p "$HF_HOME"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Hugging Face download"
"$HF_BIN" download karpathy/climbmix-400b-shuffle \
    --repo-type dataset \
    --local-dir "$DEST"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download complete"
python - <<'PY'
import os
from pathlib import Path

dest = Path(os.environ["DEST"])
parquets = sorted(dest.glob("shard_*.parquet"))
total = sum(p.stat().st_size for p in parquets)
print(f"parquet_count={len(parquets)}")
print(f"total_gib={total / 1024**3:.2f}")
print(f"has_validation={(dest / 'shard_06542.parquet').exists()}")
PY
