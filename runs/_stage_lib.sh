#!/bin/bash
# Shared stage helpers for runs/runcpu.sh and runs/speedrun.sh (source, don't execute).
# Provides: log, fmt_dur, stage_done, pipeline_map, run_stage, FORCE_STAGES parsing,
#           health_strip, print_knob_diff, write_last_run_json, print_run_tip.
# Echo contract: docs/plan/2026-11-13-fix-forward-mac.md ("Echo contract"); legend: docs/RUNNING.md.
# Requires the caller to have set NANOCHAT_BASE_DIR and PROFILE, and `set -euo pipefail`.

STAGE_KNOWN="tok_train tok_eval base_train base_eval chat_sft chat_eval"
STAGE_ELAPSED_DIR="$NANOCHAT_BASE_DIR/.stage_elapsed"
SAVED_TOTAL=0

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

fmt_dur() {
    local s="$1"
    if [ "$s" -lt 60 ]; then
        echo "${s}s"
    elif [ "$s" -lt 3600 ]; then
        echo "$((s / 60))m"
    else
        echo "$((s / 3600))h$(((s % 3600) / 60))m"
    fi
}

# FORCE_STAGES=all | comma-list of stage names. Bare =1 (or any unknown token) is rejected.
FORCE_STAGES="${FORCE_STAGES:-}"
if [ -n "$FORCE_STAGES" ] && [ "$FORCE_STAGES" != "all" ]; then
    _old_ifs="$IFS"; IFS=','
    for _tok in $FORCE_STAGES; do
        case " $STAGE_KNOWN " in
            *" $_tok "*) ;;
            *)
                IFS="$_old_ifs"
                echo "ERROR: FORCE_STAGES='$FORCE_STAGES' has unknown stage '$_tok'." >&2
                echo "  Use FORCE_STAGES=all or a comma-list of: ${STAGE_KNOWN// /,}" >&2
                echo "  (bare FORCE_STAGES=1 is not supported). See docs/RUNNING.md" >&2
                exit 2
                ;;
        esac
    done
    IFS="$_old_ifs"
fi

stage_forced() {
    [ "$FORCE_STAGES" = "all" ] && return 0
    case ",$FORCE_STAGES," in
        *",$1,"*) return 0 ;;
    esac
    return 1
}

# The artifact (file or directory) whose presence means the stage already ran.
stage_path() {
    case "$1" in
        tok_train)  echo "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ;;
        tok_eval)   echo "$NANOCHAT_BASE_DIR/tokenizer/.tok_eval_ok" ;;
        base_train) echo "$NANOCHAT_BASE_DIR/base_checkpoints" ;;
        base_eval)  echo "$NANOCHAT_BASE_DIR/base_checkpoints/.eval_ok" ;;
        chat_sft)   echo "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ;;
        chat_eval)  echo "$NANOCHAT_BASE_DIR/chatsft_checkpoints/.eval_ok" ;;
    esac
}

stage_done() {
    local p
    p="$(stage_path "$1")"
    case "$1" in
        base_train|chat_sft)
            [ -n "$(find "$p" -name 'model_*.pt' -print 2>/dev/null | head -n 1)" ]
            ;;
        *)
            [ -f "$p" ]
            ;;
    esac
}

stage_purpose() {
    case "$1" in
        tok_train)  echo "learn the vocabulary (BPE tokenizer) from web text" ;;
        tok_eval)   echo "check the tokenizer's compression ratio" ;;
        base_train) echo "teach raw next-token prediction on web text" ;;
        base_eval)  echo "measure the base model (CORE metric, bits-per-byte, samples)" ;;
        chat_sft)   echo "teach chat format, tool use and multiple choice" ;;
        chat_eval)  echo "score the chat model on benchmark tasks" ;;
    esac
}

# Human-readable disk size for a file or directory (best-effort; never fails the run).
fmt_bytes() {
    local path="$1" bytes
    bytes="$(du -sk "$path" 2>/dev/null | awk '{print $1 * 1024}')" || bytes=""
    if [ -z "$bytes" ] || [ "$bytes" -eq 0 ]; then
        echo "unknown"
        return
    fi
    if [ "$bytes" -lt 1024 ]; then
        echo "${bytes}B"
    elif [ "$bytes" -lt 1048576 ]; then
        echo "$((bytes / 1024))KiB"
    elif [ "$bytes" -lt 1073741824 ]; then
        awk -v b="$bytes" 'BEGIN { printf "%.1fMiB", b/1048576 }'
    else
        awk -v b="$bytes" 'BEGIN { printf "%.1fGiB", b/1073741824 }'
    fi
}

# One-line mem snapshot at stage boundaries (approx; Darwin vs Linux).
mem_snapshot() {
    local free_note="unknown"
    if command -v vm_stat >/dev/null 2>&1; then
        # macOS: free+inactive pages * page size
        free_note="$(vm_stat 2>/dev/null | awk '
            /page size of/ { ps=$8 }
            /Pages free/ { gsub("\\.","",$3); free=$3 }
            /Pages inactive/ { gsub("\\.","",$3); inactive=$3 }
            END {
                if (ps+0 > 0) printf "~%.1fGiB free-ish", (free+inactive)*ps/1073741824
                else print "unknown"
            }')"
    elif [ -r /proc/meminfo ]; then
        free_note="$(awk '/MemAvailable:/ { printf "~%.1fGiB available", $2/1048576 }' /proc/meminfo)"
    fi
    echo "mem: $free_note"
}

# Best-effort shard count under the climbmix (or legacy) data dir. Never fails.
_count_shards() {
    local d n=0
    for d in "$NANOCHAT_BASE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_data"; do
        if [ -d "$d" ]; then
            n="$(find "$d" -maxdepth 1 -name 'shard_*.parquet' 2>/dev/null | wc -l | tr -d ' ')" || n=0
            echo "$n"
            return 0
        fi
    done
    echo "0"
}

# Health enrichment at stage START (after purpose:). Never fails the run.
# Format: health: <mem> · ckpt=<size|n/a> · shards=<n/needed|n/a>
# Optional SHARDS_NEEDED (set by caller profile); default n/a for needed.
health_strip() {
    local mem="unknown" ckpt="n/a" shards="n/a" n needed="${SHARDS_NEEDED:-}" p
    mem="$(mem_snapshot 2>/dev/null | sed 's/^mem: //')" || mem="unknown"
    p="$(find "$NANOCHAT_BASE_DIR/base_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints" \
        -name 'model_*.pt' -print 2>/dev/null | head -n 1)" || p=""
    if [ -n "${p:-}" ]; then
        ckpt="$(fmt_bytes "$(dirname "$p")")" || ckpt="n/a"
    fi
    n="$(_count_shards)" || n="0"
    if [ -n "$needed" ]; then
        shards="${n}/${needed}"
    elif [ "$n" != "0" ]; then
        shards="${n}/n/a"
    fi
    echo "health: $mem · ckpt=$ckpt · shards=$shards"
}

# Knob diff vs $NANOCHAT_BASE_DIR/last_run.json (silent if missing). Args: profile depth batch seq.
print_knob_diff() {
    local line
    line="$(python -m nanochat.last_run diff \
        --path "$NANOCHAT_BASE_DIR/last_run.json" \
        --profile "$1" --depth "$2" \
        --device-batch-size "$3" --max-seq-len "$4" 2>/dev/null || true)"
    [ -n "$line" ] && echo "  $line"
    return 0
}

# Persist knobs after successful ALL DONE. Args: profile depth batch seq.
write_last_run_json() {
    python -m nanochat.last_run write \
        --path "$NANOCHAT_BASE_DIR/last_run.json" \
        --profile "$1" --depth "$2" \
        --device-batch-size "$3" --max-seq-len "$4" 2>/dev/null || true
    return 0
}

# Exactly one tip: line per script invocation (call once at script start).
print_run_tip() {
    echo "  tip: smaller --device-batch-size uses grad accumulation to keep the same total batch"
}

# pipeline_map "tok:tok_train base:base_train eval:base_eval ..." -> one line, first pending stage is ▶
pipeline_map() {
    local out="" seen_pending=0 item label stage mark
    for item in "$@"; do
        label="${item%%:*}"; stage="${item#*:}"
        if stage_done "$stage" && ! stage_forced "$stage"; then
            mark="✓"
        elif [ "$seen_pending" -eq 0 ]; then
            mark="▶"; seen_pending=1
        else
            mark="○"
        fi
        out="$out[$label$mark] "
    done
    echo "pipeline: ${out% }"
}

# run_stage STAGE "KNOBS" "NEXT" -- command...
# Skips when the artifact exists (unless forced), echoes START/DONE/FAIL, fails fast.
run_stage() {
    local stage="$1" knobs="$2" next="$3"
    shift 4  # also drops the literal "--"

    if stage_done "$stage"; then
        if stage_forced "$stage"; then
            log "FORCE stage=$stage reason=FORCE_STAGES=$FORCE_STAGES"
        else
            local saved_note="skip-saved: unknown (no prior timing)" prev
            if [ -f "$STAGE_ELAPSED_DIR/$stage" ]; then
                prev="$(cat "$STAGE_ELAPSED_DIR/$stage")"
                SAVED_TOTAL=$((SAVED_TOTAL + prev))
                saved_note="skip-saved: ~$(fmt_dur "$prev") (last run) · cumulative_saved=~$(fmt_dur "$SAVED_TOTAL")"
            fi
            log "SKIP stage=$stage reason=artifact exists path=$(stage_path "$stage") (FORCE_STAGES=$stage to redo)"
            echo "  $saved_note · ETA: n/a (skipped)"
            return 0
        fi
    fi

    log "START stage=$stage profile=$PROFILE${knobs:+ $knobs} ETA: not yet knowable"
    echo "  purpose: $(stage_purpose "$stage")"
    # Single enrichment after purpose: (best-effort; never fail the run).
    echo "  $(health_strip)" || true

    local t0=$SECONDS rc=0 elapsed
    "$@" || rc=$?
    elapsed=$((SECONDS - t0))

    if [ "$rc" -ne 0 ]; then
        log "FAIL stage=$stage elapsed=$(fmt_dur "$elapsed") hint=exit code $rc; fix and re-run the same command (finished stages skip)"
        echo "  root_stage=$stage · cascade_suppressed=yes (later stages not run) · try: see docs/RUNNING.md"
        exit "$rc"
    fi

    # tok_eval marker: primary writer is scripts/tok_eval.py (end of main); shell touch is
    # idempotent insurance if Python exits 0 before writing (should not happen).
    if [ "$stage" = "tok_eval" ]; then
        mkdir -p "$(dirname "$(stage_path tok_eval)")"
        touch "$(stage_path tok_eval)"
    fi
    mkdir -p "$STAGE_ELAPSED_DIR"
    echo "$elapsed" > "$STAGE_ELAPSED_DIR/$stage"

    log "DONE stage=$stage elapsed=$(fmt_dur "$elapsed") cumulative=$(fmt_dur "$SECONDS")"
    # Single enrichment line (next + optional wrote: + mem snapshot).
    case "$stage" in
        tok_train|base_train|chat_sft)
            local p size
            case "$stage" in
                tok_train) p="$NANOCHAT_BASE_DIR/tokenizer" ;;
                base_train)
                    p="$(find "$NANOCHAT_BASE_DIR/base_checkpoints" -name 'model_*.pt' -print 2>/dev/null | head -n 1)"
                    if [ -n "${p:-}" ]; then p="$(dirname "$p")"; else p="$NANOCHAT_BASE_DIR/base_checkpoints"; fi
                    ;;
                chat_sft)
                    p="$(find "$NANOCHAT_BASE_DIR/chatsft_checkpoints" -name 'model_*.pt' -print 2>/dev/null | head -n 1)"
                    if [ -n "${p:-}" ]; then p="$(dirname "$p")"; else p="$NANOCHAT_BASE_DIR/chatsft_checkpoints"; fi
                    ;;
            esac
            size="$(fmt_bytes "$p")"
            echo "  next: $next · wrote: $p ($size) · $(mem_snapshot)"
            ;;
        *)
            echo "  next: $next · $(mem_snapshot)"
            ;;
    esac
}
