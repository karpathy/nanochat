#!/bin/bash
# Shared stage helpers for runs/runcpu.sh and runs/speedrun.sh (source, don't execute).
# Provides: log, fmt_dur, stage_done, pipeline_map, run_stage, FORCE_STAGES parsing,
#           health_strip, print_knob_diff, write_last_run_json, print_run_tip.
# Echo contract: docs/plan/2026-11-13-fix-forward-mac.md ("Echo contract"); legend: docs/RUNNING.md.
# Requires the caller to have set NANOCHAT_BASE_DIR and PROFILE, and `set -euo pipefail`.
# Color and plaque helpers (gated on TTY and NO_COLOR).

# True if color should be applied (stdout is TTY and NO_COLOR unset). NANOCHAT_PLAIN is art-only.
_term_color_enabled() {
    [ -z "${NO_COLOR:-}" ] && [ -t 1 ]
}

# Paint text with color for a role (START/DONE/SKIP/FAIL/FORCE).
_paint() {
    local role="$1" text="$2"
    _term_color_enabled || { echo "$text"; return; }
    case "$role" in
        START|FORCE) printf "\033[1;36m%s\033[0m\n" "$text" ;;
        DONE) printf "\033[1;32m%s\033[0m\n" "$text" ;;
        SKIP) printf "\033[2;33m%s\033[0m\n" "$text" ;;
        FAIL) printf "\033[1;31m%s\033[0m\n" "$text" ;;
        *) echo "$text" ;;
    esac
}

# Print ASCII plaque for a stage (suppressed if NANOCHAT_PLAIN=1).
_print_plaque() {
    local stage="$1" kind="$2" subtitle="$3"
    [ "${NANOCHAT_PLAIN:-}" = "1" ] && return 0
    python3 -m nanochat.stage_art plaque --stage "$stage" --kind "$kind" --subtitle "$subtitle" 2>/dev/null \
        || echo "WARN: plaque not rendered for stage=$stage kind=$kind" >&2
}

STAGE_KNOWN="tok_train tok_eval base_train base_eval chat_sft chat_eval"
STAGE_ELAPSED_DIR="$NANOCHAT_BASE_DIR/.stage_elapsed"
SAVED_TOTAL=0

# Run record: one span line per stage/hook in $NANOCHAT_RUN_DIR/events.jsonl. No-op when unset.
record_span() {  # stage status dur_s hook_s start_epoch  (one span per stage; the module writes it)
    [ -n "${NANOCHAT_RUN_DIR:-}" ] || return 0
    python3 -m nanochat.run_record span "$NANOCHAT_RUN_DIR" "$@" || echo "WARN: run span not recorded" >&2
}

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

# Prints --resume-from-step=N when a fully-committed checkpoint exists, else nothing.
# A forced stage always starts from step 0. A probe failure is reported, then starts from 0.
resume_flag() {  # stage checkpoint_dir [world_size]
    stage_forced "$1" && return 0
    local step
    # Import-time logs share stdout with the step, so keep only the last line.
    step="$("${PYTHON:-python}" -m nanochat.checkpoint_manager find-resume-step "$2" "${3:-1}" | tail -n 1)" \
        || { echo "WARN: resume probe failed for $2; starting $1 from step 0" >&2; return 0; }
    case "$step" in
        -1) return 0 ;;
        ''|*[!0-9]*) echo "WARN: resume probe returned non-step output for $2; starting $1 from step 0" >&2; return 0 ;;
    esac
    [ "$step" -gt 0 ] && echo "--resume-from-step=$step"
    return 0
}

# The artifact whose presence means the stage finished cleanly (file markers for train/eval).
# Train stages use .train_ok (not "any model_*.pt") so a partial/probe checkpoint cannot SKIP.
stage_path() {
    case "$1" in
        tok_train)  echo "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ;;
        tok_eval)   echo "$NANOCHAT_BASE_DIR/tokenizer/.tok_eval_ok" ;;
        base_train) echo "$NANOCHAT_BASE_DIR/base_checkpoints/.train_ok" ;;
        base_eval)  echo "$NANOCHAT_BASE_DIR/base_checkpoints/.eval_ok" ;;
        chat_sft)   echo "$NANOCHAT_BASE_DIR/chatsft_checkpoints/.train_ok" ;;
        chat_eval)  echo "$NANOCHAT_BASE_DIR/chatsft_checkpoints/.eval_ok" ;;
    esac
}

stage_done() {
    local p
    p="$(stage_path "$1")"
    [ -f "$p" ]
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
    bytes="$(du -sk "$path" 2>/dev/null | awk '{printf "%d\n", $1 * 1024}')" || bytes=""
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
    local mem="unknown" ckpt="n/a" shards="n/a" n needed="${SHARDS_NEEDED:-}" p=""
    mem="$(mem_snapshot 2>/dev/null | sed 's/^mem: //')" || mem="unknown"
    # Keep find stdout even when a root is missing (pipefail would otherwise wipe p).
    p="$(find "$NANOCHAT_BASE_DIR/base_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints" \
        -name 'model_*.pt' -print 2>/dev/null | head -n 1 || true)" || true
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
    line="$(python3 -m nanochat.last_run diff \
        --path "$NANOCHAT_BASE_DIR/last_run.json" \
        --profile "$1" --depth "$2" \
        --device-batch-size "$3" --max-seq-len "$4" 2>/dev/null || true)"
    [ -n "$line" ] && echo "  $line"
    return 0
}

# Persist knobs after successful ALL DONE. Args: profile depth batch seq.
write_last_run_json() {
    python3 -m nanochat.last_run write \
        --path "$NANOCHAT_BASE_DIR/last_run.json" \
        --profile "$1" --depth "$2" \
        --device-batch-size "$3" --max-seq-len "$4" 2>/dev/null || true
    return 0
}

# Exactly one tip: line per script invocation (call once at script start).
# Text SSOT: nanochat/run_tip.py (unit-tested).
print_run_tip() {
    local tip label rest
    tip="$(python3 -m nanochat.run_tip 2>/dev/null || true)"
    [ -n "$tip" ] || return 0
    # run_tip already starts with its own "tip:" token; colour that token only.
    label="${tip%%:*}:"
    rest="${tip#*:}"
    _term_color_enabled && label="$(_paint START "$label")"
    echo "  $label$rest"
    return 0
}

# pipeline_map "tok:tok_train base:base_train eval:base_eval ..." -> one line, first pending stage is ▶
pipeline_map() {
    local out="" seen_pending=0 item label stage mark prefix
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
    prefix="pipeline:"
    _term_color_enabled && prefix="$(_paint START "$prefix")"
    echo "$prefix ${out% }"
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
            local skip_line="  $saved_note · ETA: n/a (skipped)"
            _paint "SKIP" "$skip_line"
            return 0
        fi
    fi

    _print_plaque "$stage" start "$(stage_purpose "$stage")"
    local eta_note
    case "$stage" in
        base_train|chat_sft) eta_note="ETA after first train step" ;;
        *) eta_note="ETA: n/a" ;;
    esac
    _paint "START" "$(log "START stage=$stage profile=$PROFILE${knobs:+ $knobs} $eta_note")"
    # One packed enrichment line (purpose + health) — Phase 1 sparsity discipline.
    echo "  purpose: $(stage_purpose "$stage") · $(health_strip)" || true
    # Learn line after purpose (empty string suppresses the output).
    local learn_line
    learn_line="$(python3 -m nanochat.glossary learn --stage "$stage" 2>/dev/null)" \
        || { echo "WARN: learn line not rendered for stage=$stage" >&2; learn_line=""; }
    [ -n "$learn_line" ] && echo "  $learn_line"

    # Drop stale success markers before a real run so a crashed FORCE/redo cannot SKIP as "done".
    case "$stage" in
        tok_eval|base_train|base_eval|chat_sft|chat_eval)
            rm -f "$(stage_path "$stage")"
            ;;
    esac

    local t0=$SECONDS rc=0 elapsed t0_epoch
    t0_epoch=$(date +%s)
    NANOCHAT_STAGE_BRIEFED=1 "$@" || rc=$?
    elapsed=$((SECONDS - t0))

    if [ "$rc" -ne 0 ]; then
        record_span "$stage" fail "$elapsed" 0 "$t0_epoch"
        _print_plaque "$stage" fail "$(stage_purpose "$stage")"
        _paint "FAIL" "$(log "FAIL stage=$stage elapsed=$(fmt_dur "$elapsed") hint=exit code $rc; fix and re-run the same command (finished stages skip)")"
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

    _print_plaque "$stage" done "$(stage_purpose "$stage")"
    _paint "DONE" "$(log "DONE stage=$stage elapsed=$(fmt_dur "$elapsed") cumulative=$(fmt_dur "$SECONDS")")"

    # Run post-stage hook if set (e.g., Modal commit coordination). Its time goes on the stage span.
    local hook_s=0
    if [ -n "${NANOCHAT_POST_STAGE_HOOK:-}" ]; then
        local hook_t0=$SECONDS
        if eval "$NANOCHAT_POST_STAGE_HOOK"; then
            hook_s=$((SECONDS - hook_t0))
        else
            hook_s=$((SECONDS - hook_t0))
            record_span "$stage" hook_fail "$elapsed" "$hook_s" "$t0_epoch"
            _print_plaque "$stage" fail "$(stage_purpose "$stage")"
            _paint "FAIL" "$(log "FAIL stage=$stage reason=NANOCHAT_POST_STAGE_HOOK failed")"
            # Delete marker for pure marker stages; for tok_train, don't delete the real artifact
            case "$stage" in
                tok_eval|base_train|base_eval|chat_sft|chat_eval)
                    rm -f "$(stage_path "$stage")"
                    echo "  Deleted marker $(stage_path "$stage") due to hook failure" >&2
                    exit 1
                    ;;
                tok_train)
                    echo "  Hook failed; re-run with FORCE_STAGES=tok_train to retry" >&2
                    exit 1
                    ;;
                *)
                    exit 1
                    ;;
            esac
        fi
    fi

    record_span "$stage" ok "$elapsed" "$hook_s" "$t0_epoch"

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
