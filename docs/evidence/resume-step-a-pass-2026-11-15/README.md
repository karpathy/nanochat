# Step A: data-position resume check (PASS, 2026-11-15)

Command (run with `NANOCHAT_BASE_DIR` pointed at scratch dirs, device auto-detected as mps):

    python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \
      --device-batch-size=8 --total-batch-size=16384 --eval-every=-1 --core-metric-every=-1 \
      --sample-every=-1 --num-iterations=200 --run=dummy

- control: uninterrupted 200 steps (`control_steps.txt`)
- kill run: SIGTERM after the step-7 checkpoint was written; stopped near step 10 (`kill_run_steps.txt`)
- resume: `--resume-from-step=7` (`resumed_steps.txt`)

Result: 193 overlapping steps (7-199). pq/rg mismatches: 0. Max |loss diff|: 1e-6
(the logs print 6 decimals, so this is the printed precision). See `compare_output.txt`.

Note: a first resume attempt passed `--resume-from-step=-1`, which disables resume in
`base_train`; that run started from step 0 and was discarded.
