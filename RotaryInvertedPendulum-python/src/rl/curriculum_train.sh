#!/usr/bin/env bash
# Curriculum-learning training script. 3 stages, each --resumes from the
# previous, so the policy doesn't relearn the basics every time.
#
#   Stage 1: NO DR                          (find the swing-up + balance skill)
#   Stage 2: action-lag tau ∈ [0, 30 ms]    (introduce transport delay)
#   Stage 3: action-lag tau ∈ [10, 30 ms]   (concentrate around real ≈ 20 ms)
#
# Action-lag DR is the continuous, fractional-step replacement for the
# legacy integer-step `--dr-delay-min/max` queue, which was calibrated for
# the position-mode era (~50 ms transport delay) and was teaching the
# policy to handle a much larger delay than the post-accel-mode rig
# actually has (~20 ms — measured 2026-05-20 via accel_lag_moving_probe.py).
#
# Stage-3 range rationale: the 2026-05-20 50 Hz deploy showed stage 2
# ([0, 30 ms]) BEAT stage 3 ([0, 40 ms]) on the rig — wider DR means less
# training mass concentrated near the rig's actual tau, and the policy
# became more conservative to handle scenarios that don't occur.
# Concentrating stage 3 on [10, 30 ms] keeps the curriculum widening (so
# the policy still generalises) but centres the training mass at 20 ms
# instead of 20 ms ± 10 ms uniform → policy specialises near reality
# while remaining robust to the band sim might mis-estimate by.
#
# Per the 2026-05-16 DR-sensitivity probe, every other DR dimension
# (motor accel envelope, pendulum mass/COM/friction, dt jitter, encoder
# noise, motor stiction) is benign at deploy. Stage 2/3 leave those at
# the env's defaults — they activate alongside action-lag when
# --domain-randomization is on, giving free robustness without an extra
# curriculum knob to tune.
#
# Control rate: 50 Hz validated 2026-05-20 with sustained-balance deploy
# (0.968 avg upright). The earlier "35 Hz sweet spot" finding was a
# position-mode artifact (planning thrash at high rates); accel mode
# composes commands smoothly. See docs/control_rate_selection.md.
#
# Usage:
#     ./curriculum_train.sh <run-name-prefix>
# Produces runs/<prefix>_stage{1,2,3}/, final policy at
# runs/<prefix>_stage3/best_model.zip.
#
# Run from this directory with the rotary-inverted-pendulum mamba env
# activated.
#
# Environment overrides (defaults shown):
#     CONTROL_FREQ=50
#     MAX_ACCEL_RAD_S2=150
#     STEPS_PER_STAGE=100000
#     SEED=0
#     DEVICE=cuda                          # use cpu on macOS laptop
#     DR_LAG_TAU_MIN_S2=0.000              # stage 2 lower bound (s)
#     DR_LAG_TAU_MAX_S2=0.030              # stage 2 upper bound (s)
#     DR_LAG_TAU_MIN_S3=0.010              # stage 3 lower bound (s) — centres around 20 ms
#     DR_LAG_TAU_MAX_S3=0.030              # stage 3 upper bound (s)
#     REWARD_ACTION_RATE_WEIGHT=           # if set, re-enables the (a_t − a_{t-1})² penalty
#                                          # to suppress motor chatter at balance. Try 0.02.
#                                          # Unset (default) leaves the env's 0.0 (disabled).

set -euo pipefail

PREFIX="${1:-curriculum}"
SEED="${SEED:-0}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-2000000}"
DEVICE="${DEVICE:-cuda:4}"
CONTROL_FREQ="${CONTROL_FREQ:-100}"
MAX_ACCEL_RAD_S2="${MAX_ACCEL_RAD_S2:-150}"
DR_LAG_TAU_MIN_S2="${DR_LAG_TAU_MIN_S2:-0.000}"
DR_LAG_TAU_MAX_S2="${DR_LAG_TAU_MAX_S2:-0.030}"
DR_LAG_TAU_MIN_S3="${DR_LAG_TAU_MIN_S3:-0.010}"
DR_LAG_TAU_MAX_S3="${DR_LAG_TAU_MAX_S3:-0.030}"
REWARD_ACTION_RATE_WEIGHT="${REWARD_ACTION_RATE_WEIGHT:-}"
REWARD_STILLNESS_BONUS_WEIGHT="${REWARD_STILLNESS_BONUS_WEIGHT:-}"

# Optional flag block: only pass each --reward-* arg if the user set it.
EXTRA_REWARD_ARGS=()
if [ -n "$REWARD_ACTION_RATE_WEIGHT" ]; then
    EXTRA_REWARD_ARGS+=(--reward-action-rate-weight "$REWARD_ACTION_RATE_WEIGHT")
fi
if [ -n "$REWARD_STILLNESS_BONUS_WEIGHT" ]; then
    EXTRA_REWARD_ARGS+=(--reward-stillness-bonus-weight "$REWARD_STILLNESS_BONUS_WEIGHT")
fi

run_stage1="${PREFIX}_stage1"
run_stage2="${PREFIX}_stage2"
run_stage3="${PREFIX}_stage3"

echo "Curriculum config:"
echo "  control rate: ${CONTROL_FREQ} Hz, max accel: ${MAX_ACCEL_RAD_S2} rad/s²"
echo "  stage 2 action-lag tau: [$(awk -v t="$DR_LAG_TAU_MIN_S2" 'BEGIN{ printf "%.0f", t*1000 }'), $(awk -v t="$DR_LAG_TAU_MAX_S2" 'BEGIN{ printf "%.0f", t*1000 }')] ms"
echo "  stage 3 action-lag tau: [$(awk -v t="$DR_LAG_TAU_MIN_S3" 'BEGIN{ printf "%.0f", t*1000 }'), $(awk -v t="$DR_LAG_TAU_MAX_S3" 'BEGIN{ printf "%.0f", t*1000 }')] ms"
if [ -n "$REWARD_ACTION_RATE_WEIGHT" ]; then
    echo "  reward_action_rate_weight: $REWARD_ACTION_RATE_WEIGHT (re-enabled, default 0.0)"
fi
echo

echo "=== Stage 1 (no DR) ==="
python -u train_PPO_stable_baseline.py \
    --total-steps "$STEPS_PER_STAGE" \
    --device "$DEVICE" \
    --control-freq "$CONTROL_FREQ" \
    --max-accel-rad-s2 "$MAX_ACCEL_RAD_S2" \
    "${EXTRA_REWARD_ARGS[@]}" \
    --run-name "$run_stage1" \
    --seed "$SEED" \
    --progress-bar

echo "=== Stage 2 (action-lag tau [${DR_LAG_TAU_MIN_S2}, ${DR_LAG_TAU_MAX_S2}] s) ==="
python -u train_PPO_stable_baseline.py \
    --total-steps "$STEPS_PER_STAGE" \
    --device "$DEVICE" \
    --control-freq "$CONTROL_FREQ" \
    --max-accel-rad-s2 "$MAX_ACCEL_RAD_S2" \
    --domain-randomization \
    --dr-action-lag-tau-min "$DR_LAG_TAU_MIN_S2" \
    --dr-action-lag-tau-max "$DR_LAG_TAU_MAX_S2" \
    "${EXTRA_REWARD_ARGS[@]}" \
    --resume "runs/${run_stage1}/best_model" \
    --run-name "$run_stage2" \
    --seed "$SEED"

echo "=== Stage 3 (action-lag tau [${DR_LAG_TAU_MIN_S3}, ${DR_LAG_TAU_MAX_S3}] s) ==="
python -u train_PPO_stable_baseline.py \
    --total-steps "$STEPS_PER_STAGE" \
    --device "$DEVICE" \
    --control-freq "$CONTROL_FREQ" \
    --max-accel-rad-s2 "$MAX_ACCEL_RAD_S2" \
    --domain-randomization \
    --dr-action-lag-tau-min "$DR_LAG_TAU_MIN_S3" \
    --dr-action-lag-tau-max "$DR_LAG_TAU_MAX_S3" \
    "${EXTRA_REWARD_ARGS[@]}" \
    --resume "runs/${run_stage2}/best_model" \
    --run-name "$run_stage3" \
    --seed "$SEED"

echo "=== Curriculum complete. Final policy: runs/${run_stage3}/best_model.zip ==="
echo "Deploy/fine-tune at the same control rate: --control-freq ${CONTROL_FREQ}"
