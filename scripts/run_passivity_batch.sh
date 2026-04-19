#!/usr/bin/env bash
# Queue of experiments. Each runs ~8 min.
set -u
cd /home/devs/workspace/hyzero

D=480

bash scripts/passivity_experiment.sh e2_beta07   $D HYZERO_VALUE_OUTCOME_BETA=0.7
bash scripts/passivity_experiment.sh e3_entropy  $D HYZERO_VALUE_OUTCOME_BETA=0.3 HYZERO_POLICY_ENTROPY_WEIGHT=0.02
bash scripts/passivity_experiment.sh e4_value_w3 $D HYZERO_VALUE_OUTCOME_BETA=0.3 HYZERO_VALUE_LOSS_WEIGHT=3.0
bash scripts/passivity_experiment.sh e5_combo    $D HYZERO_VALUE_OUTCOME_BETA=0.7 HYZERO_VALUE_LOSS_WEIGHT=3.0
bash scripts/passivity_experiment.sh e6_gamma    $D HYZERO_VALUE_OUTCOME_BETA=0.3 HYZERO_REWARD_OUTCOME_GAMMA=0.5
bash scripts/passivity_experiment.sh e7_extreme  $D HYZERO_VALUE_OUTCOME_BETA=0.9 HYZERO_VALUE_LOSS_WEIGHT=5.0 HYZERO_POLICY_ENTROPY_WEIGHT=0.02
echo "=== BATCH DONE ==="
python3 scripts/compare_experiments.py
