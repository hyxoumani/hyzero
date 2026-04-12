#!/usr/bin/env bash
set -euo pipefail

# Run N sequential e2e tests and append metrics to results.tsv
N=${1:-3}
DURATION=${2:-120}

RESULTS_FILE="results.tsv"

# Create header if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    printf "timestamp\tduration_s\tgames\ttrain_steps\tfirst_loss\tlast_loss\tavg_steps\terrors\n" > "$RESULTS_FILE"
fi

echo "Running $N experiments, ${DURATION}s each..."
for i in $(seq 1 $N); do
    echo "--- Experiment $i/$N ---"
    scripts/e2e_test.sh "$DURATION" || true

    # Append latest metrics to results.tsv
    LATEST=$(ls -t logs/e2e_*_metrics.txt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        # Parse metrics file and append as TSV row
        TS=$(grep "timestamp=" "$LATEST" | cut -d= -f2)
        DUR=$(grep "duration_s=" "$LATEST" | cut -d= -f2)
        GAMES=$(grep "games_completed=" "$LATEST" | cut -d= -f2)
        STEPS=$(grep "training_steps=" "$LATEST" | cut -d= -f2)
        FL=$(grep "first_loss=" "$LATEST" | cut -d= -f2)
        LL=$(grep "last_loss=" "$LATEST" | cut -d= -f2)
        AS=$(grep "avg_game_steps=" "$LATEST" | cut -d= -f2)
        ERRS=$(grep "errors=" "$LATEST" | cut -d= -f2)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$TS" "$DUR" "$GAMES" "$STEPS" "$FL" "$LL" "$AS" "$ERRS" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "Results appended to $RESULTS_FILE"
cat "$RESULTS_FILE"
