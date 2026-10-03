#!/usr/bin/env bash
# Second queue: the emulated globe rides at the route-point height (Z_HEIGHT),
# so a height sensitivity needs the full facet-thermal path re-traced at
# 0.8 m and 1.2 m. Waits for the first queue to finish before starting.
cd "$(dirname "$0")"
while pgrep -f "bash ./variants_queue.sh" > /dev/null; do sleep 60; done
Z_HEIGHT=0.8 FORCE_SVF=1 ./paper_variant_runs.sh height_z0p8 4
Z_HEIGHT=1.2 FORCE_SVF=1 ./paper_variant_runs.sh height_z1p2 4
echo "queue2 done $(date)"
