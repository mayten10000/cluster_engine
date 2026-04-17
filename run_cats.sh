#!/bin/bash
# Sequential highlight for cat44, cat60, cat37
set -e
cd /opt
for cat in 44 60 37; do
  echo "===== $(date +%H:%M:%S) Starting cat${cat} ====="
  python3 -u -m cluster_engine_v2 --highlight --category_id=${cat} -v \
    > /opt/cluster_engine_v2/cat${cat}_v9.log 2>&1
  echo "===== $(date +%H:%M:%S) Finished cat${cat} ====="
  grep -E "STATS|HIGHLIGHT COMPLETE|llm_|n_moves|Reverted" \
    /opt/cluster_engine_v2/cat${cat}_v9.log | tail -15
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
