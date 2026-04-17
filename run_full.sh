#!/bin/bash
# Full review pipeline:
#   1. Regenerate schemas for all niches in cat 37/42/44/60
#   2. Run highlight for cat60 → cat44 → cat37 → cat42 (sequential)
set -u
cd /opt

LOG=/opt/cluster_engine_v2/run_full.log
echo "===== START $(date +%F\ %H:%M:%S) =====" | tee -a $LOG

# ---- Schema regen ----
echo "===== REGEN SCHEMAS =====" | tee -a $LOG
i=0
total=$(wc -l < /tmp/niches_to_regen.txt)
while read nk cat; do
  i=$((i+1))
  echo "[$i/$total] niche=$nk cat=$cat" | tee -a $LOG
  python3 -m cluster_engine_v2.regen_schema $nk \
    >> /opt/cluster_engine_v2/regen_all.log 2>&1 || echo "  FAILED nk=$nk" | tee -a $LOG
done < /tmp/niches_to_regen.txt
echo "===== REGEN DONE $(date +%H:%M:%S) =====" | tee -a $LOG

# ---- Highlight runs ----
for cat in 60 44 37 42; do
  echo "===== $(date +%H:%M:%S) Starting cat${cat} highlight =====" | tee -a $LOG
  python3 -u -m cluster_engine_v2 --highlight --category_id=${cat} -v \
    > /opt/cluster_engine_v2/cat${cat}_full.log 2>&1
  echo "===== $(date +%H:%M:%S) Finished cat${cat} =====" | tee -a $LOG
  grep -E "STATS|HIGHLIGHT COMPLETE|llm_|n_moves|Reverted|merged" \
    /opt/cluster_engine_v2/cat${cat}_full.log | tail -20 | tee -a $LOG
done

echo "===== ALL DONE $(date +%F\ %H:%M:%S) =====" | tee -a $LOG
