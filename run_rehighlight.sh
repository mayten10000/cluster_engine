#!/bin/bash
# Re-highlight under new gates (semantic + strict-digit + global-twin).
# Sequential: cat60 → cat44 → cat37 → cat42.
# For each category:
#   1. Delete its existing moves from qc.cluster_moves
#   2. Run highlight, log to cat${cat}_rerun.log
#   3. Append summary to master log
set -u
cd /opt

LOG=/opt/cluster_engine_v2/run_rehighlight.log
CH='http://127.0.0.1:8123/'

echo "===== START $(date +%F\ %H:%M:%S) =====" | tee -a $LOG

# Wipe ALL existing moves once — we're regenerating the full review set
# from scratch under the new gates. ClickHouse has no FK to mpstats so a
# per-category subquery isn't possible.
echo "===== $(date +%H:%M:%S) wiping qc.cluster_moves =====" | tee -a $LOG
curl -s "$CH" -d "ALTER TABLE qc.cluster_moves DELETE WHERE 1=1" >> $LOG 2>&1
sleep 3
curl -s "$CH" -d "SELECT count() FROM qc.cluster_moves" 2>&1 | tee -a $LOG

for cat in 60 44 37 42; do
  echo "===== $(date +%H:%M:%S) cat${cat}: running highlight =====" | tee -a $LOG
  python3 -u -m cluster_engine_v2 --highlight --category_id=${cat} -v \
    > /opt/cluster_engine_v2/cat${cat}_rerun.log 2>&1
  rc=$?
  echo "===== $(date +%H:%M:%S) cat${cat}: finished rc=${rc} =====" | tee -a $LOG

  grep -E "STATS|HIGHLIGHT COMPLETE|llm_|n_moves|Reverted|Merged|merged-to-existing|no twin in DB" \
    /opt/cluster_engine_v2/cat${cat}_rerun.log | tail -30 | tee -a $LOG
done

echo "===== ALL DONE $(date +%F\ %H:%M:%S) =====" | tee -a $LOG
