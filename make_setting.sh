#!/usr/bin/env sh

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : make_setting
# @created     : Thursday Aug 14, 2025 15:19:13 JST
#
# @description :
######################################################################
set -x
SERVICE_CHAIN_LENGTH=3
DATADIR=nsfnet
BANDWIDTH=1000
DELAY=1
NODES=$(seq 100 100 2000)
for N in $NODES; do
  for SEED in $(seq 1 10); do
    RESDIR=data_rng/N${N}_K${SERVICE_CHAIN_LENGTH}_seed$SEED
    python make_setting.py --seed $SEED --K $SERVICE_CHAIN_LENGTH --datadir $DATADIR --resdir $RESDIR --bw $BANDWIDTH --delay $DELAY --N $N --is_rng --candidates 2
  done &
done

for SEED in $(seq 1 10); do
  RESDIR=data_rng/NSFNET_bw${BANDWIDTH}_seed$SEED
  python make_setting.py --seed $SEED --datadir $DATADIR --resdir $RESDIR --bw $BANDWIDTH
done
