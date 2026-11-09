#!/usr/bin/env sh

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : run
# @created     : Tuesday May 27, 2025 12:22:17 JST
#
# @description :
######################################################################
set -x
BS=128
SEED_LIST=$(seq 1 10)
SERVICE_CHAIN_LENGTH=4
SEED=1
BS=128
SERVICE_CHAIN_LENGTH=3
NODES=$(seq 10 10 100)
CAPACITY=2e9
DATADIR=data_example
BASEDIR="results/example/"
RESDIR="${BASEDIR}"
BS=128
SEED=1

CAPACITIES="5e8"
for CAPACITY in $CAPACITIES; do
  for SEED in $SEED_LIST; do
    DATADIR=data_rng/NSFNET_bw1000_seed${SEED}
    RESDIR=results/nsfnet/nsfnet_capacity${CAPACITY}_seed${SEED}
    mkdir -p $RESDIR
    python measure_memory.py model_splitting_placement_chaining_z.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/ilp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/ilp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/layered_dijkstra_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/mspr_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_z_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/lp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/lp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra_capacity.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/layered_dijkstra_capacity_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_blocked_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/blocked_mspr_memory_usage.log
  done
done

# Layers
CAPACITY=1e9
MODELS="resnet18 resnet34 resnet50 resnet101 resnet152"
for MODEL in $MODELS; do
  for SEED in $SEED_LIST; do
    NODE=1000
    DATADIR=data_rng/N${NODE}_K3_seed${SEED}
    RESDIR=results/layers/${MODEL}_seed${SEED}
    mkdir -p $RESDIR
    python measure_memory.py model_splitting_placement_chaining_z.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/ilp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/ilp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_z_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/lp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/lp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/layered_dijkstra_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/mspr_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra_capacity.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --bs ${BS} --capacity ${CAPACITY} --model ${MODEL} --is_test >${RESDIR}/layered_dijkstra_capacity_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_blocked_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --model ${MODEL} --bs ${BS} --capacity ${CAPACITY} --is_test >${RESDIR}/blocked_mspr_memory_usage.log
  done
done

# SCALABILITY TO PHYSICAL NODES
CAPACITY=1e9
NODES=$(seq 100 100 2000)
for NODE in $NODES; do
  for SEED in $SEED_LIST; do
    RESDIR=results/nodes/N${NODE}_seed${SEED}
    DATADIR=data_rng/N${NODE}_K3_seed${SEED}
    mkdir -p $RESDIR
    python measure_memory.py model_splitting_placement_chaining_z.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/ilp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/ilp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/layered_dijkstra_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/mspr_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_z_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/lp_z_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_lp.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/lp_layered_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_layered_dijkstra_capacity.py --resdir ${RESDIR} --datadir ${DATADIR} --capacity ${CAPACITY} --bs ${BS} --seed ${SEED} --is_test >${RESDIR}/layered_dijkstra_capacity_memory_usage.log
    python measure_memory.py model_splitting_placement_chaining_blocked_mspr.py --resdir ${RESDIR} --datadir ${DATADIR} --seed ${SEED} --capacity ${CAPACITY} --bs ${BS} --is_test >${RESDIR}/blocked_mspr_memory_usage.log
  done
done
