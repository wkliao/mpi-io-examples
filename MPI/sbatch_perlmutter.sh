#!/bin/bash  -l
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -t 00:10:00

#SBATCH --nodes=16
#SBATCH --job-name=alltoallw
#SBATCH -o qout.%x.%j
#SBATCH -e qout.%x.%j
#------------------------------------------------------------------------#
cd $PWD

if test "x$SLURM_NTASKS_PER_NODE" = x ; then
   SLURM_NTASKS_PER_NODE=128
fi
NP=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))

export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_RX_MATCH_MODE=software
export MPICH_OFI_NIC_POLICY=NUMA

echo "------------------------------------------------------"
echo "---- Running on Perlmutter CPU nodes ----"
echo "---- SLURM_CLUSTER_NAME      = $SLURM_CLUSTER_NAME"
echo "---- SLURM_JOB_QOS           = $SLURM_JOB_QOS"
echo "---- SLURM_JOB_PARTITION     = $SLURM_JOB_PARTITION"
echo "---- SLURM_JOB_NAME          = $SLURM_JOB_NAME"
echo "---- SBATCH_CONSTRAINT       = $SBATCH_CONSTRAINT"
echo "---- SLURM_JOB_NODELIST      = $SLURM_JOB_NODELIST"
echo "---- SLURM_JOB_NUM_NODES     = $SLURM_JOB_NUM_NODES"
echo "---- SLURM_NTASKS_PER_NODE   = $SLURM_NTASKS_PER_NODE"
echo "---- SLURM_JOB_ID            = $SLURM_JOB_ID"
echo "---- SLURM out/err file      = qout.$SLURM_JOB_NAME.$SLURM_JOB_ID"
echo ""
echo "ENV explicitly set:"
echo "---- FI_MR_CACHE_MONITOR     = $FI_MR_CACHE_MONITOR"
echo "---- FI_UNIVERSE_SIZE        = $FI_UNIVERSE_SIZE"
echo "---- FI_CXI_DEFAULT_CQ_SIZE  = $FI_CXI_DEFAULT_CQ_SIZE"
echo "---- FI_CXI_RX_MATCH_MODE    = $FI_CXI_RX_MATCH_MODE"
echo "---- MPICH_COLL_SYNC         = $MPICH_COLL_SYNC"
echo "---- MPICH_OFI_NIC_POLICY    = $MPICH_OFI_NIC_POLICY"
echo "------------------------------------------------------"
echo ""

# For fast executable loading on Cori and Perlmutter
EXE_FILE=alltoallw
EXE=/tmp/${USER}_${EXE_FILE}
sbcast ${EXE_FILE} ${EXE}

echo ""
echo "========================================================================"
echo ""

NTIMES=3
for ntime in $(seq 1 ${NTIMES}) ; do
   date
   echo "---- iteration $ntime -----------------------------------------------"
   echo ""

   CMD_OPTS="-n 253 -r 32"

   CMD="srun -n $NP ${EXE} $CMD_OPTS"
   echo "CMD=$CMD"
   $CMD

   echo ""
   echo "====================================================================="
done  # loop ntimes

date

