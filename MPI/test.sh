#!/bin/bash
#
# Copyright (C) 2024, Northwestern University
# See COPYRIGHT notice in top-level directory.
#

# Exit immediately if a command exits with a non-zero status.
set -e

MPIRUN="mpiexec ${MPIRUN_OPTS} -n 4"

for f in ${check_PROGRAMS} ; do
    if test "$f" = "alltomany" ; then
       OPTS=
    fi
    CMD="${MPIRUN} ./$f ${OPTS}"
    echo "==========================================================="
    echo "    Parallel testing on 4 MPI processes"
    echo ""
    echo "    $CMD"
    echo ""
    ${CMD}
    echo "==========================================================="
done


