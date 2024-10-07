#!/bin/bash
#
# Copyright (C) 2023, Northwestern University
# See COPYRIGHT notice in top-level directory.
#

# Exit immediately if a command exits with a non-zero status.
set -e

MPIRUN="mpiexec ${MPIRUN_OPTS} -n 4"

for f in ${check_PROGRAMS} ; do
    if test "$f" = "print_mpi_io_hints" ; then
       OPTS="testfile"
    elif test "$f" = "indexed_fsize" ; then
       OPTS="-f testfile"
    elif test "$f" = "hindexed_fsize" ; then
       OPTS="-f testfile"
    elif test "$f" = "struct_fsize" ; then
       OPTS="-f testfile"
    elif test "$f" = "nvars" ; then
       OPTS="-r -f testfile"
    fi
    CMD="${MPIRUN} ./$f ${OPTS}"
    echo "==========================================================="
    echo "    Parallel testing on 4 MPI processes"
    echo ""
    echo "    $CMD"
    ${CMD}
    echo "==========================================================="
done

# delete output file
rm -f ./testfile

