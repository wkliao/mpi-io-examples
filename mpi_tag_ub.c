/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *  Copyright (C) 2019, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 * This program shows how to obtain the value of MPI_TAG_UB, which is an
 * attribute of an MPI communicator.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "mpi_utils.h"

int main(int argc, char **argv)
{
    void *value;
    int err, rank, tag_ub, isSet;

    MPI_Init(&argc,&argv);

    MPI_CHECK_ERR( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

    MPI_CHECK_ERR( MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &isSet) );

    tag_ub = *(int *) value;
    if (isSet)
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is %d\n",rank, tag_ub);
    else
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is NOT set\n",rank);

    MPI_Finalize();
    return 0;
}
