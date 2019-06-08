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

#define CHECK_ERR(err) { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__, errorString); \
    } \
}

int main(int argc, char **argv)
{
    void *value;
    int err, rank, tag_ub, isSet;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    err = MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &isSet);
    CHECK_ERR(err)

    tag_ub = *(int *) value;
    if (isSet)
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is %d\n",rank, tag_ub);
    else
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is NOT set\n",rank);

    MPI_Finalize();
    return 0; 
}

