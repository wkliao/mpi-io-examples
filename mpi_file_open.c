/*********************************************************************
 *
 *  Copyright (C) 2019, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define LEN 10

#include "mpi_utils.h"

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    char *filename;
    int err, rank, nprocs, cmode, omode;
    MPI_File fh;
    MPI_Info info;

    MPI_Init(&argc, &argv);
    MPI_CHECK_ERR( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_CHECK_ERR( MPI_Comm_size(MPI_COMM_WORLD, &nprocs) );

    filename = "testfile.out";
    if (argc > 1) filename = argv[1];

    /* Users can set customized I/O hints in info object */
    info = MPI_INFO_NULL;  /* no user I/O hint */

    /* set file open mode */
    cmode  = MPI_MODE_CREATE; /* to create a new file */
    cmode |= MPI_MODE_WRONLY; /* with write-only permission */

    /* collectively open a file, shared by all processes in MPI_COMM_WORLD */
    MPI_CHECK_ERR( MPI_File_open(MPI_COMM_WORLD, filename, cmode, info, &fh) );

    /* collectively close the file */
    MPI_CHECK_ERR( MPI_File_close(&fh) );

    /* set file open mode */
    omode = MPI_MODE_RDONLY; /* with read-only permission */

    /* collectively open a file, shared by all processes in MPI_COMM_WORLD */
    MPI_CHECK_ERR( MPI_File_open(MPI_COMM_WORLD, filename, omode, info, &fh) );

    /* collectively close the file */
    MPI_CHECK_ERR( MPI_File_close(&fh) );

prog_exit:
    MPI_Finalize();
    return 0;
}
