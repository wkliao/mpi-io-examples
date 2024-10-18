/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *  Copyright (C) 2024, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 * This program shows how to set a 2D column-wise data partitioning in an MPI
 * derived data type, which is then used to set the file view. The global 2D
 * array is of size (len x number of MPI processes), where len can be set by
 * the command-line option '-l'.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        nerrs++; \
        goto err_out; \
    }

/*----< usage() >------------------------------------------------------------*/
static void usage (char *argv0) {
    char *help = "Usage: %s [OPTION]\n\
       [-h] Print this help message\n\
       [-v] Verbose mode (default: no)\n\
       [-l len] length of Y dimension (default: 10)\n\
       [-o path] Output file path\n";
    fprintf (stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    extern int optind;
    extern char *optarg;
    char *filename;
    int i, rank, nprocs, err, nerrs=0, verbose, omode, len;
    int sizes[2], subsizes[2], starts[2];
    float *buf;
    MPI_Datatype fileType;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info=MPI_INFO_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    verbose = 0;
    len = 10;
    /* command-line arguments */
    while ((i = getopt (argc, argv, "hvl:o:")) != EOF)
        switch (i) {
            case 'v':
                verbose = 1;
                break;
            case 'l':
                len = atoi(optarg);
                break;
            case 'o':
                filename = strdup(optarg);
                break;
            case 'h':
            default:
                if (rank == 0) usage(argv[0]);
                goto err_out;
        }

    if (filename == NULL) { /* output file is mandatory */
        if (!rank) usage (argv[0]);
        goto err_out;
    }

    buf = (float*) malloc(sizeof(float) * len);

    /* construct filetype */
    sizes[0]    = len;
    sizes[1]    = nprocs;
    subsizes[0] = len;
    subsizes[1] = 1;
    starts[0]   = 0;
    starts[1]   = rank;

    err = MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_FLOAT, &fileType); ERR
    err = MPI_Type_commit(&fileType); ERR

    /* open file and truncate it to zero sized */
    omode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, omode, info, &fh); ERR
    err = MPI_File_set_size(fh, 0); ERR

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", info); ERR
    err = MPI_Type_free(&fileType); ERR

    /* write to the file */
    err = MPI_File_write_all(fh, buf, len, MPI_FLOAT, &status); ERR

    MPI_File_close(&fh);

    free(buf);

err_out:
    MPI_Finalize();
    return 0;
}


