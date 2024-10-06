/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * This program tests collective write using a file datatype constructed from
 * multiple subarray datatypes concatenated by MPI_Type_indexed(). Each
 * variable is partitioned among processes in a 2D block-block fashion. At the
 * end, it checks the file size whether or not it is expected.
 *
 * This program is the same as hindexed_fsize.c, except it calls
 * MPI_Type_hindexed(), while hindexed_fsize.c.c calls
 * MPI_Type_create_hindexed().
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>

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

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-hvrc | -n num | -l len ] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-v] verbose mode\n"
    "       [-n num] number of variables to be written\n"
    "       [-l len] length of local X and Y dimension sizes\n"
    "        -f filename: output file name\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256], *buf;
    int i, rank, nprocs, err, nerrs=0, verbose, omode, nvars, len;
    int psizes[2], sizes[2], subsizes[2], starts[2], *blks;
    int *disp;
    MPI_Datatype subType, fileType;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info=MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    verbose     = 0;
    nvars       = 2;     /* default number of variables */
    len         = 10;    /* default dimension size */
    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hvn:l:f:")) != EOF)
        switch(i) {
            case 'v': verbose = 1;
                      break;
            case 'n': nvars = atoi(optarg);
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'f': strcpy(filename, optarg);
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 1;
        }

    if (filename[0] == '\0') {
        if (rank==0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (verbose && rank == 0) {
        printf("Number of MPI processes:  %d\n",nprocs);
        printf("Number of varaibles:      %d\n",nvars);
        printf("Each subarray is of size: %d x %d (int) = %zd\n",
               len, len, sizeof(int)*len*len);
    }

    /* calculate number of processes along each dimension */
    psizes[0] = psizes[1] = 0;
    err = MPI_Dims_create(nprocs, 2, psizes); ERR

    if (verbose)
        printf("%d: 2D rank IDs: %d, %d\n",rank,rank/psizes[1], rank%psizes[1]);

    /* create a subarray datatype */
    sizes[0]    = len * psizes[0];
    sizes[1]    = len * psizes[1];
    subsizes[0] = len;
    subsizes[1] = len;
    starts[0]   = len * (rank / psizes[1]);
    starts[1]   = len * (rank % psizes[1]);
    err = MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_BYTE, &subType); ERR

    if (verbose)
        printf("%d: sizes=%d %d subsizes=%d %d starts=%d %d\n",rank,
               sizes[0],sizes[1], subsizes[0],subsizes[1], starts[0],starts[1]);

    /* concatenate nvars subTypes into fileType */
    disp = (int*) malloc(sizeof(int) * nvars);
    blks = (int*) malloc(sizeof(int) * nvars);
    for (i=0; i<nvars; i++) {
        disp[i] = i;
        blks[i] = 1;
    }
    err = MPI_Type_indexed(nvars, blks, disp, subType, &fileType); ERR
    err = MPI_Type_commit(&fileType); ERR

    err = MPI_Type_free(&subType); ERR
    free(disp);
    free(blks);

    /* allocate I/O buffers */
    buf = (char*)malloc(nvars * len * len);

    /* open file and truncate it to zero sized */
    omode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, omode, info, &fh); ERR
    err = MPI_File_set_size(fh, 0); ERR

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", info); ERR
    err = MPI_Type_free(&fileType); ERR

    /* write to the file */
    err = MPI_File_write_all(fh, buf, len*len*nvars, MPI_BYTE, &status); ERR

    /* flush and close the file */
    err = MPI_Barrier(MPI_COMM_WORLD); ERR
    err = MPI_File_sync(fh); ERR
    err = MPI_Barrier(MPI_COMM_WORLD); ERR
    err = MPI_File_close(&fh); ERR
    err = MPI_Barrier(MPI_COMM_WORLD); ERR

    /* check file size */
    if (rank == 0) {
        int fd = open(filename, O_RDONLY, 0600);
        off_t fsize = lseek(fd, 0, SEEK_END);
        off_t expected = len * len * nvars * nprocs;
        if (fsize != expected) {
            printf("Error: expecting file size %lld, but got %lld\n", expected, fsize);
            err = 1;
        }
        else
            printf("Test passed\n");
        close(fd);
    }
    MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (err > 0) nerrs++;
    free(buf);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}

