/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective write and read calls using a fileview datatype
 * of size that is a multiple of buffer datatype size.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */

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
    "Usage: %s [-hq | -l len | -n num] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "       [-l len] length of local X and Y dimension sizes\n"
    "       [-n num] number of file datatype to be written\n"
    "        -f filename: output file name\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256];
    size_t i, j, k;
    int err, nerrs=0, rank, nprocs, mode, verbose=1, ntimes, len;
    int psizes[2], gsizes[2], subsizes[2], starts[2], lsizes[2];
    int local_rank[2], *buf=NULL, type_size, gap, max_nerrs;
    double timing, max_timing;     
    MPI_Aint lb, displace[2], extent;
    MPI_Datatype bufType, fileType;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ntimes = 2;
    len = 100;  /* default dimension size */
    gap = 4;    /* gap between 2 blocks in bufType */
    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hql:n:f:")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
                      break;
            case 'n': ntimes = atoi(optarg);
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
        printf("Creating a buffer datatype consisting of %d blocks\n",ntimes);
        printf("Each block is of size %d x %d (int)= %zd\n",
               len, len, sizeof(int)*len*len);
        printf("Gap between two consecutive blocks is %d ints\n", gap);
    }

    /* calculate number of processes along each dimension */
    psizes[0] = psizes[1] = 0;
    MPI_Dims_create(nprocs, 2, psizes);
    if (verbose && rank == 0)
        printf("process dimension psizes = %d %d\n", psizes[0], psizes[1]);
    
    /* find its local rank IDs along each dimension */
    local_rank[0] = rank / psizes[1];
    local_rank[1] = rank % psizes[1];
    if (verbose) 
        printf("rank %2d: local rank =      %d %d\n",
               rank,local_rank[0],local_rank[1]);

    /* create fileview data type */
    gsizes[0] = len * psizes[0] * ntimes; /* global array size */
    gsizes[1] = len * psizes[1];
    if (verbose && rank == 0)
        printf("global variable shape:     %d %d\n", gsizes[0],gsizes[1]);

    starts[0]   = local_rank[0] * len * ntimes;
    starts[1]   = local_rank[1] * len;
    subsizes[0] = len * ntimes;
    subsizes[1] = len;
    err = MPI_Type_create_subarray(2, gsizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &fileType);
    ERR
    err = MPI_Type_commit(&fileType); ERR

    MPI_Type_size(fileType, &type_size);
    lb = 0;
    MPI_Type_get_extent(fileType, &lb, &extent);
    if (verbose && rank == 0)
        printf("file   type size = %d extent = %ld\n", type_size, extent);

    /* create a datatype consists of 2 blocks, with a gap in between. The size
     * of data type should be equal to len*len * sizeof(int).
     */
    lsizes[0] = lsizes[1] = len * len / 2;
    displace[0] = 0;
    displace[1] = (lsizes[0] + gap)* sizeof(int);
    err = MPI_Type_create_hindexed(2, lsizes, displace, MPI_INT, &bufType);
    ERR
    err = MPI_Type_commit(&bufType); ERR

    /* allocate I/O buffer */
    MPI_Type_size(bufType, &type_size);
    lb = 0;
    MPI_Type_get_extent(bufType, &lb, &extent);
    if (verbose && rank == 0)
        printf("buffer type size = %d extent = %ld\n", type_size, extent);

    buf = (int*) calloc(extent * ntimes, sizeof(int));
    j = 0;
    for (k=0; k<ntimes; k++) {
        for (i=0; i<lsizes[0]; i++, j++) buf[j] = (j + 17 + rank) % 2147483647;
        j += gap;
        for (i=0; i<lsizes[1]; i++, j++) buf[j] = (j + 17 + rank) % 2147483647;
    }

    /* open file */
    mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh); ERR

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", info);
    ERR

    /* write to the file */
    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();
    err = MPI_File_write_all(fh, buf, ntimes, bufType, &status); ERR

    for (i=0; i<len*len; i++) buf[i] = 0;
    err = MPI_File_seek(fh, 0, MPI_SEEK_SET); ERR
    err = MPI_File_read_all(fh, buf, ntimes, bufType, &status); ERR
    timing = MPI_Wtime() - timing;

    j = 0;
    for (k=0; k<ntimes; k++) {
        for (i=0; i<lsizes[0]; i++, j++) {
            int exp = (j + 17 + rank) % 2147483647;
            if (buf[j] != exp) {
                printf("Error: buf[%zd] expect %d but got %d\n", j, exp, buf[j]);
                nerrs++;
                break;
            }
        }
        j += gap;
        for (i=0; i<lsizes[1]; i++, j++) {
            int exp = (j + 17 + rank) % 2147483647;
            if (buf[j] != exp) {
                printf("Error: buf[%zd] expect %d but got %d\n", j, exp, buf[j]);
                nerrs++;
                break;
            }
        }
    }

    err = MPI_File_close(&fh); ERR
    err = MPI_Type_free(&bufType); ERR
    err = MPI_Type_free(&fileType); ERR
    free(buf);

    MPI_Allreduce(&nerrs, &max_nerrs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (max_nerrs == 0 && rank == 0)
        printf("Time of collective write and read = %.2f sec\n", max_timing);

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}

