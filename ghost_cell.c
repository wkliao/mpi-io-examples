/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *  Copyright (C) 2023, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 * This example writes a 2D local array with ghost cells to a global array.
 * Ghost cells are the array elements in the local array that are not written
 * to the file. This example shows how to define an MPI derived data type to
 * describe a 2D subarray with ghost cells. The ghost cells appear on both ends
 * of each dimension of the local buffer. The contents of ghost cells are set
 * to '-8' and non-ghost cells are the process rank IDs.
 *
 * To compile:
 *        mpicc -O2 ghost_cell.c -o ghost_cell
 * To run:
 *        mpiexec -n num_processes ./ghost_cell -c num -l len filename
 *
 * num is the size of ghost cells on both ends of each dimension,
 * len is the size of local array, which is len x len.
 *
 * When using #define EXPECT(rank,x) (rank)
 * data contents in the output file
 *         0, 0, 0, 0, 1, 1, 1, 1,
 *         0, 0, 0, 0, 1, 1, 1, 1,
 *         0, 0, 0, 0, 1, 1, 1, 1,
 *         0, 0, 0, 0, 1, 1, 1, 1,
 *         2, 2, 2, 2, 3, 3, 3, 3,
 *         2, 2, 2, 2, 3, 3, 3, 3,
 *         2, 2, 2, 2, 3, 3, 3, 3,
 *         2, 2, 2, 2, 3, 3, 3, 3 ;
 *
 * the contents of local buffers are shown below.
 *
 * rank 0:                                rank 1:
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8,  0,  0,  0,  0, -8, -8     -8, -8,  1,  1,  1,  1, -8, -8
 *    -8, -8,  0,  0,  0,  0, -8, -8     -8, -8,  1,  1,  1,  1, -8, -8
 *    -8, -8,  0,  0,  0,  0, -8, -8     -8, -8,  1,  1,  1,  1, -8, -8
 *    -8, -8,  0,  0,  0,  0, -8, -8     -8, -8,  1,  1,  1,  1, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *
 * rank 2:                                rank 3:
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8,  2,  2,  2,  2, -8, -8     -8, -8,  3,  3,  3,  3, -8, -8
 *    -8, -8,  2,  2,  2,  2, -8, -8     -8, -8,  3,  3,  3,  3, -8, -8
 *    -8, -8,  2,  2,  2,  2, -8, -8     -8, -8,  3,  3,  3,  3, -8, -8
 *    -8, -8,  2,  2,  2,  2, -8, -8     -8, -8,  3,  3,  3,  3, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *    -8, -8, -8, -8, -8, -8, -8, -8     -8, -8, -8, -8, -8, -8, -8, -8
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* strcpy() */
#include <unistd.h>    /* getopt() */
#include <sys/types.h> /* open() */
#include <sys/stat.h>  /* open() */
#include <fcntl.h>     /* open() */

#include <mpi.h>

#define EXPECT(rank,x) (rank)

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        nerrs++; \
    }

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-h | -q | -c num | -l len | -n num | file_name]\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "       [-l len] size of each dimension of the local array (default: 4)\n"
    "       [-c num] number of ghost cells along each dimension (default: 2) \n"
    "       [-n num] write count of buffer data type (default: 2) \n"
    "       [filename] output file name (default: testfile.dat)\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256];
    int i, j, k, x, rank, nprocs, mode, len, bufsize, ntimes, err, nerrs=0;
    int psizes[2], gsizes[2], subsizes[2], *gstarts=NULL, starts[2], nghosts;
    int fd, sizes[2], local_rank[2], *buf=NULL, *buf_ptr, type_size, verbose;

    MPI_Aint lb, extent;
    MPI_File fh;
    MPI_Offset off;
    MPI_Datatype buf_type, file_type;
    MPI_Info info = MPI_INFO_NULL;
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    verbose = 1;
    nghosts = 2;
    len     = 4;
    ntimes  = 1;
    off     = 10;

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hqn:c:l:")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'c': nghosts = atoi(optarg);
                      break;
            case 'n': ntimes = atoi(optarg);
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 1;
        }
    if (argv[optind] == NULL) strcpy(filename, "testfile.nc");
    else                      snprintf(filename, 256, "%s", argv[optind]);

    len = (len <= 0) ? 4 : len;
    nghosts = (nghosts < 0) ? 2 : nghosts;
    ntimes = (ntimes <= 0) ? 1 : ntimes;

    if (verbose && rank == 0) {
        printf("local array size         = %d %d\n", len, len);
        printf("number of ghost cells    = %d\n", nghosts);
        printf("file starting offset     = %lld\n", off);
        printf("number of buffer types   = %d\n", ntimes);
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

    gsizes[0] = len * psizes[0]; /* global array size */
    gsizes[1] = len * psizes[1];
    if (verbose && rank == 0)
        printf("global variable shape:     %lld %lld\n", gsizes[0],gsizes[1]);

    /* create fileview data type */
    starts[0]   = local_rank[0] * len;
    starts[1]   = local_rank[1] * len;
    subsizes[0] = len;
    subsizes[1] = len;
    err = MPI_Type_create_subarray(2, gsizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &file_type);
    ERR
    err = MPI_Type_commit(&file_type);
    ERR

    if (!rank) gstarts = (int*) malloc(nprocs * 2 * sizeof(int));
    MPI_Gather(starts, 2, MPI_INT, gstarts, 2, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_size(file_type, &type_size);
    MPI_Type_get_extent(file_type, &lb, &extent);
    if (verbose && rank == 0)
        printf("file_type size=%d lb=%ld extent=%ld\n",type_size,lb,extent);

    /* create buffer data type with ghost cells */
    sizes[0]  = len + nghosts * 2;
    sizes[1]  = len + nghosts * 2;
    starts[0] = nghosts;
    starts[1] = nghosts;
    err = MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &buf_type);
    ERR

    err = MPI_Type_commit(&buf_type);
    ERR

    MPI_Type_size(buf_type, &type_size);
    MPI_Type_get_extent(buf_type, &lb, &extent);
    if (verbose && rank == 0)
        printf(" buf_type size=%d lb=%ld extent=%ld\n",type_size,lb,extent);

    /* initialize buffer with ghost cells on both ends of each dim */
    bufsize = (len + 2 * nghosts) * (len + 2 * nghosts);
    buf = (int *) malloc(bufsize * ntimes * sizeof(int));
    buf_ptr = buf;
    for (k=0; k<ntimes; k++) {
        x = 0;
        for (i=0; i<len+2*nghosts; i++)
        for (j=0; j<len+2*nghosts; j++) {
            if (nghosts <= i && i < len+nghosts &&
                nghosts <= j && j < len+nghosts)
                buf_ptr[i*(len+2*nghosts) + j] = EXPECT(rank, x);
            else
                /* set all ghost cells value to -8 */
                buf_ptr[i*(len+2*nghosts) + j] = -8;
        }
        buf_ptr += bufsize;
    }

    /* create the file */
    mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh);
    ERR

    /* set the file view */
    err = MPI_File_set_view(fh, off, MPI_BYTE, file_type, "native", info);
    ERR

    /* write to the file */
    err = MPI_File_write_all(fh, buf, ntimes, buf_type, &status);
    ERR

    MPI_File_close(&fh);

    err = MPI_Type_free(&file_type);
    ERR
    err = MPI_Type_free(&buf_type);
    ERR

    if (rank) goto err_out;

    /* root process reads the entire file and checks contents */
    free(buf);
    bufsize = gsizes[0] * gsizes[1];
    buf = (int *) malloc(bufsize * ntimes * sizeof(int));

    fd = open(filename, O_RDONLY, 0400);
    lseek(fd, off, SEEK_SET);
    read(fd, buf, bufsize * ntimes * sizeof(int));
    close(fd);

    /* print the entire array */
    if (verbose) {
        for (k=0; k<ntimes; k++) {
            printf("k = %d\n",k);
            for (i=0; i<gsizes[0]; i++) {
                for (j=0; j<gsizes[1]; j++)
                    printf(" %d",buf[k*bufsize+gsizes[1]*i+j]);
                printf("\n");
            }
            printf("\n");
        }
    }

    /* check if the contents are expected */
    for (k=0; k<ntimes; k++) {
        int p;
        for (p=0; p<nprocs; p++) {
            int local_indx = gsizes[0] * gsizes[1] * k;
            local_indx += gstarts[p*2] * gsizes[1] + gstarts[p*2+1];

            x = 0;
            for (i=0; i<len; i++, local_indx += gsizes[1]) {
                for (j=0; j<len; j++) {
                    if (buf[local_indx + j] != EXPECT(p, x)) {
                        printf("Error: Unexpected value %d at k=%d p=%d i=%d j=%d\n",
                                buf[local_indx + j],k,p,i,j);
                        nerrs++;
                        goto err_out;
                    }
                }
            }
        }
    }

err_out:
    free(buf);
    if (gstarts != NULL) free(gstarts);

    MPI_Finalize();
    return (nerrs > 0);
}

