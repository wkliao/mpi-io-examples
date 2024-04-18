/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests collective write and read calls using a noncontiguous
 * user buffer datatype, which consists of 2 blocks separated by a gap. The
 * block size and gap size can be adjusted through command-line options.
 *
 * This program is to test the performance impact of many memcpy() called in
 * ROMIO's subroutine ADIOI_LUSTRE_Fill_send_buffer() when the user buffer is
 * non contiguous.
 *
 * The performance issue is discovered when running a PIO test program using
 * Lustre. When read/write requests are large and the Lustre striping size is
 * small, then the number of calls to memcpy() can become large, hurting the
 * performance.
 *
 * The original settings of PIO test program using the followings:
 *   The number of MPI process clients = 2048
 *   The number of I/O tasks (aggregators) = 16
 *   The number of variables = 64
 *   One extra small variable is written before 64 variables.
 *   Each variables is a 2D array of size 58 x 10485762
 *   Data partitioning is done along the 2nd dimension
 *   Writes to all 64 subarrays are aggregated into one MPI_File_write call
 *   User buffer consists of two separately allocated memory spaces.
 *
 * To compile:
 *   % mpicc -O2 pio_noncontig.c -o pio_noncontig
 *
 * Example output of running 16 processes on a local Linux machine using UFS:
 * Note the 2 runs below differ only on whether option "-g 0" is used. Option
 * "-g 0" does not add a gap in the user buffer, making the buffer contiguous.
 *
 *   % mpiexec -n 16 pio_noncontig -k 256 -c 32768 -w
 *     Number of global variables = 64
 *     Each global variable is of size 256 x 32768 bytes
 *     Each  local variable is of size 256 x 16 bytes
 *     Gap between the first 2 variables is of size 16 bytes
 *     Number of subarray types concatenated is 8192
 *     Each process makes a request of amount 33554688 bytes
 *     ROMIO hint set: cb_buffer_size = 1048576
 *     ROMIO hint set: cb_nodes = 4
 *     ---------------------------------------------------------
 *     Time of collective write = 33.07 sec
 *     ---------------------------------------------------------
 *
 *   % mpiexec -n 16 pio_noncontig -k 256 -c 32768 -w -g 0
 *     Number of global variables = 64
 *     Each global variable is of size 256 x 32768 bytes
 *     Each  local variable is of size 256 x 16 bytes
 *     Gap between the first 2 variables is of size 0 bytes
 *     Number of subarray types concatenated is 8192
 *     Each process makes a request of amount 33554688 bytes
 *     ROMIO hint set: cb_buffer_size = 1048576
 *     ROMIO hint set: cb_nodes = 4
 *     ---------------------------------------------------------
 *     Time of collective write = 8.27 sec
 *     ---------------------------------------------------------
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* getopt() */

#include <mpi.h>

#define NVARS 64         /* Number of variables */
#define NROWS 58         /* Number of rows in each variable */
#define NCOLS 1048576    /* Number of rows in each variable */
#define NAGGR 16         /* Number of I/O aggregators */
#define NCLIENTS 2048    /* Number of MPI process clients */
#define GAP 16           /* gap size in the user buffer, mimic 2 malloc() */

#define cb_buffer_size "1048576"
#define cb_nodes "4"

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
    "Usage: %s [-hvrw | -n num | -k num | -c num | -g num | file_name]\n"
    "       [-h] Print this help\n"
    "       [-v] verbose mode\n"
    "       [-w] performs write only (default: both write and read)\n"
    "       [-r] performs read  only (default: both write and read)\n"
    "       [-n num] number of global variables (default: %d)\n"
    "       [-k num] number of rows    in each global variable (default: %d)\n"
    "       [-c num] number of columns in each global variable (default: %d)\n"
    "       [-g num] gap in bytes between first 2 blocks (default: %d)\n"
    "       [file_name] output file name\n";
    fprintf(stderr, help, argv0, NVARS, NROWS, NCOLS, GAP);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256];
    int i, err, nerrs=0, rank, nprocs, mode, verbose=0, nvars, nreqs;
    int gap, ncols_g, nrows, ncols, *blocklen, btype_size, ftype_size;
    int do_write, do_read;
    char *buf;
    double timing[2], max_timing[2];
    MPI_Aint lb, *displace, buf_ext, file_ext;
    MPI_Datatype bufType, fileType, *subTypes;
    MPI_File fh;
    MPI_Offset wlen;
    MPI_Status status;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    nvars   = NVARS;
    nrows   = NROWS;
    ncols_g = NCOLS;
    gap     = GAP;
    do_write = 1;
    do_read  = 1;

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hvwrn:k:c:g:")) != EOF)
        switch(i) {
            case 'v': verbose = 1;
                      break;
            case 'n': nvars = atoi(optarg);
                      break;
            case 'k': nrows = atoi(optarg);
                      if (nrows < 0) {
                          if (rank == 0)
                              printf("Error: number of rows must >= 0\n");
                          MPI_Finalize();
                          return 1;
                      }
                      break;
            case 'c': ncols_g = atoi(optarg);
                      if (ncols_g < 2048) {
                          if (rank == 0)
                              printf("Error: number of columns must >= %d\n",
                                     NCLIENTS);
                          MPI_Finalize();
                          return 1;
                      }
                      break;
            case 'g': gap = atoi(optarg);
                      break;
            case 'w': do_read = 0;
                      break;
            case 'r': do_write = 0;
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 1;
        }
    if (argv[optind] == NULL)
        sprintf(filename, "%s.out", argv[0]);
    else
        snprintf(filename, 256, "%s", argv[optind]);

    /* Calculate number of subarray requests each aggregator writes or reads.
     * Each original MPI process client forwards all its requests to one of
     * the I/O tasks. To run the original case, run 16 MPI processes.
     */
    nreqs = nvars * NCLIENTS / nprocs;
    nreqs++; /* one small variable at the beginning */

    /* Data partitioning is done along 2nd dimension */
    ncols = ncols_g / NCLIENTS;

    wlen = (MPI_Offset)nrows * ncols * (nreqs - 1) + nrows;
    if (rank == 0) {
        printf("Number of global variables = %d\n", nvars);
        printf("Each global variable is of size %d x %d bytes\n",nrows,ncols_g);
        printf("Each  local variable is of size %d x %d bytes\n",nrows,ncols);
        printf("Gap between the first 2 variables is of size %d bytes\n", gap);
        printf("Number of subarray types concatenated is %d\n", nreqs-1);
        printf("Each process makes a request of amount %lld bytes\n", wlen);
        printf("ROMIO hint set: cb_buffer_size = %s\n", cb_buffer_size);
        printf("ROMIO hint set: cb_nodes = %s\n", cb_nodes);
    }
    /* check 4-byte integer overflow */
    if (wlen > 2147483647) {
        if (rank == 0) {
            printf("Error: local write size %lld > INT_MAX.\n", wlen);
            printf("       Try increasing number of processes\n");
            printf("       or reduce the block size.\n");
            printf("       nrows=%d ncols=%d\n", nrows,ncols);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    blocklen = (int*) malloc(sizeof(int) * nreqs);
    displace = (MPI_Aint*) malloc(sizeof(MPI_Aint) * nreqs);

    /* User buffer consists of two noncontiguous spaces. To mimic this, we
     * allocate one space but add a gap in between
     */
    blocklen[0] = nrows; /* a small request of size nrows */
    blocklen[1] = nrows * ncols * (nreqs - 1);

    displace[0] = 0;
    displace[1] = nrows + gap;

    /* construct buffer datatype */
    err = MPI_Type_create_hindexed(2, blocklen, displace, MPI_BYTE, &bufType);
    ERR
    err = MPI_Type_commit(&bufType); ERR

    /* allocate I/O buffer */
    err = MPI_Type_size(bufType, &btype_size); ERR
    err = MPI_Type_get_extent(bufType, &lb, &buf_ext); ERR
    buf = (char*) calloc(buf_ext, 1);

    /* construct file type:
     * + there are nreqs subarrays, each uses a subarray datatype
     * + all subarray datatypes are concatenated into one to be used as fileview
     */
    subTypes = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * nreqs);

    /* first is the small variable at the beginning of the file */
    err = MPI_Type_contiguous(nrows, MPI_BYTE, &subTypes[0]); ERR
    blocklen[0] = 1;
    displace[0] = nrows * rank;

    for (i=1; i<nreqs; i++) {
        int gsizes[2], subsizes[2], starts[2];

        gsizes[0]   = nrows;
        gsizes[1]   = ncols * nprocs;
        subsizes[0] = nrows;
        subsizes[1] = ncols;
        starts[0]   = 0;
        starts[1]   = ncols * rank;
        err = MPI_Type_create_subarray(2, gsizes, subsizes, starts,
                       MPI_ORDER_C, MPI_BYTE, &subTypes[i]); ERR
        blocklen[i] = 1;
        displace[i] = (MPI_Aint)nrows * nprocs
                    + (MPI_Aint)gsizes[0] * gsizes[1] * (i - 1);
    }

    /* concatenate all subTypes into one datatype */
    err = MPI_Type_create_struct(nreqs, blocklen, displace, subTypes,
                                 &fileType); ERR
    err = MPI_Type_commit(&fileType); ERR

    for (i=0; i<nreqs; i++) {
        err = MPI_Type_free(&subTypes[i]); ERR
    }
    free(subTypes);
    free(displace);
    free(blocklen);

    /* check datatype extent and size */
    err = MPI_Type_get_extent(fileType, &lb, &file_ext); ERR
    err = MPI_Type_size(fileType, &ftype_size); ERR

    if (ftype_size != btype_size) {
        if (rank == 0)
            printf("Error: sizes of fileType and bufType mismatch (%d != %d)\n",
                   ftype_size, btype_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }
    if (verbose)
        printf("%2d: buf_ext=%ld btype_size=%d file_ext=%ld ftype_size=%d\n",
               rank,buf_ext,btype_size,file_ext,ftype_size);

    /* set hints to mimic Lustre striping size of 1MB and count of 4 on a UFS */
    MPI_Info_create(&info);
    MPI_Info_set(info, "cb_config_list", "*:*");
    MPI_Info_set(info, "cb_buffer_size", cb_buffer_size);
    MPI_Info_set(info, "cb_nodes", cb_nodes);

    mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh); ERR

    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", MPI_INFO_NULL);
    ERR

    MPI_Info_free(&info);

    /* write to the file */
    if (do_write) {
        MPI_Barrier(MPI_COMM_WORLD);
        timing[0] = MPI_Wtime();
        err = MPI_File_write_at_all(fh, 0, buf, 1, bufType, &status); ERR
        timing[0] = MPI_Wtime() - timing[0];
    }

    /* read from the file */
    if (do_read) {
        MPI_Barrier(MPI_COMM_WORLD);
        timing[1] = MPI_Wtime();
        err = MPI_File_read_at_all(fh, 0, buf, 1, bufType, &status); ERR
        timing[1] = MPI_Wtime() - timing[1];
    }

    err = MPI_File_close(&fh); ERR

    err = MPI_Type_free(&fileType); ERR
    err = MPI_Type_free(&bufType); ERR
    free(buf);

    MPI_Reduce(timing, max_timing, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("---------------------------------------------------------\n");
        if (do_write)
            printf("Time of collective write = %.2f sec\n", max_timing[0]);
        if (do_read)
            printf("Time of collective read  = %.2f sec\n", max_timing[1]);
        printf("---------------------------------------------------------\n");
    }

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}

