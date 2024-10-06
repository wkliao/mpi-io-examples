/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * This program tests collective write and read calls with a fileview
 * micmicking multiple variables stored in the file and each variable is
 * partitioned among processes in a 2D block-block fashion.
 *
 * Each variable is a 3D array of type 'int'. Only the Y and X dimensions are
 * partitioned, while the Z dimension is not. The number of subarray datatypes
 * are concatenated into one filetype, which is used to set the fileview.
 * Similarly the buffer type is concatenated from multiple subarrays if ghost
 * cell option is used.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy(), strdup() */
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

int verbose;

#define ZDIMS 2

int create_fileType(MPI_Comm      comm,
                    int           nvars,
                    int           len,
                    MPI_Datatype *fileType)
{
    int i, err, nerrs=0, rank, nprocs;
    int psizes[2], sizes[3], subsizes[3], starts[3];
    int type_size, *blks=NULL;
    MPI_Aint *disp=NULL, lb, extent;
    MPI_Datatype subType;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* calculate number of processes along each dimension */
    psizes[0] = psizes[1] = 0;
    MPI_Dims_create(nprocs, 2, psizes);

    /* calculate starting offsets and lengths of a subarray */
    sizes[0]    = ZDIMS;
    sizes[1]    = len * psizes[0];
    sizes[2]    = len * psizes[1];
    subsizes[0] = ZDIMS;
    subsizes[1] = len;
    subsizes[2] = len;
    starts[0]   = 0;
    starts[1]   = len * (rank / psizes[1]);
    starts[2]   = len * (rank % psizes[1]);
    err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &subType); ERR

    if (verbose && rank == 0) {
        printf("Each global variable is of size     %d x %d x %d (int) = %zd\n",
               sizes[0], sizes[1], sizes[2], sizeof(int)*sizes[0]*sizes[1]*sizes[2]);
        printf("process dimension psizes:           %d %d\n", psizes[0], psizes[1]);
    }
    if (verbose)
        printf("%d: sizes=%d %d %d subsizes=%d %d %d starts=%d %d %d\n", rank,
               sizes[0],sizes[1],sizes[2], subsizes[0],subsizes[1],subsizes[2],
               starts[0],starts[1],starts[2]);

    /* calculate file starting file offsets of all variables */
    disp = (MPI_Aint*) malloc(sizeof(MPI_Aint) * nvars);
    blks = (int*) malloc(sizeof(int) * nvars);
    for (i=0; i<nvars; i++) {
        disp[i] = sizeof(int) * i * sizes[0] * sizes[1] * sizes[2];
        blks[i] = 1;
        if (verbose && rank == 0)
            printf("disp[%2d]=%ld\n",i,disp[i]);
    }

    /* concate nvars subTypes into one fileType */
    err = MPI_Type_create_hindexed(nvars, blks, disp, subType, fileType); ERR
    err = MPI_Type_commit(fileType); ERR
    err = MPI_Type_free(&subType); ERR
    if (disp != NULL) free(disp);
    if (blks != NULL) free(blks);

    /* check fileType size and extent */
    MPI_Type_size(*fileType, &type_size);
    lb = 0;
    MPI_Type_get_extent(*fileType, &lb, &extent);
    if (verbose && rank == 0)
        printf("%d: file type size=%d extent=%ld lb=%ld\n", rank, type_size, extent, lb);

err_out:
    return nerrs;
}

int create_bufType(MPI_Comm        comm,
                    int            nvars,
                    int            len,
                    int            ngcells,
                    int          **buf,
                    MPI_Datatype  *bufType)
{
    int i, err, nerrs=0, rank;
    int sizes[3], subsizes[3], starts[3];
    int type_size, *blks=NULL;
    MPI_Aint *disp=NULL, lb, extent;
    MPI_Datatype subType;

    MPI_Comm_rank(comm, &rank);

    /* calculate starting offsets and lengths of a subarray */
    sizes[0]    = ZDIMS;
    sizes[1]    = len + ngcells * 2;
    sizes[2]    = len + ngcells * 2;
    subsizes[0] = ZDIMS;
    subsizes[1] = len;
    subsizes[2] = len;
    starts[0]   = 0;
    starts[1]   = ngcells;
    starts[2]   = ngcells;
    err = MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                   MPI_INT, &subType); ERR
    if (verbose && rank == 0)
        printf("local variable sizes=%d %d %d subsizes=%d %d %d starts=%d %d %d\n",
               sizes[0],sizes[1],sizes[2], subsizes[0],subsizes[1],subsizes[2],
               starts[0],starts[1],starts[2]);

    /* calculate file starting file offsets of all variables */
    disp = (MPI_Aint*) malloc(sizeof(MPI_Aint) * nvars);
    blks = (int*) malloc(sizeof(int) * nvars);
    for (i=0; i<nvars; i++) {
        MPI_Get_address(buf[i], &disp[i]);
        blks[i] = 1;
    }

    /* concatenate nvars subTypes into bufType */
    err = MPI_Type_create_hindexed(nvars, blks, disp, subType, bufType); ERR
    err = MPI_Type_commit(bufType); ERR
    err = MPI_Type_free(&subType); ERR

    if (disp != NULL) free(disp);
    if (blks != NULL) free(blks);

    /* check bufType size and extent */
    MPI_Type_size(*bufType, &type_size);
    lb = 0;
    MPI_Type_get_extent(*bufType, &lb, &extent);
    if (verbose && rank == 0)
        printf("buffer type size = %d extent = %ld\n", type_size, extent);

err_out:
    return nerrs;
}

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-hvrc | -n num | -l len | -g num | -a num | -s num] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-v] verbose mode\n"
    "       [-r] perform read operations after writes\n"
    "       [-c] make user buffer contiguous and no ghost cells \n"
    "       [-n num] number of variables to be written\n"
    "       [-l len] length of local X and Y dimension sizes\n"
    "       [-g num] number of ghost cells\n"
    "       [-a num] set cb_nodes hint\n"
    "       [-s num] set cb_buffer_size hint\n"
    "        -f filename: output file name\n";
    fprintf(stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    extern char *optarg;
    char filename[256], *cb_nodes=NULL, *cb_buffer_size=NULL;
    int i, j, k, z, cube, do_read;
    int err, nerrs=0, rank, nprocs, mode, nvars, len, xlen;
    int **buf=NULL, ngcells, max_nerrs, buf_contig;
    double timing[2], max_timing[2];
    MPI_Datatype bufType=MPI_INT, fileType;
    MPI_File fh;
    MPI_Status status;
    MPI_Info info = MPI_INFO_NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    verbose     = 0;
    do_read     = 0;
    buf_contig  = 0;
    nvars       = 2;     /* default number of variables */
    len         = 10;    /* default dimension size */
    ngcells     = 2;     /* number of ghost cells */
    filename[0] = '\0';

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hvrcn:l:g:a:s:f:")) != EOF)
        switch(i) {
            case 'v': verbose = 1;
                      break;
            case 'r': do_read = 1;
                      break;
            case 'c': buf_contig = 1;
                      break;
            case 'n': nvars = atoi(optarg);
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'g': ngcells = atoi(optarg);
                      break;
            case 'a': cb_nodes = strdup(optarg);
                      break;
            case 's': cb_buffer_size = strdup(optarg);
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

    if (buf_contig == 1) ngcells = 0;

    if (cb_nodes != NULL || cb_buffer_size != NULL) {
        MPI_Info_create(&info);
        if (cb_nodes != NULL)
            MPI_Info_set(info, "cb_nodes", cb_nodes);
        if (cb_buffer_size != NULL)
            MPI_Info_set(info, "cb_buffer_size", cb_buffer_size);
    }

    if (verbose && rank == 0) {
        printf("Number of MPI processes:            %d\n",nprocs);
        printf("Number of varaibles:                %d\n",nvars);
        printf("Each  local variable is of size     %d x %d x %d (int) = %zd\n",
               ZDIMS, len, len, sizeof(int)*ZDIMS*len*len);
        printf("Number of ghost cells is            %d\n", ngcells);
        if (cb_nodes != NULL)
            printf("Set MPI-IO hint 'cb_nodes' to       %s\n", cb_nodes);
        if (cb_buffer_size != NULL)
            printf("Set MPI-IO hint 'cb_buffer_size' to %s\n", cb_buffer_size);
    }

    /* create file datatype */
    err = create_fileType(MPI_COMM_WORLD, nvars, len, &fileType);
    if (err != 0) {
        nerrs++;
        goto err_out;
    }

    /* allocate I/O buffers */
    buf = (int**)malloc(sizeof(int*) * nvars);
    xlen = len + ngcells * 2;
    cube = ZDIMS * xlen * xlen;
    if (buf_contig) {
        buf[0] = (int*) malloc(sizeof(int) * cube * nvars);
        for (k=1; k<nvars; k++)
            buf[k] = buf[k-1] + cube;
    }
    else {
        for (k=0; k<nvars; k++)
            buf[k] = (int*) malloc(sizeof(int) * cube);
    }

    /* initialize contents of buffer */
    for (k=0; k<nvars; k++) {
        for (i=0; i<cube; i++)
            buf[k][i] = -1;

        for (z=0; z<ZDIMS ; z++)
            for (i=ngcells; i<len+ngcells ; i++)
                for (j=ngcells; j<len+ngcells ; j++)
                     buf[k][z*xlen*xlen + i*xlen+j] = rank;
    }

    if (!buf_contig) {
        /* create buffer datatype */
        err = create_bufType(MPI_COMM_WORLD, nvars, len, ngcells, buf, &bufType);
        if (err != 0) {
            nerrs++;
            goto err_out;
        }
    }

    /* open file */
    mode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, info, &fh); ERR

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, fileType, "native", MPI_INFO_NULL);
    ERR

    /* write to the file */
    MPI_Barrier(MPI_COMM_WORLD);
    timing[0] = MPI_Wtime();
    if (buf_contig) {
        err = MPI_File_write_all(fh, buf[0], cube*nvars, bufType, &status);
        ERR
    }
    else {
        err = MPI_File_write_all(fh, MPI_BOTTOM, 1, bufType, &status);
        ERR
    }
    timing[0] = MPI_Wtime() - timing[0];

    if (!do_read) goto verify_err;

    /* reset read buffer to all -1s */
    for (k=0; k<nvars; k++) {
        for (i=0; i<cube; i++)
            buf[k][i] = -1;
    }

    /* reset file pointer */
    err = MPI_File_seek(fh, 0, MPI_SEEK_SET); ERR

    timing[1] = MPI_Wtime();
    /* read from the file */
    if (buf_contig) {
        err = MPI_File_read_all(fh, buf[0], cube*nvars, bufType, &status);
        ERR
    }
    else {
        err = MPI_File_read_all(fh, MPI_BOTTOM, 1, bufType, &status);
        ERR
    }
    timing[1] = MPI_Wtime() - timing[1];

    /* check contents of read buffer */
    for (k=0; k<nvars; k++) {
        for (z=0; z<ZDIMS ; z++)
            for (i=0; i<xlen ; i++)
                for (j=0; j<xlen ; j++) {
                    int exp = rank;
                    if (i < ngcells || i >= len+ngcells ||
                        j < ngcells || j >= len+ngcells)
                        exp = -1;
                    if (buf[k][z*xlen*xlen + i*xlen+j] != exp) {
                        printf("Error: buf[%d][%d][%d][%d] expect %d but got %d\n",
                               k, z, i, j, exp, buf[k][z*cube + i*xlen+j]);
                        nerrs++;
                        goto verify_err;
                    }
                }
    }

verify_err:
    if (bufType != MPI_INT)
        MPI_Type_free(&bufType);
    MPI_Type_free(&fileType);

    if (info != MPI_INFO_NULL) MPI_Info_free(&info);
    err = MPI_File_close(&fh); ERR

    if (buf_contig)
        free(buf[0]);
    else {
        for (i=0; i<nvars; i++)
            free(buf[i]);
    }
    free(buf);

    MPI_Allreduce(&nerrs, &max_nerrs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Reduce(timing, max_timing, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (max_nerrs == 0 && rank == 0) {
        printf("Number of MPI processes:             %d\n", nprocs);
        printf("Number of variables:                 %d\n", nvars);
        printf("Size of each variables:              %d x %d (int)\n", len, len);
        printf("User buffer is contiguous:           %s\n", (buf_contig)?"yes":"no");
        printf("Number of ghost cells on both sizes: %d\n", ngcells);
        double amnt, amntM, amntG;
        amnt = (double)nprocs * nvars * ZDIMS * len * len * sizeof(int);
        amntM = amnt / 1048576.0;
        amntG = amnt / 1073741824.0;
        printf("Total write amount:                  %.0f B, %.2f MB, %.2f GB\n",
               amnt, amntM, amntG);
        printf("Time of collective write:            %.2f sec\n", max_timing[0]);
        printf("Write bandwidth:                     %.2f MB/sec, %.2f GB/sec\n",
               amntM/max_timing[0], amntG/max_timing[0]);
        if (do_read) {
            printf("Total read amount:                   %.0f B, %.2f MB, %.2f GB\n",
                   amnt, amntM, amntG);
            printf("Time of collective read:             %.2f sec\n", max_timing[1]);
            printf("Read  bandwidth:                     %.2f MB/sec, %.2f GB/sec\n",
                   amntM/max_timing[1], amntG/max_timing[1]);
        }
    }

err_out:
    if (cb_nodes != NULL) free(cb_nodes);
    if (cb_buffer_size != NULL) free(cb_buffer_size);
    MPI_Finalize();
    return (nerrs > 0);
}

