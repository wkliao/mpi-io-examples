/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 *
 * This program tests large requests made by each MPI process.
 * Both buffer and fileview data types are noncontiguous.
 *
 * The local buffer datatype comprises NVARS arrays. Each array is 2D of size
 * len x len.  A gap of size GAP can be added at the end of each dimension to
 * make the data type noncontiguous.
 *
 * The fileview of each process comprises NVARS subarrays.  Each global array
 * is 2D of size (len * psize[0]) x (len * psize[1]). Each local array is of
 * size (len - GAP)  x (len - GAP).
 *
 * Example output:
 *     % mpiexec -n 2 ./a.out -f output.dat
 *     Output file name = output.dat
 *     nprocs=2 nvars=1100 len=2048
 *     Expecting file size=18454933502 bytes (17600.0 MB, 17.2 GB)
 *     Each global variable is of size 8388608 bytes (8.0 MB)
 *     Each process writes 4609229900 bytes (4395.7 MB, 4.3 GB)
 *     ** For nonblocking I/O test, the amount is twice
 *     -------------------------------------------------------
 *     rank   1: gsize=4096 2048 start=2048    0 count=2047 2047
 *     rank   0: gsize=4096 2048 start=   0    0 count=2047 2047
 *     file   type size =      4190209 extent =      8388608
 *     buffer type size =   4609229900 extent =   4613734400
 *     -------------------------------------------------------
 *     Time of              collective write =  92.97 sec
 *     Time of             independent write = 144.64 sec
 *     Time of nonblocking  collective write = 431.02 sec
 *     Time of nonblocking independent write = 512.06 sec
 *     Time of              collective read  = 120.46 sec
 *     Time of             independent read  =  35.08 sec
 *     Time of nonblocking  collective read  = 459.52 sec
 *     Time of nonblocking independent read  = 473.05 sec
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <assert.h>

#include <mpi.h>

#define CHECK_MPI_ERROR(fname) \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d when calling %s: %s\n",__LINE__,fname,errorString); \
        nerrs++; \
        goto err_out; \
    }

#define CHECK_MPIO_ERROR(fname) { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d when calling %s: %s\n",__LINE__,fname,errorString); \
        nerrs++; \
        goto err_out; \
    } \
    else if (verbose) { \
        if (rank == 0) \
            printf("---- pass LINE %d of calling %s\n", __LINE__, fname); \
        fflush(stdout); \
        MPI_Barrier(MPI_COMM_WORLD); \
    } \
}

#define LEN   2048
#define GAP   1
#define NVARS 1100

int check_contents(int r_rank, int nvars, int len, int gap, char *buf, char *msg)
{
    size_t i, j, k, q;

    /* check the contents of read buffer */
    q = 0;
    for (i=0; i<nvars; i++) {
        for (j=0; j<len-gap; j++) {
            for (k=0; k<len-gap; k++) {
                char exp = (char)((r_rank + q) % 128);
                if (buf[q] != exp) {
                    printf("Error: %s [i=%zd j=%zd k=%zd] expect %d but got %d\n",
                           msg, i, j, k, exp, buf[q]);
                    return 1;
                }
                q++;
            }
            q += gap;
        }
        q += gap * len;
    }
    return 0;
}

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-hvwr | -n num | -l num | -g num ] -f file_name\n"
    "       [-h] Print this help\n"
    "       [-v] verbose mode\n"
    "       [-w] performs write only (default: both write and read)\n"
    "       [-r] performs read  only (default: both write and read)\n"
    "       [-n num] number of global variables (default: %d)\n"
    "       [-l num] length of dimensions X and Y each local variable (default: %d)\n"
    "       [-g num] gap at the end of each dimension (default: %d)\n"
    "        -f file_name: output file name\n";
    fprintf(stderr, help, argv0, NVARS, LEN, GAP);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    char filename[512];
    size_t i, buf_len;
    int ret, err, nerrs=0, rank, verbose, omode, nprocs, do_read, do_write;
    int nvars, len, gap, psize[2], gsize[2], count[2], start[2];
    char *buf;
    double timing, max_timing;
    MPI_File     fh;
    MPI_Datatype subType, filetype, buftype;
    MPI_Status   status;
    MPI_Offset fsize;
    MPI_Count type_size;
    int *array_of_blocklengths;
    MPI_Aint lb, extent, *array_of_displacements;
    MPI_Datatype *array_of_types;
    MPI_Request req[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* default values */
    len = LEN;
    gap = GAP;
    nvars = NVARS;
    do_write = 1;
    do_read  = 1;
    verbose = 0;
    filename[0] = '\0';

    /* get command-line arguments */
    while ((ret = getopt(argc, argv, "hvwrn:l:g:f:")) != EOF)
        switch(ret) {
            case 'v': verbose = 1;
                      break;
            case 'w': do_read = 0;
                      break;
            case 'r': do_write = 0;
                      break;
            case 'n': nvars = atoi(optarg);
                      break;
            case 'l': len = atoi(optarg);
                      break;
            case 'g': gap = atoi(optarg);
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

    array_of_blocklengths = (int*) malloc(sizeof(int) * nvars);
    array_of_displacements = (MPI_Aint*) malloc(sizeof(MPI_Aint) * nvars);
    array_of_types = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * nvars);

    /* Creates a division of processors in a cartesian grid */
    psize[0] = psize[1] = 0;
    err = MPI_Dims_create(nprocs, 2, psize);
    CHECK_MPI_ERROR("MPI_Dims_create");

    /* each 2D variable is of size gsizes[0] x gsizes[1] bytes */
    gsize[0] = len * psize[0];
    gsize[1] = len * psize[1];

    /* set subarray offset and length */
    start[0] = len * (rank / psize[1]);
    start[1] = len * (rank % psize[1]);
    count[0] = len - gap;   /* -1 to create holes */
    count[1] = len - gap;

    fsize = (MPI_Offset)gsize[0] * gsize[1] * nvars - (len+1);
    if (verbose) {
        buf_len = (size_t)nvars * (len-1) * (len-1);
        if (rank == 0) {
            printf("Output file name = %s\n", filename);
            printf("nprocs=%d nvars=%d len=%d\n", nprocs, nvars, len);
            printf("Expecting file size=%lld bytes (%.1f MB, %.1f GB)\n",
                   fsize*2, (float)fsize*2/1048576,(float)fsize*2/1073741824);
            printf("Each global variable is of size %d bytes (%.1f MB)\n",
                   gsize[0]*gsize[1],(float)gsize[0]*gsize[1]/1048576);
            printf("Each process writes %zd bytes (%.1f MB, %.1f GB)\n",
                   buf_len,(float)buf_len/1048576,(float)buf_len/1073741824);
            printf("** For nonblocking I/O test, the amount is twice\n");
            printf("-------------------------------------------------------\n");
        }
        printf("rank %3d: gsize=%4d %4d start=%4d %4d count=%4d %4d\n", rank,
               gsize[0],gsize[1],start[0],start[1],count[0],count[1]);
    }

    /* create 2D subarray datatype for fileview */
    err = MPI_Type_create_subarray(2, gsize, count, start, MPI_ORDER_C, MPI_BYTE, &filetype);
    CHECK_MPI_ERROR("MPI_Type_create_subarray");
    err = MPI_Type_commit(&filetype);
    CHECK_MPI_ERROR("MPI_Type_commit");

    MPI_Type_size_c(filetype, &type_size);
    lb = 0;
    MPI_Type_get_extent(filetype, &lb, &extent);
    if (verbose && rank == 0)
        printf("file   type size = %12lld extent = %12ld\n", type_size, extent);

    /* Create local buffer datatype: each 2D variable is of size len x len */
    gsize[0] = len;
    gsize[1] = len;
    start[0] = 0;
    start[1] = 0;
    count[0] = len - gap;  /* -1 to create holes */
    count[1] = len - gap;

    err = MPI_Type_create_subarray(2, gsize, count, start, MPI_ORDER_C, MPI_BYTE, &subType);
    CHECK_MPI_ERROR("MPI_Type_create_subarray");
    err = MPI_Type_commit(&subType);
    CHECK_MPI_ERROR("MPI_Type_commit");

    /* concatenate nvars subType into a buftype */
    for (i=0; i<nvars; i++) {
        array_of_blocklengths[i] = 1;
        array_of_displacements[i] = len*len*i;
        array_of_types[i] = subType;
    }

    /* create a buftype by concatenating nvars subTypes */
    err = MPI_Type_create_struct(nvars, array_of_blocklengths,
                                        array_of_displacements,
                                        array_of_types,
                                        &buftype);
    CHECK_MPI_ERROR("MPI_Type_create_struct");
    err = MPI_Type_commit(&buftype);
    CHECK_MPI_ERROR("MPI_Type_commit");
    err = MPI_Type_free(&subType);
    CHECK_MPI_ERROR("MPI_Type_free");

    free(array_of_blocklengths);
    free(array_of_displacements);
    free(array_of_types);

    MPI_Type_size_c(buftype, &type_size);
    lb = 0;
    MPI_Type_get_extent(buftype, &lb, &extent);
    if (verbose && rank == 0) {
        printf("buffer type size = %12lld extent = %12ld\n", type_size, extent);
        printf("-------------------------------------------------------\n");
    }
    fflush(stdout);

    /* allocate a local buffer */
    buf_len = (size_t)nvars * len * len;
    buf = (char*) malloc(buf_len);
    for (i=0; i<buf_len; i++) buf[i] = (char)((rank + i) % 128);

    /* open to create a file */
    omode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, omode, MPI_INFO_NULL, &fh);
    CHECK_MPIO_ERROR("MPI_File_open");

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
    CHECK_MPIO_ERROR("MPI_File_set_view");

    if (do_write) {
        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        /* MPI collective write */
        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_write_all(fh, buf, 1, buftype, &status);
        CHECK_MPIO_ERROR("MPI_File_write_all");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of              collective write = %.2f sec\n", max_timing);
        fflush(stdout);

        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        /* MPI independent write */
        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_write(fh, buf, 1, buftype, &status);
        CHECK_MPIO_ERROR("MPI_File_write");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of             independent write = %.2f sec\n", max_timing);
        fflush(stdout);

        /* MPI nonblocking collective write */
        char *buf2 = (char*) malloc(buf_len);
        for (i=0; i<buf_len; i++) buf2[i] = (char)((rank + i) % 128);

        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_iwrite_all(fh, buf, 1, buftype, &req[0]);
        CHECK_MPIO_ERROR("MPI_File_iwrite_all 1");

        err = MPI_File_iwrite_all(fh, buf2, 1, buftype, &req[1]);
        CHECK_MPIO_ERROR("MPI_File_iwrite_all 2");

        err = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        CHECK_MPIO_ERROR("MPI_Waitall");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of nonblocking  collective write = %.2f sec\n", max_timing);
        fflush(stdout);

        /* MPI nonblocking independent write */
        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_iwrite(fh, buf, 1, buftype, &req[0]);
        CHECK_MPIO_ERROR("MPI_File_iwrite 1");

        err = MPI_File_iwrite(fh, buf2, 1, buftype, &req[1]);
        CHECK_MPIO_ERROR("MPI_File_iwrite 2");

        err = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        CHECK_MPIO_ERROR("MPI_Waitall");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of nonblocking independent write = %.2f sec\n", max_timing);
        fflush(stdout);

        free(buf2);
    }

    if (do_read) {
        /* change fileview, rank i used (i+1)'s fileview */
        int r_rank = (rank == nprocs - 1) ? 0 : rank + 1;

        err = MPI_Type_free(&filetype);
        CHECK_MPI_ERROR("MPI_Type_free");

        /* each 2D variable is of size gsizes[0] x gsizes[1] bytes */
        gsize[0] = len * psize[0];
        gsize[1] = len * psize[1];

        /* set subarray offset and length */
        start[0] = len * (r_rank / psize[1]);
        start[1] = len * (r_rank % psize[1]);
        count[0] = len - gap;   /* -1 to create holes */
        count[1] = len - gap;

        /* create 2D subarray datatype for fileview */
        err = MPI_Type_create_subarray(2, gsize, count, start, MPI_ORDER_C, MPI_BYTE, &filetype);
        CHECK_MPI_ERROR("MPI_Type_create_subarray");
        err = MPI_Type_commit(&filetype);
        CHECK_MPI_ERROR("MPI_Type_commit");

        /* set the file view */
        err = MPI_File_set_view(fh, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
        CHECK_MPIO_ERROR("MPI_File_set_view");

        /* reset contents of read buffer */
        for (i=0; i<buf_len; i++) buf[i] = -1;

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_read_all(fh, buf, 1, buftype, &status);
        CHECK_MPIO_ERROR("MPI_File_read_all");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of              collective read  = %.2f sec\n", max_timing);
        fflush(stdout);

        err += check_contents(r_rank, nvars, len, gap, buf, "MPI_File_read_all");
        if (err != 0) goto err_out;

        /* MPI independent read */
        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        /* reset contents of read buffer */
        for (i=0; i<buf_len; i++) buf[i] = -1;

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_read(fh, buf, 1, buftype, &status);
        CHECK_MPIO_ERROR("MPI_File_read");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of             independent read  = %.2f sec\n", max_timing);
        fflush(stdout);

        err += check_contents(r_rank, nvars, len, gap, buf, "MPI_File_read");
        if (err != 0) goto err_out;

        char *buf2 = (char*) malloc(buf_len);

        /* reset contents of read buffer */
        for (i=0; i<buf_len; i++) buf[i] = buf2[i] = -1;

        /* MPI nonblocking collective read */
        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();
        err = MPI_File_iread_all(fh, buf, 1, buftype, &req[0]);
        CHECK_MPIO_ERROR("MPI_File_iread_all 1");

        err = MPI_File_iread_all(fh, buf2, 1, buftype, &req[1]);
        CHECK_MPIO_ERROR("MPI_File_iread_all 2");

        err = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        CHECK_MPIO_ERROR("MPI_Waitall");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of nonblocking  collective read  = %.2f sec\n", max_timing);
        fflush(stdout);

        err += check_contents(r_rank, nvars, len, gap, buf, "MPI_File_iread_all 1");
        if (err != 0) goto err_out;

        err += check_contents(r_rank, nvars, len, gap, buf2, "MPI_File_iread_all 2");
        if (err != 0) goto err_out;

        /* MPI nonblocking independent read */

        /* reset contents of read buffer */
        for (i=0; i<buf_len; i++) buf[i] = buf2[i] = -1;

        err = MPI_File_seek(fh, 0, MPI_SEEK_SET);
        CHECK_MPIO_ERROR("MPI_File_seek");

        MPI_Barrier(MPI_COMM_WORLD);
        timing = MPI_Wtime();

        err = MPI_File_iread(fh, buf, 1, buftype, &req[0]);
        CHECK_MPIO_ERROR("MPI_File_iread 1");

        err = MPI_File_iread(fh, buf2, 1, buftype, &req[1]);
        CHECK_MPIO_ERROR("MPI_File_iread 2");

        err = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        CHECK_MPIO_ERROR("MPI_Waitall");

        timing = MPI_Wtime() - timing;
        MPI_Reduce(&timing, &max_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            printf("Time of nonblocking independent read  = %.2f sec\n", max_timing);
        fflush(stdout);

        err += check_contents(r_rank, nvars, len, gap, buf, "MPI_File_iread 1");
        if (err != 0) goto err_out;

        err += check_contents(r_rank, nvars, len, gap, buf2, "MPI_File_iread 2");
        if (err != 0) goto err_out;

        free(buf2);
    }

    err = MPI_File_close(&fh);
    CHECK_MPIO_ERROR("MPI_File_close");

    free(buf);

    err = MPI_Type_free(&filetype);
    CHECK_MPI_ERROR("MPI_Type_free");
    err = MPI_Type_free(&buftype);
    CHECK_MPI_ERROR("MPI_Type_free");

err_out:
    MPI_Finalize();
    return (nerrs > 0);
}
