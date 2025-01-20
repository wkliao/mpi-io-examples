/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2025, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * Evaluate performane of all-to-many personalized communication implemented
 * with MPI_Alltoallw() and MPI_Issend()/MPI_Irecv(). The communication pattern
 * uses a trace from one of PnetCDF's benchmark programs, WRF-IO, running the
 * following commands on 8 CPU nodes, 128 MPI processes each.
 *    srun -n 1024 wrf_io -l 5200 -w 7600 output.nc
 * It also used Lustre striping count 8, striping size 8 MB, MPI-IO hints of
 * cb_nodes 32 and cb_buffer_size 16 MB.
 *
 * To compile:
 *   % mpicc -O2 trace_alltomany.c -o trace_alltomany
 *
 * Usage: this program requires an input file as the argument.
 *        A trace file 'trace_1024p_253n.dat.gz' is provided. Run command
 *        'gunzip trace_1024p_253n.dat.gz' before using it.
 *        This program can run with 1024 or less number of MPI processes.
 *
 * Example run command and output on screen:
 *   % mpiexec -n 1024 ./trace_alltomany
 *     number of MPI processes         = 1024
 *     number of iterations            = 253
 *     Comm amount using MPI_Issend/Irecv = 129074.16 MB
 *     Time for using MPI_Issend/Irecv    = 2.81 sec
 *             Time bucket[1] = 0.31 sec
 *             Time bucket[2] = 0.31 sec
 *             Time bucket[3] = 0.29 sec
 *             Time bucket[4] = 0.30 sec
 *             Time bucket[5] = 0.29 sec
 *             Time bucket[6] = 0.29 sec
 *             Time bucket[7] = 0.29 sec
 *             Time bucket[8] = 0.30 sec
 *             Time bucket[9] = 0.19 sec
 *     Comm amount using MPI_alltoallw    = 129074.16 MB
 *     Time for using MPI_alltoallw       = 7.48 sec
 *             Time bucket[1] = 0.83 sec
 *             Time bucket[2] = 0.79 sec
 *             Time bucket[3] = 0.77 sec
 *             Time bucket[4] = 0.79 sec
 *             Time bucket[5] = 0.77 sec
 *             Time bucket[6] = 0.77 sec
 *             Time bucket[7] = 0.80 sec
 *             Time bucket[8] = 0.77 sec
 *             Time bucket[9] = 0.53 sec
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <mpi.h>

#define NTIMES 253
#define NPROCS 1024

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        goto err_out; \
    }

typedef struct {
    int nprocs;  /* number of peers with non-zero amount */
    int *ranks;  /* rank IDs of peers with non-zero amount */
    int *amnts;  /* amounts of peers with non-zero amount */
} trace;

/* all-to-many personalized communication by calling MPI_Alltoallw() */
void run_alltoallw(int     ntimes,
                   trace  *sender,
                   trace  *recver,
                   char  **sendBuf,
                   char  **recvBuf)
{
    int i, j, err, nprocs, rank, bucket_len;
    int *sendCounts, *recvCounts, *sendDisps, *recvDisps;
    MPI_Datatype *sendTypes, *recvTypes;
    MPI_Offset amnt, sum_amnt;
    double start_t, end_t, timing[10], maxt[10];

    for (i=0; i<10; i++) timing[i]=0;
    bucket_len = ntimes / 10;
    if (ntimes % 10) bucket_len++;

    MPI_Barrier(MPI_COMM_WORLD);
    timing[0] = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sendTypes = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * nprocs * 2);
    recvTypes = sendTypes + nprocs;
    for (i=0; i<nprocs * 2; i++) sendTypes[i] = MPI_BYTE;

    sendCounts = (int*) malloc(sizeof(int) * nprocs * 4);
    recvCounts = sendCounts + nprocs;
    sendDisps  = recvCounts + nprocs;
    recvDisps  = sendDisps  + nprocs;

    start_t = MPI_Wtime();
    amnt = 0;
    for (j=0; j<ntimes; j++) {
        int disp, peer;
        /* set up sendcounts and sdispls arguments of MPI_Alltoallw() */
        for (i=0; i<nprocs*4; i++)
            sendCounts[i] = 0;

        disp = 0;
        for (i=0; i<sender[j].nprocs; i++) {
            peer = sender[j].ranks[i];
            if (peer >= nprocs) continue;
            sendCounts[peer] = sender[j].amnts[i];
            sendDisps[peer] = disp;
            disp += sendCounts[peer];
        }
        disp = 0;
        for (i=0; i<recver[j].nprocs; i++) {
            peer = recver[j].ranks[i];
            if (peer >= nprocs) continue;
            recvCounts[peer] = recver[j].amnts[i];
            recvDisps[peer] = disp;
            disp += recvCounts[peer];
        }
        amnt += disp;

        err = MPI_Alltoallw(sendBuf[j], sendCounts, sendDisps, sendTypes,
                            recvBuf[j], recvCounts, recvDisps, recvTypes,
                            MPI_COMM_WORLD); ERR

        /* record timing */
        if (j > 0 && j % bucket_len == 0) {
            end_t = MPI_Wtime();
            timing[j / bucket_len] = end_t - start_t;
            start_t = end_t;
        }
    }
    end_t = MPI_Wtime();
    timing[9] = end_t - start_t;
    timing[0] = end_t - timing[0]; /* end-to-end time */

err_out:
    free(sendTypes);
    free(sendCounts);

    MPI_Reduce(&timing, &maxt, 10, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&amnt, &sum_amnt, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Comm amount using MPI_alltoallw    = %.2f MB\n",
               (float)sum_amnt/1048576.0);
        printf("Time for using MPI_alltoallw       = %.2f sec\n", maxt[0]);
        for (i=1; i<10; i++)
            printf("\tTime bucket[%d] = %.2f sec\n", i, maxt[i]);
        fflush(stdout);
    }
}

/* all-to-many personalized communication by calling MPI_Issend/Irecv() */
void run_async_send_recv(int    ntimes,
                         trace  *sender,
                         trace  *recver,
                         char  **sendBuf,
                         char  **recvBuf)
{
    char *sendPtr, *recvPtr;
    int i, j, err, nprocs, rank, nreqs, bucket_len;
    MPI_Request *reqs;
    MPI_Status *st;
    MPI_Offset amnt, sum_amnt;
    double start_t, end_t, timing[10], maxt[10];

    for (i=0; i<10; i++) timing[i]=0;
    bucket_len = ntimes / 10;
    if (ntimes % 10) bucket_len++;

    MPI_Barrier(MPI_COMM_WORLD);
    timing[0] = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* allocate MPI_Request and MPI_Status arrays */
    reqs = (MPI_Request*) malloc(sizeof(MPI_Request) * 2 * nprocs);
    st = (MPI_Status*) malloc(sizeof(MPI_Status) * 2 * nprocs);

    start_t = MPI_Wtime();
    amnt = 0;
    for (j=0; j<ntimes; j++) {
        nreqs = 0;

        /* receivers */
        recvPtr = recvBuf[j];
        for (i=0; i<recver[j].nprocs; i++) {
            if (recver[j].ranks[i] >= nprocs) continue;
            err = MPI_Irecv(recvPtr, recver[j].amnts[i], MPI_BYTE,
                            recver[j].ranks[i], 0, MPI_COMM_WORLD,
                            &reqs[nreqs++]);
            ERR
            recvPtr += recver[j].amnts[i];
            amnt += recver[j].amnts[i];
        }
        /* senders */
        sendPtr = sendBuf[j];
        for (i=0; i<sender[j].nprocs; i++) {
            if (sender[j].ranks[i] >= nprocs) continue;
            err = MPI_Issend(sendPtr, sender[j].amnts[i], MPI_BYTE,
                             sender[j].ranks[i], 0, MPI_COMM_WORLD,
                             &reqs[nreqs++]);
            ERR
            sendPtr += sender[j].amnts[i];
        }

        err = MPI_Waitall(nreqs, reqs, st); ERR

        if (j > 0 && j % bucket_len == 0) {
            end_t = MPI_Wtime();
            timing[j / bucket_len] = end_t - start_t;
            start_t = end_t;
        }
    }
    end_t = MPI_Wtime();
    timing[9] = end_t - start_t;
    timing[0] = end_t - timing[0]; /* end-to-end time */

err_out:
    free(st);
    free(reqs);

    MPI_Reduce(&timing, &maxt, 10, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&amnt, &sum_amnt, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Comm amount using MPI_Issend/Irecv = %.2f MB\n",
               (float)sum_amnt/1048576.0);
        printf("Time for using MPI_Issend/Irecv    = %.2f sec\n", maxt[0]);
        for (i=1; i<10; i++)
            printf("\tTime bucket[%d] = %.2f sec\n", i, maxt[i]);
        fflush(stdout);
    }
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    int i, j, fd, rank, nprocs, ntimes;
    char **sendBuf, **recvBuf;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (rank == 0) printf("Input trace file is required\n");
        goto err_out;
    }

    if (nprocs > NPROCS) {
        if (rank == 0) printf("Number of MPI processes must be <= %d\n",NPROCS);
        goto err_out;
    }

    ntimes = NTIMES;
    if (rank == 0) {
        printf("number of MPI processes         = %d\n", nprocs);
        printf("number of iterations            = %d\n", ntimes);
    }

    if ((fd = open(argv[1], O_RDONLY, 0600)) == -1) {
        printf("Error! open() failed %s (error: %s)\n",argv[1],strerror(errno));
        goto err_out;
    }

    /* trace file was generated using 1024 MPI processes, running 253
     * iterations. nprocs can be less than 1024
     */
    /* read nprocs */
    int in_nprocs;
    read(fd, &in_nprocs, sizeof(int));
    assert(in_nprocs == NPROCS);

    /* read ntimes */
    int in_ntimes;
    read(fd, &in_ntimes, sizeof(int));
    assert(in_ntimes == NTIMES);

    /* read block_lens[NPROCS] */
    int *block_lens = (int*) malloc(sizeof(int) * NPROCS);
    read(fd, block_lens, sizeof(int) * NPROCS);

    /* read block 'rank' */
    int *file_block = (int*) malloc(sizeof(int) * block_lens[rank]);
    off_t off=0;
    for (i=0; i<rank; i++) off += block_lens[i];
    off *= sizeof(int);
    lseek(fd, off, SEEK_CUR);
    read(fd, file_block, sizeof(int) * block_lens[rank]);

    free(block_lens);

    /* close input file */
    close(fd);

    /* allocate buffer for storing pairwise communication amounts */
    trace *sender = (trace*) malloc(sizeof(trace) * NTIMES);
    trace *recver = (trace*) malloc(sizeof(trace) * NTIMES);

    int *nonzero_nprocs, *ptr=file_block;

    /* populate sender communication pattern */
    nonzero_nprocs = ptr;
    ptr += NTIMES;
    for (i=0; i<NTIMES; i++) {
        sender[i].nprocs = nonzero_nprocs[i];
        sender[i].ranks  = ptr;
        ptr += nonzero_nprocs[i];
        sender[i].amnts  = ptr;
        ptr += nonzero_nprocs[i];
    }

    /* populate receiver communication pattern */
    nonzero_nprocs = ptr;
    ptr += NTIMES;
    for (i=0; i<NTIMES; i++) {
        recver[i].nprocs = nonzero_nprocs[i];
        recver[i].ranks  = ptr;
        ptr += nonzero_nprocs[i];
        recver[i].amnts  = ptr;
        ptr += nonzero_nprocs[i];
    }

    /* allocate send and receive message buffers */
    sendBuf = (char**) malloc(sizeof(char*) * ntimes);
    for (i=0; i<ntimes; i++) {
        size_t amnt=0;
        for (j=0; j<sender[i].nprocs; j++) {
            if (sender[i].ranks[j] >= nprocs) break;
            amnt += sender[i].amnts[j];
        }
        sendBuf[i] = (amnt == 0) ? NULL : (char*) malloc(amnt);
        for (j=0; j<amnt; j++) sendBuf[i][j] = (rank+j)%128;
    }

    /* recv buffer is reused in each iteration */
    recvBuf = (char**) malloc(sizeof(char*) * ntimes);
    size_t recv_amnt = 0;
    for (i=0; i<ntimes; i++) {
        size_t amnt=0;
        for (j=0; j<recver[i].nprocs; j++) {
            if (recver[i].ranks[j] >= nprocs) break;
            amnt += recver[i].amnts[j];
        }
        if (amnt > recv_amnt) recv_amnt = amnt;
    }
    recvBuf[0] = (recv_amnt == 0) ? NULL : (char*) malloc(recv_amnt);
    for (i=1; i<ntimes; i++) recvBuf[i] = recvBuf[0];

    for (i=0; i<3; i++) {

        /* perform all-to-many communication */
        MPI_Barrier(MPI_COMM_WORLD);
        run_async_send_recv(ntimes, sender, recver, sendBuf, recvBuf);

        /* perform all-to-many communication */
        MPI_Barrier(MPI_COMM_WORLD);
        run_alltoallw(ntimes, sender, recver, sendBuf, recvBuf);

    }

    for (i=0; i<ntimes; i++) if (sendBuf[i] != NULL) free(sendBuf[i]);
    free(sendBuf);
    if (recvBuf[0] != NULL) free(recvBuf[0]);
    free(recvBuf);
    free(sender);
    free(recver);
    free(file_block);

err_out:
    MPI_Finalize();
    return 0;
}

