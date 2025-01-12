/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2025, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * Evaluate performane of all-to-many personalized communication implemented
 * with MPI_Alltoallw() and MPI_Issend()/MPI_Irecv().
 *
 * To compile:
 *   % mpicc -O2 alltoallw.c -o alltoallw
 *
 * Usage:
 *   % ./alltoallw -h
 *   Usage: alltoallw [OPTION]
 *      [-h] Print this help message
 *      [-v] Verbose mode (default: no)
 *      [-d] Debug mode to check receive buffer contents (default: no)
 *      [-n num] number of iterations (default: 1)
 *      [-r num] every ratio processes is a receiver (default: 1)
 *      [-l num] receive amount per iteration (default: 8 MB)
 *      [-g num] gap between 2 consecutive send/recv buffers (default: 4 int)
 *
 * Example run command and output on screen:
 *   % mpiexec -n 2048 ./alltoallw -n 253 -r 32
 *   number of MPI processes         = 2048
 *   number of iterations            = 253
 *   numbe of receivers              = 64
 *   individual message length       = 4096 bytes
 *   send/recv buffer gap            = 4 int(s)
 *   Recv amount per iteration       = 8388608 bytes
 *   Time for using MPI_alltoallw    = 53.60 sec
 *   Time for using MPI_Issend/Irecv = 2.59 sec
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include <mpi.h>

static int verbose;
static int debug;

#define ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__,errorString); \
        goto err_out; \
    }

/* initilized the contents of send buffer */
void initialize_send_buf(int  ntimes,
                         int  num_recvers,
                         int  len,
                         int  gap,
                         int *sendBuf)
{
    int i, j, k, m, nprocs, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (i=0; i<(len + gap)*ntimes*num_recvers; i++)
        sendBuf[i] = -2;
    m = 0;
    for (i=0; i<ntimes; i++) {
    for (j=0; j<num_recvers; j++) {
    for (k=0; k<len; k++) {
        sendBuf[m++] = rank;
    }
    m += gap;
    }
    }
}

/* initilized the contents of receive buffer */
void initialize_recv_buf(int  len,
                         int  gap,
                         int *recvBuf)
{
    int i, nprocs;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (i=0; i<(len + gap)*nprocs; i++)
        recvBuf[i] = -3;
}

/* check if the contents of receive buffer are correct */
int check_recv_buf(char *comm_op,
                   int   len,
                   int   gap,
                   int  *recvBuf)
{
    int i, j, k, expect, err=0, nprocs, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    k = 0;
    for (i=0; i<nprocs; i++) {
    for (j=0; j<len+gap; j++) {
        expect = (i == rank) ? -3 : ((j < len) ? i : -3);
        if (recvBuf[k] != expect) {
            printf("Error(%s): rank %d i=%d j=%d expect %d but got %d\n",
                   comm_op, rank, i, j, expect, recvBuf[k]);
            goto err_out;
        }
        k++;
    }
    }
err_out:
    return err;
}

/* all-to-many personalized communication by calling MPI_Alltoallw() */
void run_alltoallw(int  ntimes,
                   int  ratio,
                   int  is_receiver,
                   int  len,
                   int  gap,
                   int *sendBuf,
                   int *recvBuf)
{
    int *sendPtr;
    int i, j, err, nprocs, rank, num_recvers;
    int *sendCounts, *recvCounts, *sendDisps, *recvDisps;
    MPI_Datatype *sendTypes, *recvTypes;
    double timing, maxt;

    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    num_recvers = nprocs/ ratio;

    sendTypes = (MPI_Datatype*) malloc(sizeof(MPI_Datatype) * nprocs * 2);
    recvTypes = sendTypes + nprocs;
    for (i=0; i<nprocs * 2; i++) sendTypes[i] = MPI_INT;

    sendCounts = (int*) calloc(nprocs * 2, sizeof(int));
    recvCounts = sendCounts + nprocs;
    sendDisps  = (int*) calloc(nprocs * 2, sizeof(int));
    recvDisps  = sendDisps + nprocs;

    /* Only receivers has non-zero data to receive */
    if (is_receiver) {
        j = 0;
        for (i=0; i<nprocs; i++) {
            if (i != rank) { /* skip receiving from self */
                recvCounts[i] = len;
                recvDisps[i] = (len + gap) * j * sizeof(int);
            }
            j++;
            if (verbose && i != rank)
                printf("%2d recv from %2d of %d\n",rank,i,recvCounts[i]);
        }
    }

    /* All ranks send to each receivers */
    j = 0;
    for (i=0; i<nprocs; i++) {
        if (i % ratio) continue; /* i is not a receiver */
        if (i != rank) { /* skip sending to self */
            sendCounts[i] = len;
            sendDisps[i] = (len + gap) * j * sizeof(int);
        }
        j++;
        if (verbose && i != rank)
            printf("%2d send to %2d of %d\n",rank,i,sendCounts[i]);
    }

    sendPtr = sendBuf;
    for (i=0; i<ntimes; i++) {
        if (debug && is_receiver)
            initialize_recv_buf(len, gap, recvBuf);

        err = MPI_Alltoallw(sendPtr, sendCounts, sendDisps, sendTypes,
                            recvBuf, recvCounts, recvDisps, recvTypes,
                            MPI_COMM_WORLD); ERR
        sendPtr += num_recvers * (len + gap);

        if (debug && is_receiver)
            check_recv_buf("alltoallw", len, gap, recvBuf);
    }

err_out:
    free(sendTypes);
    free(sendCounts);
    free(sendDisps);

    timing = MPI_Wtime() - timing;
    MPI_Reduce(&timing, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Time for using MPI_alltoallw    = %.2f sec\n", maxt);
}

/* all-to-many personalized communication by calling MPI_Issend/Irecv() */
void run_async_send_recv(int  ntimes,
                         int  ratio,
                         int  is_receiver,
                         int  len,
                         int  gap,
                         int *sendBuf,
                         int *recvBuf)
{
    int *sendPtr, *recvPtr;
    int i, j, err, nprocs, rank, nreqs, num_recvers;
    MPI_Request *reqs;
    MPI_Status *st;
    double timing, maxt;

    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    num_recvers = nprocs/ ratio;

    /* allocate MPI_Request and MPI_Status arrays */
    reqs = (MPI_Request*) malloc(sizeof(MPI_Request) * (nprocs + num_recvers));
    st = (MPI_Status*) malloc(sizeof(MPI_Status) * (nprocs + num_recvers));

    sendPtr = sendBuf;
    for (i=0; i<ntimes; i++) {
        if (debug && is_receiver)
            initialize_recv_buf(len, gap, recvBuf);

        nreqs = 0;
        recvPtr = recvBuf;

        /* Only receivers post recv requests */
        if (is_receiver) {
            for (j=0; j<nprocs; j++) {
                if (rank != j) { /* skip recv from self */
                    err = MPI_Irecv(recvPtr, len, MPI_INT, j, 0, MPI_COMM_WORLD,
                                    &reqs[nreqs++]);
                    ERR
                }
                recvPtr += len + gap;
            }
        }

        /* all ranks post send requests */
        for (j=0; j<nprocs; j++) {
            if (j % ratio) continue; /* j is not a receiver */
            if (rank != j) { /* skip send to self */
                err = MPI_Issend(sendPtr, len, MPI_INT, j, 0, MPI_COMM_WORLD,
                                 &reqs[nreqs++]);
                ERR
            }
            sendPtr += len + gap;
        }

        err = MPI_Waitall(nreqs, reqs, st); ERR

        if (debug && is_receiver)
            check_recv_buf("issend/irecv", len, gap, recvBuf);
    }

err_out:
    free(st);
    free(reqs);

    timing = MPI_Wtime() - timing;
    MPI_Reduce(&timing, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Time for using MPI_Issend/Irecv = %.2f sec\n", maxt);
}

/*----< usage() >------------------------------------------------------------*/
static void usage (char *argv0) {
    char *help = "Usage: %s [OPTION]\n\
       [-h] Print this help message\n\
       [-v] Verbose mode (default: no)\n\
       [-d] Debug mode to check receive buffer contents (default: no)\n\
       [-n num] number of iterations (default: 1)\n\
       [-r num] every ratio processes is a receiver (default: 1)\n\
       [-l num] receive amount per iteration (default: 8 MB)\n\
       [-g num] gap between 2 consecutive send/recv buffers (default: 4 int)\n";
    fprintf (stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    extern int optind;
    extern char *optarg;
    int i, rank, nprocs;
    int len, gap, block_len, ntimes, ratio, num_recvers, is_receiver;
    int *sendBuf, *recvBuf=NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    verbose = 0;
    debug = 0;
    ntimes = 1;
    ratio = 1;
    block_len = 8 * 1024 * 1024;
    gap = 4;

    /* command-line arguments */
    while ((i = getopt (argc, argv, "hvdn:r:l:g:")) != EOF)
        switch (i) {
            case 'v':
                verbose = 1;
                break;
            case 'd':
                debug = 1;
                break;
            case 'n':
                ntimes = atoi(optarg);
                break;
            case 'r':
                ratio = atoi(optarg);
                break;
            case 'l':
                block_len = atoi(optarg);
                break;
            case 'g':
                gap = atoi(optarg);
                break;
            case 'h':
            default:
                if (rank == 0) usage(argv[0]);
                goto err_out;
        }

    /* set the number of receivers */
    if (ratio <= 0 || ratio > nprocs) ratio = 1;
    num_recvers = nprocs / ratio;

    /* set whether this rank has non-zero data to receive */
    is_receiver = (rank % ratio == 0) ? 1 : 0;

    /* per message size */
    len = block_len / sizeof(int) / nprocs;

    if (verbose && rank == 0)
        printf("nprocs=%d ntimes=%d block_len=%d num_recvers=%d len=%d gap=%d\n",
               nprocs, ntimes, block_len, num_recvers, len, gap);

    if (verbose && is_receiver)
        printf("rank %2d is_receiver\n", rank);

    if (verbose) fflush(stdout);

    if (rank == 0) {
        printf("number of MPI processes         = %d\n", nprocs);
        printf("number of iterations            = %d\n", ntimes);
        printf("numbe of receivers              = %d\n", num_recvers);
        printf("individual message length       = %zd bytes\n",len*sizeof(int));
        printf("send/recv buffer gap            = %d int(s)\n",gap);
        printf("Recv amount per iteration       = %d bytes\n",block_len);
    }

    /* allocate and initialize send buffer */
    sendBuf = (int*) malloc(sizeof(int) * (len + gap) * ntimes * num_recvers);
    initialize_send_buf(ntimes, num_recvers, len, gap, sendBuf);

    if (is_receiver)
        /* receive buffer is reused every iteration */
        recvBuf = (int*) malloc(sizeof(int) * (len + gap) * nprocs);

    /* perform all-to-many communication */
    MPI_Barrier(MPI_COMM_WORLD);
    run_alltoallw(ntimes, ratio, is_receiver, len, gap, sendBuf, recvBuf);

    /* perform all-to-many communication */
    MPI_Barrier(MPI_COMM_WORLD);
    run_async_send_recv(ntimes, ratio, is_receiver, len, gap, sendBuf, recvBuf);

    if (is_receiver)
        free(recvBuf);
    free(sendBuf);

err_out:
    MPI_Finalize();
    return 0;
}

