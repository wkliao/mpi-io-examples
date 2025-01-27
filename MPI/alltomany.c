/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (C) 2024, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * This program shows 3 ways of performing all-to-many MPI communication.
 *
 * Command-line option '-a' uses MPI_Alltoallv
 * Command-line option '-s' uses MPI_Issend/MPI_Irecv/MPI_Waitall
 * The default is to use MPI_Isend/MPI_Irecv/MPI_Waitall
 *
 * Command-line option '-n' sets the number of iterations
 * Command-line option '-m' sets the maximal number of receivers
 * Command-line option '-r' can be used to set the number of receivers.  For
 *     example, when running 128 MPI processes per compute node, setting '-r
 *     128' will pick one receiver per compute node.
 * Command-line option '-l' can be used to set the message size.
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
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/*----< usage() >------------------------------------------------------------*/
static void usage (char *argv0) {
    char *help = "Usage: %s [OPTION]\n\
       [-h] Print this help message\n\
       [-v] Verbose mode (default: no)\n\
       [-a] use MPI_alltoallv (default: MPI_Isend/Irecv)\n\
       [-s] use MPI_Issend (default: MPI_Isend/Irecv)\n\
       [-n num] number of iterations (default: 1)\n\
       [-m num] number of receivers (default: total number of processes / ratio)\n\
       [-r ratio] ratio of number of receivers to all processes (default: 1)\n\
       [-l len] individual message size (default: 48)\n";
    fprintf (stderr, help, argv0);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    extern int optind;
    extern char *optarg;
    int i, j, rank, nprocs, err, nerrs=0, verbose, len, ntimes, ratio;
    int use_alltoall, use_issend, is_recver, num_recvers, *recver_rank;
    int max_num_recvers;
    char *buf;
    double timing, maxt;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    verbose = 0;
    use_alltoall = 0;
    use_issend = 0;
    len = 48;
    ntimes = 1;
    ratio = 1;
    max_num_recvers = nprocs;

    /* command-line arguments */
    while ((i = getopt (argc, argv, "hvasl:n:r:m:")) != EOF)
        switch (i) {
            case 'v':
                verbose = 1;
                break;
            case 's':
                use_issend = 1;
                break;
            case 'a':
                use_alltoall = 1;
                break;
            case 'l':
                len = atoi(optarg);
                break;
            case 'n':
                ntimes = atoi(optarg);
                break;
            case 'r':
                ratio = atoi(optarg);
                break;
            case 'm':
                max_num_recvers = atoi(optarg);
                break;
            case 'h':
            default:
                if (rank == 0) usage(argv[0]);
                goto err_out;
        }

    if (use_alltoall == 1 && use_issend == 1) {
        if (rank == 0)
            printf("Error: command-line options '-a' and '-s' cannot be both set\n");
        goto err_out;
    }

    is_recver = 0;
    num_recvers = nprocs / ratio;
    if (num_recvers > max_num_recvers) num_recvers = max_num_recvers;

    if (rank == 0) {
        if (use_alltoall)
            printf("---- Using MPI_Alltoallv\n");
        else if (use_issend)
            printf("---- Using MPI_Issend/Irecv\n");
        else
            printf("---- Using MPI_Isend/Irecv\n");
#ifdef MPICH_VERSION
        printf("---- This MPI is based on MPICH version %s\n",MPICH_VERSION);
#endif
#ifdef CRAY_MPICH_VERSION
        printf("---- This MPI is based on Cray MPICH version %s\n",
               TOSTRING(CRAY_MPICH_VERSION));
#endif
        printf("nprocs      = %d\n", nprocs);
        printf("len         = %d\n", len);
        printf("ntimes      = %d\n", ntimes);
        printf("ratio       = %d\n", ratio);
        printf("num_recvers = %d\n", num_recvers);
    }

    recver_rank = (int*) malloc(sizeof(int) * num_recvers);
    if (verbose && rank == 0) printf("recver_rank: ");
    for (i=0; i<num_recvers; i++) {
        recver_rank[i] = i * ratio;
        if (rank == recver_rank[i]) is_recver = 1;
        if (verbose && rank == 0) printf(" %d", recver_rank[i]);
    }
    if (verbose && rank == 0) printf("\n");
    if (verbose) fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();

    buf = (char*) malloc((nprocs + num_recvers) * len);

    if (use_alltoall == 0) {
        MPI_Request *reqs;
        MPI_Status *st;

        reqs = (MPI_Request*) calloc(nprocs + num_recvers, sizeof(MPI_Request));
        st = (MPI_Status*)malloc(sizeof(MPI_Status) * (nprocs + num_recvers));

        for (i=0; i<ntimes; i++) {
            int nreqs=0;
            char *ptr = buf;

            /* post recv requests */
            if (is_recver) {
                for (j=0; j<nprocs; j++) {
                    err = MPI_Irecv(ptr, len, MPI_BYTE, j, 0, MPI_COMM_WORLD,
                                    &reqs[nreqs++]);
                    ERR
                    ptr += len;
                }
            }

            /* post send requests */
            for (j=0; j<num_recvers; j++) {
                if (use_issend)
                    err = MPI_Issend(ptr, len, MPI_BYTE, recver_rank[j], 0,
                                     MPI_COMM_WORLD, &reqs[nreqs++]);
                else
                    err = MPI_Isend(ptr, len, MPI_BYTE, recver_rank[j], 0,
                                    MPI_COMM_WORLD, &reqs[nreqs++]);
                ERR
                ptr += len;
            }

            err = MPI_Waitall(nreqs, reqs, st);
            ERR
        }
        free(st);
        free(reqs);
    }
    else {
        int *sendCounts, *recvCounts, *sdispls, *rdispls;
        char *r_buf, *s_buf;

        sendCounts = (int*) calloc(nprocs * 2, sizeof(int));
        recvCounts = sendCounts + nprocs;
        sdispls = (int*) calloc(nprocs * 2, sizeof(int));
        rdispls = sdispls + nprocs;

        if (is_recver) {
            for (i=0; i<nprocs; i++) {
                recvCounts[i] = len;
                rdispls[i] = len * i;
            }
        }
        r_buf = buf;

        for (i=0; i<num_recvers; i++) {
            sendCounts[recver_rank[i]] = len;
            sdispls[recver_rank[i]] = len * i;
        }
        s_buf = r_buf + nprocs * len;

        for (i=0; i<ntimes; i++) {
            err = MPI_Alltoallv(s_buf, sendCounts, sdispls, MPI_BYTE,
                                r_buf, recvCounts, rdispls, MPI_BYTE,
                                MPI_COMM_WORLD);
            ERR
        }
        free(sendCounts);
        free(sdispls);
    }
    free(buf);
    free(recver_rank);

    timing = MPI_Wtime() - timing;
    MPI_Reduce(&timing, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double wb = (double)len * nprocs * ntimes * num_recvers;
        wb /= 1048576.0; /* in MB */
        printf("Total message amount: %.2f MiB\n", wb);
        printf("Max time:             %.2f sec\n", maxt);
        printf("Comm bandwidth:       %.2f MiB/sec\n", wb / maxt);
    }

err_out:
    MPI_Finalize();
    return 0;
}


