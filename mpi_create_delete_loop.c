#include <stdio.h>
#include <stdlib.h>
#include <errno.h>  /* errno */
#include <string.h> /* memset(), strerror() */
#include <unistd.h> /* unlink() */
#include <mpi.h>

#define MAX_TRIES 2000

#define CHECK_ERR \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: %s\n",__LINE__, errorString); \
        break; /* loop i */ \
    }

int main(int argc, char** argv) {
    char *filename, buf[512];
    int i, rank, nprocs, err;
    MPI_Status status;
    MPI_File fh;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    filename = "testfile";
    memset(buf, 0, 512);

    for (i=0; i<MAX_TRIES; i++) {

        /* mimic NC_CLOBBER */
        if (rank == 0) {
            err = unlink(filename);
            if (err < 0 && errno != ENOENT) { /* ignore ENOENT: file not exist */
                printf("Error: errno=%d (%s)\n",errno,strerror(errno));
                break; /* loop i */
            }
        }

        /* all processes must wait here until file deletion is completed */
        err = MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
        CHECK_ERR

        /* all processes open the file in parallel */
        err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
        CHECK_ERR

        if (rank == 0) { /* mimic PnetCDF rank 0 writes to file header */
            err = MPI_File_write(fh, buf, 512, MPI_BYTE, &status);
            CHECK_ERR
        }

        err = MPI_File_close(&fh);
        CHECK_ERR
    }

    MPI_Finalize();
    return 0;
}
