#include <errno.h> /* errno */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* memset(), strerror() */
#include <unistd.h> /* unlink() */

#define MAX_TRIES 100000

#include "mpi_utils.h"

int main(int argc, char **argv) {

    char *filename, buf[512];
    int i, rank, nprocs, err, sys_err;
    MPI_Status status;
    MPI_File fh;

    MPI_Init(&argc, &argv);
    MPI_CHECK_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK_ERR(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    filename = "testfile";
    memset(buf, 0, 512);

    for (i = 0; i < MAX_TRIES; i++) {

        /* mimic NC_CLOBBER */
        if (rank == 0) {
            err = unlink(filename);
            if (err < 0 && errno != ENOENT) { /* ignore ENOENT: file not exist */
                printf("Error: unlink() errno=%d (%s)\n", errno, strerror(errno));
                sys_err = -1;
            }
            else
                sys_err = 0;
        }

        /* all processes must wait here until file deletion is completed */
        err = MPI_CHECK_ERR(MPI_Bcast(&sys_err, 1, MPI_INT, 0, MPI_COMM_WORLD));

        if (err != MPI_SUCCESS)
            break; /* loop i */

        if (sys_err != 0)
            break; /* loop i */

        /* all processes open the file in parallel */
        err = MPI_CHECK_ERR(MPI_File_open(MPI_COMM_WORLD, filename,
                                          MPI_MODE_CREATE | MPI_MODE_RDWR,
                                          MPI_INFO_NULL, &fh));

        if (err != MPI_SUCCESS)
            break; /* loop i */

        if (rank == 0) { /* mimic PnetCDF rank 0 writes to file header */
            err = MPI_CHECK_ERR(MPI_File_write(fh, buf, 512, MPI_BYTE, &status));

            if (err != MPI_SUCCESS)
                break; /* loop i */
        }

        err = MPI_CHECK_ERR(MPI_File_close(&fh));
        if (err != MPI_SUCCESS)
            break; /* loop i */
    }

    MPI_Finalize();
    return 0;
}
