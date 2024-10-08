/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *  Copyright (C) 2019, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 * This program shows an example of calling MPI_File_set_view(), which sets
 * a visible file region to the calling MPI process. With such fileview set,
 * all successive MPI read and write function calls will read and write only
 * the visible region. In this example program, the visible region to process
 * rank i starts from file offset (rank * 10 * sizeof(int)).
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHECK_ERR(func) { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: calling %s (%s)\n",__LINE__, #func, errorString); \
    } \
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    char *filename;
    int i, err, cmode, rank, buf[10];
    MPI_Offset offset;
    MPI_File fh;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CHECK_ERR(MPI_Comm_rank);

    filename = "testfile.out";
    if (argc > 1) filename = argv[1];

    /* open a file (create if the file does not exist) */
    cmode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    err = MPI_File_open(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &fh);
    CHECK_ERR(MPI_File_open);

    /* initialize write buffer contents */
    for (i=0; i<10; i++) buf[i] = 100 * rank + i;

    /* set file offset for this calling process */
    offset = (MPI_Offset)rank * 10 * sizeof(int);

    /* MPI_File_set_view() sets the file visible region to this process starts
     * at offset. Setting etype argument to MPI_INT means this file will be
     * accessed in the units of integer size. Setting filetype argument to
     * MPI_INT means a contiguous 4-byte region (assuming integer size if 4
     * bytes) is recursively applied to the file to form the visible region to
     * the calling process, starting from its "offset" set in the offset
     * argument. In this example, the "file view" of a process is the entire
     * file starting from its offset.
     */
    err = MPI_File_set_view(fh, offset, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
    CHECK_ERR(MPI_File_set_view);

    /* Each process writes 3 integers to the file region visible to it.
     * Note the file pointer will advance 3x4 bytes after this call.
     */
    err = MPI_File_write_all(fh, &buf[0], 3, MPI_INT, &status);
    CHECK_ERR(MPI_File_write_all);

    /* Each process continues to write next 7 integers to the file region
     * visible to it, starting from the file pointer updated from the previous
     * write call.
     */
    err = MPI_File_write_all(fh, &buf[3], 7, MPI_INT, &status);
    CHECK_ERR(MPI_File_write_all);

    /* close the file collectively */
    err = MPI_File_close(&fh);
    CHECK_ERR(MPI_File_close);

    MPI_Finalize();
    return 0;
}


