/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 *  Copyright (C) 2023, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 * This program shows an example of calling MPI_Type_create_subarray() to
 * create an MPI derived data type of a 2D subarray to a 2D global array. It is
 * then used in the call to MPI_File_set_view(), which applies a "mask"
 * (described by the subarray derived data type) to the file, so the unmasked,
 * visible non-contiguous file regions are virtually coalesced into a
 * contiguous one.  With such fileview set, all successive MPI read and write
 * function calls can read and write the visible region as if they are
 * contiguous in file.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#define COL 10
#define ROW 10

#define CHECK_ERR(func) { \
    if (err != MPI_SUCCESS) { \
        int errorStringLen; \
        char errorString[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(err, errorString, &errorStringLen); \
        printf("Error at line %d: calling %s (%s)\n",__LINE__, #func, errorString); \
    } \
}


/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    char *filename;
    int i, err, rank, nprocs, mode, verbose=0, psizes[2]={0,0};
    int gsizes[2], lsizes[2], starts[2], buf[COL*ROW], io_len;
    MPI_File     fh;
    MPI_Datatype file_type;
    MPI_Status   status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    filename = "testfie.out";
    if (argc > 1) filename = argv[1];

    MPI_Barrier(MPI_COMM_WORLD);

    /* Creates a division of processors in a 2D Cartesian grid */
    err = MPI_Dims_create(nprocs, 2, psizes);
    CHECK_ERR("MPI_Dims_create");

    if (verbose)
        printf("rank %2d: psizes=%2d %2d\n", rank, psizes[0],psizes[1]);

    /* set local array sizes */
    lsizes[0]  = COL;
    lsizes[1]  = ROW;

    /* set global array sizes */
    gsizes[0] = COL * psizes[0];
    gsizes[1] = ROW * psizes[1];

    /* set this process's starting offsets to the global array */
    starts[0] = COL * (rank / psizes[1]);
    starts[1] = ROW * (rank % psizes[1]);

    if (verbose)
        printf("rank %2d: gsizes=%2d %2d lsizes=%2d %2d starts=%2d %2d\n", rank,
               gsizes[0],gsizes[1],lsizes[0],lsizes[1],starts[0],starts[1]);

    /* Initialize the contents of this process's write buffer.
     * io_len is the write amount (units of type 'int') by this process.
     */
    io_len = lsizes[0] * lsizes[1];
    for (i=0; i<io_len; i++)
        buf[i] = io_len*rank + i;

    /* Create file type: this process's view to the global 2D array. This
     * process's view is a subarray of size specified by lsizes[] with starting
     * indices specified in starts[]. The index order of gsizes[], lsizes[],
     * and starts[] are in the C order, i.e. row major (e.g. gsize[0] is the
     * most significant dimension.) The unit MPI_INT (etype argument) says the
     * array element is of type 'int'.
     */
    err = MPI_Type_create_subarray(2, gsizes, lsizes, starts,
                                   MPI_ORDER_C, MPI_INT, &file_type);
    CHECK_ERR("MPI_Type_create_subarray");

    /* An MPI derived datatype must be committed before it can be used. */
    err = MPI_Type_commit(&file_type);
    CHECK_ERR("MPI_Type_commit");

    /* Writing to the file using the 2D subarray file type -------------------*/

    /* Set the file open mode to overwrite the file if exists */
    mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;

    /* open to create the file */
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, MPI_INFO_NULL, &fh);
    CHECK_ERR("MPI_File_open");

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_INT, file_type, "native", MPI_INFO_NULL);
    CHECK_ERR("MPI_File_set_view");

    /* MPI collective write
     * buf occupies a contiguous space in memory.
     * This call will writes io_len elements of type 'int' starting from the
     * memory address pointed by 'buf'.
     */
    err = MPI_File_write_all(fh, buf, io_len, MPI_INT, &status);
    CHECK_ERR("MPI_File_write_all");

    /* close the file */
    err = MPI_File_close(&fh);
    CHECK_ERR("MPI_File_close");

    /* Reading from the file using the 2D subarray file type -----------------*/

    /* Set the file open mode to open for read only */
    mode = MPI_MODE_RDONLY;

    /* open the same file to read */
    err = MPI_File_open(MPI_COMM_WORLD, filename, mode, MPI_INFO_NULL, &fh);
    CHECK_ERR("MPI_File_open");

    /* set the file view */
    err = MPI_File_set_view(fh, 0, MPI_INT, file_type, "native", MPI_INFO_NULL);
    CHECK_ERR("MPI_File_set_view");

    /* initialize the contents of read buffer */
    for (i=0; i<io_len; i++)
        buf[i] = -1;

    /* MPI collective read */
    err = MPI_File_read_all(fh, buf, io_len, MPI_INT, &status);
    CHECK_ERR("MPI_File_read_all");

    /* Check the contents for correctness */
    for (i=0; i<io_len; i++) {
        if (buf[i] != io_len*rank + i) {
            printf("rank %d: Error buf[%d]=%d, but expect %d\n",
                   rank, i, buf[i], io_len*rank + i);
            break;
        }
    }

    /* close the file */
    err = MPI_File_close(&fh);
    CHECK_ERR("MPI_File_close");

    /* free the file data type */
    err = MPI_Type_free(&file_type);
    CHECK_ERR("MPI_Type_free");

    MPI_Finalize();
    return 0;
}
