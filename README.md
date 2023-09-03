## A collection of MPI-IO example programs

### List of example programs
* mpi_file_open.c
  * Calls `MPI_File_open` function to create a new file.
  * Calls `MPI_File_open` function to open an existing file.
* mpi_file_set_view.c
  * Set the visible file region to each process.
* print_mpi_io_hints.c
  * Prints out all default MPI I/O hints.
* mpi_tag_ub.c
  * Obtains the value of attribute `MPI_TAG_UB` attached to communicator
    `MPI_COMM_WORLD`. `MPI_TAG_UB` is the upper bound for tag value.
* fileview_subarray.c
  * Uses MPI_Type_create_subarray() to create an MPI derived data type of a 2D
    subarray to a 2D global array, which is then used in the call to
    MPI_File_set_view().
* ghost_cell.c
  * Writes a 2D local array with ghost cells to a global array in a file. Ghost
    cells are the elements in the local array that are not written to the file.
    This example shows how to define an MPI derived data type to describe a 2D
    subarray with ghost cells used it in I/O.


### To compile
* Modify file `Makefile` if necessary to change the path of MPI C compiler.
* Run command `make [name of example program]`

### Useful links to learn MPI
* [MPI Forum](https://www.mpi-forum.org)
* [MPICH](https://www.mpich.org), an implementation of MPI standard
* [OpenMPI](https://www.open-mpi.org), an implementation of MPI standard
* Book - [MPI: The Complete Reference](http://www.netlib.org/utk/papers/mpi-book/mpi-book.html)

## Questions/Comments:
email: wkliao@eecs.northwestern.edu

Copyright (C) 2019, Northwestern University.

See [COPYRIGHT](COPYRIGHT) notice in top-level directory.

