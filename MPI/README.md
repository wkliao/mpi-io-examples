## A collection of MPI example programs

### List of example programs
* **alltomany.c** implements the all-to-many communication with 3 options.
  * Default uses MPI_Isend, MPI_Irecv, and MPI_Wait_all.
  * Command-line option '-a' uses MPI_alltoallv.
  * Command-line option '-s' uses MPI_Issend, MPI_Irecv, and MPI_Wait_all.

### To compile
* Modify file `Makefile` if necessary to change the path of MPI C compiler.
* Run command `make [name of example program]`

## Questions/Comments:
email: wkliao@eecs.northwestern.edu

Copyright (C) 2019, Northwestern University.

See [COPYRIGHT](../COPYRIGHT) notice in top-level directory.

