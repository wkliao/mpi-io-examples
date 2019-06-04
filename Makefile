MPICC		= mpicc

CPPFLAGS	=
CFLAGS          = -O0 -g
LDFLAGS		=
LIBS		=

.c.o:
	$(MPICC) $(CFLAGS) $(INCLUDES) -c $<

all:

mpi_file_open: mpi_file_open.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

mpi_create_delete_loop: mpi_create_delete_loop.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f core.* *.o mpi_file_open mpi_create_delete_loop

.PHONY: clean
