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

print_mpi_io_hints: print_mpi_io_hints.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

mpi_tag_ub: mpi_tag_ub.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

mpi_create_delete_loop: mpi_create_delete_loop.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f core.* *.o testfile \
	mpi_file_open mpi_create_delete_loop mpi_tag_ub print_mpi_io_hints

.PHONY: clean
