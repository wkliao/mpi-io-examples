CC       = mpicc
CFLAGS   = -O0 -g

check_PROGRAMS = mpi_file_set_view \
                 mpi_file_open \
                 print_mpi_io_hints \
                 mpi_tag_ub \
                 fileview_subarray \
                 ghost_cell

all: $(check_PROGRAMS)

TESTS_ENVIRONMENT = export check_PROGRAMS="$(check_PROGRAMS)";

check: all
	@$(TESTS_ENVIRONMENT) \
	./test.sh 4 || exit 1

clean:
	rm -f core.* *.o testfile.out $(check_PROGRAMS)

.PHONY: clean

