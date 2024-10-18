CC       = mpicc
CFLAGS   = -O0 -g

SUBDIRS  = MPI

check_PROGRAMS = mpi_file_set_view \
                 mpi_file_open \
                 print_mpi_io_hints \
                 mpi_tag_ub \
                 fileview_subarray \
                 ghost_cell \
                 indexed_fsize \
                 hindexed_fsize \
                 nvars \
                 struct_fsize

all: $(check_PROGRAMS)
	@if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && make) ; \
	    done; \
	fi

TESTS_ENVIRONMENT = export check_PROGRAMS="$(check_PROGRAMS)";

check: all
	@$(TESTS_ENVIRONMENT) \
	./test.sh 4 || exit 1
	@if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && make check) ; \
	    done; \
	fi

clean:
	rm -f core.* *.o testfile.out $(check_PROGRAMS)
	@if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && make clean) ; \
	    done; \
	fi


.PHONY: clean

