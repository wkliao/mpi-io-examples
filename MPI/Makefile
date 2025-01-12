CC       = mpicc
CFLAGS   = -O0 -g

check_PROGRAMS = alltomany alltoallw

all: $(check_PROGRAMS)

TESTS_ENVIRONMENT = export check_PROGRAMS="$(check_PROGRAMS)";

check: all
	@$(TESTS_ENVIRONMENT) \
	./test.sh 4 || exit 1

clean:
	rm -f core.* *.o $(check_PROGRAMS)

.PHONY: clean

