#ifndef MPI_UTILS_H_
#define MPI_UTILS_H_

#include <mpi.h>
#include <stdio.h>

int mpi_check_error(int err,
                    char const *const func,
                    const char *const file,
                    int const line)
{

    if (err != MPI_SUCCESS)
    {
        int errorStringLen;
        char errorString[MPI_MAX_ERROR_STRING];
        MPI_Error_string(err, errorString, &errorStringLen);
        fprintf(stderr, "Error at %s:%d: calling %s ==> %s\n",file, line, func, errorString);
    }

    return err;
} // check_mpi_error

#define MPI_CHECK_ERR(value) mpi_check_error((value), #value, __FILE__, __LINE__)

#endif // MPI_UTILS_H_
