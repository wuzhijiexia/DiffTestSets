#!/bin/bashr

# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

MKL_INCLUDE_PATH=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/mkl/include/
MKL_LIB_PATH_1=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64_lin/
MKL_LIB_PATH_2=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/compiler/lib/intel64_lin/
icc -DDATA_FLOAT -DVERITY_RESULT -O2 -qopenmp -xhost -restrict \
    -I${MKL_INCLUDE_PATH} -L${MKL_LIB_PATH_1} -L${MKL_LIB_PATH_2} \
    -o dgemm_perf dgemm_perf.c \
    -Wl,-rpath=${MKL_LIB_PATH_1} -Wl,-rpath=${MKL_LIB_PATH_2} \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

icc -DDATA_FLOAT -DVERITY_RESULT -O2 -qopenmp -xhost -restrict \
    -I${MKL_INCLUDE_PATH} -L${MKL_LIB_PATH_1} -L${MKL_LIB_PATH_2} \
    -o sgemm_perf sgemm_perf.c \
    -Wl,-rpath=${MKL_LIB_PATH_1} -Wl,-rpath=${MKL_LIB_PATH_2} \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

echo "Begin to run gemm \033[31m[float] \033[36m......"
./sgemm_perf
echo "Begin to run gemm \033[31m[double] \033[36]m......"
./dgemm_perf
