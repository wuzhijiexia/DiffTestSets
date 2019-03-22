#!/bin/bashr

# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

MKL_INCLUDE_PATH=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/mkl/include/
MKL_LIB_PATH_1=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/mkl/lib/intel64_lin/
MKL_LIB_PATH_2=${HOME}/person/intel-2019/compilers_and_libraries_2019.2.187/linux/compiler/lib/intel64_lin/
icc -DDATA_FLOAT -DVERITY_RESULT -O2 -qopenmp -xhost -restrict \
    -I${MKL_INCLUDE_PATH} -L${MKL_LIB_PATH_1} -L${MKL_LIB_PATH_2} \
    -o test_mklConv test_mklConv.c mkl_conv.c def.c \
    -Wl,-rpath=${MKL_LIB_PATH_1} -Wl,-rpath=${MKL_LIB_PATH_2} \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

echo "Begin to run ......"

./test_mklConv
