#!/bin/bashr

# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

MKLDNN_INCLUDE_PATH=${HOME}/person/mkl-dnn/include/
MKLDNN_LIB_PATH=${HOME}/person/mkl-dnn/lib64/
icc -DDATA_FLOAT -DVERITY_RESULT -O0 -qopenmp -xhost -restrict \
    -I${MKLDNN_INCLUDE_PATH} -L${MKLDNN_LIB_PATH} \
    -o test_mkldnnConv test_mkldnnConv.c mkldnn_conv.c def.c \
    -Wl,-rpath=${MKLDNN_LIB_PATH} \
    -lmkldnn #-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

echo "Begin to run ......"

./test_mkldnnConv
