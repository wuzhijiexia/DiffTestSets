#!/bin/bashr

# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

# P100: arch=compute_60,code=sm_60
# k20: arch=compute_35,code=sm_35

nvcc -DDATA_FLOAT -DVERITY_RESULT -gencode arch=compute_60,code=sm_60 -Xcompiler -fopenmp -o test_cudnnConv test_cudnnConv.c  def.c cudnn_conv.c -lcuda -lcudnn
echo "Begin to run ......"

./test_cudnnConv
