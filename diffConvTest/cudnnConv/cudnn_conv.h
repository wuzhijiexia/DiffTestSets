#ifndef _CUDNN_CONV_H_
#define _CUDNN_CONV_H_

#include <cuda.h>
#include <cudnn.h>
#include "def.h"

#define CYCLE       10

#define K20_S_PEAK_PERF   3.52
#define K20_D_PEAK_PERF   1.17
#define P100_S_PEAK_PERF  10.6
#define P100_D_PEAK_PERF  5.3
#define PEAK_PERF    P100_S_PEAK_PERF

#ifdef DATA_FLOAT
#define CUDNN_DataType	CUDNN_DATA_FLOAT
#endif

#ifdef DATA_DOUBLE
#define CUDNN_DataType CUDNN_DATA_DOUBLE
#endif

#define CUDNN_CHECK(flag) \
    do{ \
        cudnnStatus_t value = flag; \
        if(value != CUDNN_STATUS_SUCCESS) { \
            fprintf(stdout, "[ERROR ## cudnn error] "); \
            CODE_MESSAGE(); \
            exit(1); \
        } \
    }while(false)

#define CUDA_CHECK(flag) \
    do{ \
        cudaError_t value = flag; \
        if(value != cudaSuccess) { \
            fprintf(stdout, "[ERROR ## cuda error] "); \
            CODE_MESSAGE(); \
            exit(1); \
        } \
    }while(false)

#define GPUFREE(pdata) \
    do{ \
        if(pdata != NULL) CUDA_CHECK(cudaFree(pdata)); \
    }while(false)

void cudnnConvolutionTest(convMess*);
#endif
