#ifndef _DEF_H_
#define _DEF_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#define CYCLE       10
#define false       0
#define true        1

#define TIME_MS(a,b) (1.0*(b.tv_sec-a.tv_sec)*1e3+1.0*(b.tv_usec-a.tv_usec)*1e-3)

#ifdef DATA_FLOAT
#define DataType    float
#define CUDNN_DataType	CUDNN_DATA_FLOAT
#endif

#ifdef DATA_DOUBLE
#define DataType    double
#define CUDNN_DataType CUDNN_DATA_DOUBLE
#endif

#define K20_S_PEAK_PERF     3.52
#define K20_D_PEAK_PERF     1.17
#define P100_S_PEAK_PERF    10.6
#define P100_D_PEAK_PERF    5.3
#define TITANV_S_PEAK_PERF  14.9
#define TITANV_D_PEAK_PERF  7.45
#define PEAK_PERF    TITANV_D_PEAK_PERF

#define CODE_MESSAGE(tips) \
do { \
    fprintf(stdout, "[%s : %s : %d] >>> \033[31m%s\033[0m\n", \
            __FILE__, __FUNCTION__, __LINE__, tips); \
}while(false)

#define CPUFREE(pdata) \
do{ \
    if(pdata != NULL) free(pdata); \
}while(false)

#define GPUFREE(pdata) \
    do{ \
        if(pdata != NULL) CUDA_CHECK(cudaFree(pdata)); \
    }while(false)

#define CUDNN_CHECK(flag) \
    do{ \
        cudnnStatus_t value = flag; \
        if(value != CUDNN_STATUS_SUCCESS) { \
            fprintf(stdout, "return value: %d\n", value); \
            CODE_MESSAGE("cudnn error!"); \
            exit(1); \
        } \
    }while(false)

#define CUDA_CHECK(flag) \
    do{ \
        cudaError_t value = flag; \
        if(value != cudaSuccess) { \
            CODE_MESSAGE("cuda error!"); \
            exit(1); \
        } \
    }while(false)

typedef enum {
    CONVOLUTION         = 0,
    CROSS_CORRELATION   = 1
} convMode_t;

struct convMessStruct {
    DataType *in_, *flt_, *out_;
    DataType alpha_, beta_;

    int N_, C_, inH_, inW_;
    int K_, fltH_, fltW_;
    int outH_, outW_;
    int padh_, padw_;
    int strideh_, stridew_;

    DataType *indiff_, *fltdiff_, *outdiff_;
    DataType *bdiff_;
};
typedef struct convMessStruct *convMess;

void fileInit4dData(FILE *pfile, DataType *data, int *dim);
void randomInit4dData(DataType *data, int *dim);
void zeroInit4dData(DataType *data, int *dim);
void compare4dData(DataType *out, DataType *out_verify, int *dim);

// convolution forward
void cudnn_convFwd(convMess cMess);
void simple_convFwd(convMess cMess);

// conv backward for data
void cudnn_convBwdData(convMess cMess);
void simple_convBwdData(convMess cMess);

// conv backward for filter
void cudnn_convBwdFilter(convMess cMess);
void simple_convBwdFilter(convMess cMess);

// conv backward for bias
void cudnn_convBwdBias(convMess cMess);
void simple_convBwdBias(convMess cMess);

#endif
