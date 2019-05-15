#ifndef _DEF_H_
#define _DEF_H_

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#include <cuda.h>
#include <cudnn.h>

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

#define K20_S_PEAK_PERF   3.52
#define K20_D_PEAK_PERF   1.17
#define P100_S_PEAK_PERF  10.6
#define P100_D_PEAK_PERF  5.3
#define PEAK_PERF    P100_S_PEAK_PERF

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
    PER_ACTIVATION      = 0, // 1,C,H,W
    SPATIAL             = 1  // 1,C,1,1
}batchNormMode_t;

struct batchNormMessStruct{
    int N_, C_, H_, W_;
    DataType alpha_, beta_;
    DataType *in_, *out_;
    batchNormMode_t mode_;
    DataType *bnScale_, *bnBias_;
    DataType *runMean_, *runVariance_;
    DataType *saveMean_, *saveInvVariance_;
    DataType epsilon_, expAvgFactor_;

    DataType *indiff_, *outdiff_;
    DataType *bndiffScale_, *bndiffBias_;
    DataType alphaDataDiff_, betaDataDiff_;
    DataType alphaParamDiff_, betaParamDiff_;
};
typedef struct batchNormMessStruct *batchNormMess;

void fileInit4dData(FILE *pfile, DataType *data, int *dim);
void randomInit4dData(DataType *data, int size);
void zeroInit4dData(DataType *data, int size);
void compare4dData(DataType *out, DataType *out_verify, int size);

// batchnorm forward for inference
void cudnn_batchNormFwdInference(batchNormMess bnMess);

// batchnorm forward for training
void cudnn_batchNormFwdTrain(batchNormMess bnMess);

// batchnorm backward for training
void cudnn_batchNormBwdTrain(batchNormMess bnMess);

#endif
