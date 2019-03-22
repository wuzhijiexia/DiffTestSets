#ifndef _MKL_CONV_H_
#define _MKL_CONV_H_

#include <mkl.h>
#include "def.h"

#define CYCLE	    10

#define XEON_8153_PEAK_PERF          3.276
#define XEON_E5_2695V4_PEAK_PERF     2.4192
#define PEAK_PERF   XEON_E5_2695V4_PEAK_PERF

#define MKL_CHECK(flag) \
    do{ \
        dnnError_t value = flag; \
        if(value != E_SUCCESS) { \
            fprintf(stdout, "[ERROR ## mkldnn error] "); \
            CODE_MESSAGE(); \
        } \
    }while(false)

#define MKLFREE(pdata) \
    do{ \
        if(pdata != NULL) mkl_free(pdata); \
    }while(false)


void mkldnnConvolutionTest(convMess*);

#endif
