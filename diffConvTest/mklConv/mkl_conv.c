#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "mkl_conv.h"

void mklConvolutionTest(convMess *pcm)
{
    int i, j, k;

    // Convolution Configure
    int N, C, K, inH, inW;
    int R, S;
    int padh, padw, strideh, stridew;
    int outH, outW;
    int insize, fltsize, outsize;

    struct timeval stime, etime;
    double time;
    double gflops;

    N = pcm->N_;
    C = pcm->C_;
    K = pcm->K_;
    inH = pcm->inH_;
    inW = pcm->inW_;
    outH = pcm->outH_;
    outW = pcm->outW_;
    R = pcm->R_;
    S = pcm->S_;
    padh = pcm->padh_;
    padw = pcm->padw_;
    strideh = pcm->strideh_;
    stridew = pcm->stridew_;

    insize = N*C*inH*inW;
    fltsize = K*C*R*S;
    outsize = N*K*outH*outW;

    /* Malloc the space for data. */
    DataType* in, *flt, *out; 
    in = (DataType *)mkl_malloc(insize*sizeof(DataType), 64);
    flt = (DataType *)mkl_malloc(fltsize*sizeof(DataType), 64);
    out = (DataType *)mkl_malloc(outsize*sizeof(DataType), 64);

#pragma omp parallel for private(k)
    for(k = 0; k < insize; k++)
        in[k] = 0.1*(rand()%4);
#pragma omp parallel for private(k)
    for(k = 0; k < fltsize; k++)
        flt[k] = 0.1*(rand()%4);

    size_t inSize[DataDim] = {inW, inH, C, N};
    size_t inStride[DataDim] = {1, inW, inW*inH, inW*inH*C};

    size_t fltSize[DataDim] = {S, R, C, K};
    size_t fltStride[DataDim] = {1, S, R*S, R*S*C};

    size_t outSize[DataDim] = {outW, outH, K, N};
    size_t outStride[DataDim] = {1, outW, outH*outW, outH*outW*K};

    size_t convStride[DataDim-2] = {stridew, strideh};
    int inOffset[DataDim-2] = {-padw, -padh};

    dnnLayout_t ly_user_in = NULL,
                ly_user_flt = NULL,
                ly_user_out = NULL;

    dnnPrimitive_t conv = NULL;
    dnnLayout_t ly_conv_in = NULL,
                ly_conv_flt = NULL,
                ly_conv_out = NULL;

    dnnPrimitive_t cv_user_to_conv_in = NULL,
                   cv_user_to_conv_flt = NULL,
                   cv_conv_to_user_out = NULL;

    DataType *resConv[dnnResourceNumber] = {0};

    dnnPrimitiveAttributes_t attributes = NULL;

    dnnLayoutCreate_F32(&ly_user_in, DataDim, inSize, inStride);
    dnnLayoutCreate_F32(&ly_user_flt, DataDim, fltSize, fltStride);
    dnnLayoutCreate_F32(&ly_user_out, DataDim, outSize, outStride);

    /* Initialize attributes */
    dnnPrimitiveAttributesCreate_F32(&attributes);

    /* convolution section */
    dnnConvolutionCreateForward_F32(&conv, attributes,
            dnnAlgorithmConvolutionDirect, DataDim, inSize,
            outSize, fltSize, convStride, inOffset,
            dnnBorderZeros);

    /* convolution description */
    dnnLayoutCreateFromPrimitive_F32(&ly_conv_in, conv, dnnResourceSrc);
    dnnLayoutCreateFromPrimitive_F32(&ly_conv_flt, conv, dnnResourceFilter);
    dnnLayoutCreateFromPrimitive_F32(&ly_conv_out, conv, dnnResourceDst);

    /* conversion create */
    dnnConversionCreate_F32(&cv_user_to_conv_in, ly_user_in, ly_conv_in);
    dnnAllocateBuffer_F32((void**)&resConv[dnnResourceSrc], ly_conv_in);
    dnnConversionCreate_F32(&cv_user_to_conv_flt, ly_user_flt, ly_conv_flt);
    dnnAllocateBuffer_F32((void**)&resConv[dnnResourceFilter], ly_conv_flt);
    dnnAllocateBuffer_F32((void**)&resConv[dnnResourceDst], ly_conv_out);
    dnnConversionCreate_F32(&cv_conv_to_user_out, ly_conv_out, ly_user_out);

    /* conversion execute */
    dnnConversionExecute_F32(cv_user_to_conv_in, in, resConv[dnnResourceSrc]);
    dnnConversionExecute_F32(cv_user_to_conv_flt, flt, resConv[dnnResourceFilter]);

    /* Preheat */
    dnnExecute_F32(conv, (void **)resConv);

    gettimeofday(&stime, NULL);
    for(i = 0; i < CYCLE; i++)
    {
        //dnnConversionExecute_F32(cv_user_to_conv_in, in, resConv[dnnResourceSrc]);
        //dnnConversionExecute_F32(cv_user_to_conv_flt, flt, resConv[dnnResourceFilter]);
        dnnExecute_F32(conv, (void **)resConv);
        //dnnConversionExecute_F32(cv_conv_to_user_out, resConv[dnnResourceDst], out);
    }
    gettimeofday(&etime, NULL);
    
    time = (1.0*(etime.tv_sec - stime.tv_sec)*1000 + 1.0*(etime.tv_usec - stime.tv_usec)/1000)/CYCLE;
    gflops = (1.0*((2*R*S-1)*C+(C-1))*outH/1e3*outW*K/1e3*N)/time;
    
    /* fprintf(stdout, "[N C K (inH inW) (R S) (outH outW) (padh padw) (strideh stridew)] = "); */
    fprintf(stdout, "[%4d %4d %4d (%4d %4d) (%3d %3d) (%4d %4d) (%3d %3d) (%3d %3d)] ",
            N, C, K, inH, inW, R, S, outH, outW, padh, padw, strideh, stridew);
    fprintf(stdout, "|| elapsed-time %8.2f ms || Perf %7.2f gflops, %6.2f peak!\n",
            time, gflops, gflops/1e3/PEAK_PERF*100);
    
    /* gain finally output data */
    dnnConversionExecute_F32(cv_conv_to_user_out, resConv[dnnResourceDst], out);
    
#ifdef VERITY_RESULT
    verityConvolutionResult(pcm, in, flt, out);
#endif

    // Clean environment
    dnnDelete_F32(conv);

    dnnLayoutDelete_F32(ly_conv_in);
    dnnLayoutDelete_F32(ly_conv_flt);
    dnnLayoutDelete_F32(ly_conv_out);

    dnnLayoutDelete_F32(ly_user_in);
    dnnLayoutDelete_F32(ly_user_flt);
    dnnLayoutDelete_F32(ly_user_out);

    dnnReleaseBuffer_F32(resConv[dnnResourceSrc]);
    dnnReleaseBuffer_F32(resConv[dnnResourceFilter]);
    dnnReleaseBuffer_F32(resConv[dnnResourceDst]);

    dnnPrimitiveAttributesDestroy_F32(attributes);

    /* Free data memory space. */
    MKLFREE(in); 
    MKLFREE(flt); 
    MKLFREE(out); 
}
