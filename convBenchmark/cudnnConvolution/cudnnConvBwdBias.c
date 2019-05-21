#include <cuda.h>
#include <cudnn.h>
#include "def.h"

void cudnn_convBwdBias(convMess cm)
{
    fprintf(stdout, "\033\[32mCUDNN Convolution Backward for Bias ......\033\[0m\n");

    int i, j, k;
    int N, C, inH, inW;
    int K, fltH, fltW;
    int outH, outW;

    DataType *h_outdiff, *h_bdiff;
    DataType *d_outdiff, *d_bdiff;
    DataType alpha, beta;

    struct timeval stime, etime;
    double gflops, time_ms;

    N       = cm->N_;
    C       = cm->C_;
    inH     = cm->inH_;
    inW     = cm->inW_;
    K       = cm->K_;
    fltH    = cm->fltH_;
    fltW    = cm->fltW_;
    outH    = cm->outH_;
    outW    = cm->outW_;
    
    alpha   = cm->alpha_;
    beta    = cm->beta_;
    
    h_outdiff   = cm->outdiff_;
    h_bdiff     = cm->bdiff_;

    int outdim[] = {N, K, outH, outW};
    int outdimInv[] = {K*outH*outW, outH*outW, outW, 1};
    int outsize = N*K*outH*outW;

    int bdim[] = {1, K, 1, 1};
    int bdimInv[] = {K, 1, 1, 1};
    int bsize = 1*K*1*1;

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t outdiffDesc, bdiffDesc;
    cudaEvent_t scuda, ecuda;

    cudnnCreate(&handle);
    
    CUDA_CHECK(cudaMalloc((void **)&d_outdiff, outsize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_outdiff, h_outdiff, outsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_bdiff, bsize*sizeof(DataType)));

    CUDA_CHECK(cudaEventCreate(&scuda));
    CUDA_CHECK(cudaEventCreate(&ecuda));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outdiffDesc)); // output diff descriptor
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(outdiffDesc, CUDNN_DataType,
                4, outdim, outdimInv));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bdiffDesc)); // bias diff descriptor
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(bdiffDesc, CUDNN_DataType,
                4, bdim, bdimInv));

    fprintf(stdout, "[N K outH outW] = [%d %d %d %d]\n", N, K, outH, outW);
    fprintf(stdout, "[alpha beta] = [%lf %lf]\n", alpha, beta);

    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &alpha, // 第一次运行预热
                outdiffDesc, d_outdiff, &beta, bdiffDesc, d_bdiff));

#ifdef PERF
    gettimeofday(&stime, NULL);
    cudaEventRecord(scuda, 0);
    for(j = 0; j < CYCLE; j++){
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &alpha, // 第一次运行预热
                    outdiffDesc, d_outdiff, &beta, bdiffDesc, d_bdiff));
    }
    CUDA_CHECK(cudaEventRecord(ecuda, 0));
    CUDA_CHECK(cudaEventSynchronize(scuda));
    CUDA_CHECK(cudaEventSynchronize(ecuda));
    gettimeofday(&etime ,NULL);

    time_ms = TIME_MS(stime, etime)/CYCLE;
    gflops = (1.0*N*K*(outH*outW-1))/1e6/time_ms;
    fprintf(stdout, "Elapsed-time: %8.2f ms, Perf %7.2f gflops, %6.2f%% peak!\n",
            time_ms, gflops, gflops/1e3/PEAK_PERF*100);
#endif

    CUDA_CHECK(cudaMemcpy(h_bdiff, d_bdiff, bsize*sizeof(DataType), cudaMemcpyDeviceToHost));

    // clean environment
    GPUFREE(d_outdiff);
    GPUFREE(d_bdiff);

    CUDA_CHECK(cudaEventDestroy(scuda));
    CUDA_CHECK(cudaEventDestroy(ecuda));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outdiffDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bdiffDesc));
}
