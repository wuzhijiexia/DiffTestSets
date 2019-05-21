#include <cuda.h>
#include <cudnn.h>
#include "def.h"

void cudnn_convBwdData(convMess cm)
{
    fprintf(stdout, "\033\[32mCUDNN Convolution Backward for Data ......\033\[0m\n");

    int i, j, k;
    int N, C, inH, inW;
    int K, fltH, fltW;
    int outH, outW;

    DataType *h_indiff, *h_flt, *h_outdiff;
    DataType *d_indiff, *d_flt, *d_outdiff;
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
    
    h_flt       = cm->flt_;
    h_outdiff   = cm->outdiff_;
    h_indiff    = cm->indiff_;

    int indim[] = {N, C, inH, inW};
    int indimInv[] = {C*inH*inW, inH*inW, inW, 1};
    int insize = N*C*inH*inW;
    
    int fltdim[] = {K, C, fltH, fltW};
    int fltdimInv[] = {C*fltH*fltW, fltH*fltW, fltW, 1};
    int fltsize = K*C*fltH*fltW;

    int outdim[] = {N, K, outH, outW};
    int outdimInv[] = {K*outH*outW, outH*outW, outW, 1};
    int outsize = N*K*outH*outW;

    int pad[] = {cm->padh_, cm->padw_};
    int stride[] = {cm->strideh_, cm->stridew_};
    int upscaleA[] = {1, 1};

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t inDesc, outDesc;
    cudnnFilterDescriptor_t fltDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
#ifdef ROTATE
    cudnnConvolutionMode_t convMode = CUDNN_CONVOLUTION;
#endif
#ifdef NOROTATE
    cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;
#endif
    
    size_t free_byte;
    size_t total_byte;
    size_t bwdData_sizeInBytes_;
    void *workSpace;
    cudaEvent_t scuda, ecuda;

    cudnnCreate(&handle);
    
    CUDA_CHECK(cudaMalloc((void **)&d_flt, fltsize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_flt, h_flt, fltsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_outdiff, outsize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_outdiff, h_outdiff, outsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_indiff, insize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_indiff, h_indiff, insize*sizeof(DataType), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&scuda));
    CUDA_CHECK(cudaEventCreate(&ecuda));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inDesc)); // input descriptor
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DataType,
                4, indim, indimInv));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outDesc)); // output descriptor
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(outDesc, CUDNN_DataType,
                4, outdim, outdimInv));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&fltDesc)); // filter descriptor
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(fltDesc, CUDNN_DataType,
                CUDNN_TENSOR_NCHW, 4, fltdim));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc)); // convolution descriptor
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, stride,
                upscaleA, convMode, CUDNN_DataType));
    
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    // get convolution forward algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle, fltDesc, outDesc,
                        convDesc, inDesc,
                        //CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
                        //CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                        free_byte, &bwdDataAlgo));

    // get convolution forward workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, fltDesc, outDesc,
                convDesc, inDesc, bwdDataAlgo, &bwdData_sizeInBytes_));
    if(bwdData_sizeInBytes_ > free_byte){ // for bug
        bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;//(cudnnConvolutionbwdDataAlgo_t)(1);
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, fltDesc, outDesc,
                    convDesc, inDesc, bwdDataAlgo, &bwdData_sizeInBytes_));
    }
    CUDA_CHECK(cudaMalloc((void **)&workSpace, bwdData_sizeInBytes_));

    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha, // 第一次运行预热
                fltDesc, d_flt, outDesc, d_outdiff,
                convDesc, bwdDataAlgo, workSpace, bwdData_sizeInBytes_, &beta,
                inDesc, d_indiff));

#ifdef PERF
    gettimeofday(&stime, NULL);
    cudaEventRecord(scuda, 0);
    for(j = 0; j < CYCLE; j++){
        CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha,
                    fltDesc, d_flt, outDesc, d_outdiff,
                    convDesc, bwdDataAlgo, workSpace, bwdData_sizeInBytes_, &beta,
                    inDesc, d_indiff));
    }
    CUDA_CHECK(cudaEventRecord(ecuda, 0));
    CUDA_CHECK(cudaEventSynchronize(scuda));
    CUDA_CHECK(cudaEventSynchronize(ecuda));
    gettimeofday(&etime ,NULL);

    time_ms = TIME_MS(stime, etime)/CYCLE;
    /* gflops = (1.0*((2*fltH*fltW-1)*C+(C-1))*outH/1e3*outW*K/1e3*N)/time_ms; */
    /* fprintf(stdout, "Elapsed-time: %8.2f ms, Perf %7.2f gflops, %6.2f%% peak!\n", */
            /* time_ms, gflops, gflops/1e3/PEAK_PERF*100); */
#endif

    CUDA_CHECK(cudaMemcpy(h_indiff, d_indiff, insize*sizeof(DataType), cudaMemcpyDeviceToHost));

    // clean environment
    GPUFREE(d_flt);
    GPUFREE(d_outdiff);
    GPUFREE(d_indiff);
    GPUFREE(workSpace);

    CUDA_CHECK(cudaEventDestroy(scuda));
    CUDA_CHECK(cudaEventDestroy(ecuda));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(fltDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));
}
