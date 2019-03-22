#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "cudnn_conv.h"

void cudnnConvolutionTest(convMess* pcm)
{
    int i, j, k;

    // Convolution Configure
    int N, C, K, inH, inW;
    int R, S;
    int padh, padw, strideh, stridew;
    int outH, outW;

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

    // cuDNN Convolution
    DataType *h_in, *h_flt, *h_out;
    DataType *d_in, *d_flt, *d_out;

    struct timeval stime, etime;
    double time;
    double gflops;

    int pad[] = {padh, padw};
    int stride[] = {strideh, stridew};
    int upscaleA[2] = {1, 1};
    DataType alpha = 1.0f;
    DataType beta = 0.0f;

    int indim[] = {N, C, inH, inW};
    int indim_stride[] = {C*inH*inW, inH*inW, inW, 1};
    int insize = N*C*inH*inW;

    int fltdim[] = {K, C, R, S};
    int fltdim_stride[] = {C*R*S, R*S, S, 1};
    int fltsize = K*C*R*S;

    int outdim[] = {N, K, outH, outW};
    int outdim_stride[] = {K*outH*outW, outH*outW, outW, 1};
    int outsize = N*K*outH*outW;
    
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    cudnnConvolutionDescriptor_t conv_desc_;

    cudnnConvolutionFwdAlgo_t fwd_algo_;
    size_t free_byte;
    size_t total_byte;
    size_t fwd_sizeInBytes_;
    void *workSpace_;

    h_in = (DataType *)malloc(insize*sizeof(DataType));
    h_flt = (DataType *)malloc(fltsize*sizeof(DataType));
    h_out = (DataType *)malloc(outsize*sizeof(DataType));

    for(k = 0; k < insize; k++)
        h_in[k] = 0.1*(rand()%4);
    for(k = 0; k < fltsize; k++)
        h_flt[k] = 0.1*(rand()%4);

    CUDA_CHECK(cudaMalloc((void **)&d_in, insize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, insize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_flt, fltsize*sizeof(DataType)));
    CUDA_CHECK(cudaMemcpy(d_flt, h_flt, fltsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_out, outsize*sizeof(DataType)));

    cudaEvent_t scuda, ecuda;
    CUDA_CHECK(cudaEventCreate(&scuda));
    CUDA_CHECK(cudaEventCreate(&ecuda));

    // input data descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bottom_desc_));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(bottom_desc_, CUDNN_DataType,
                        4, indim, indim_stride));

    // filter data descriptor
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(filter_desc_, CUDNN_DataType,
                        CUDNN_TENSOR_NCHW, 4, fltdim));

    // output data descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&top_desc_));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(top_desc_, CUDNN_DataType,
                        4, outdim, outdim_stride));

    // convolution message descriptor
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc_, 2, pad, stride,
                        upscaleA, CUDNN_CROSS_CORRELATION, CUDNN_DataType));

    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    // get convolution forward algorithm
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle, bottom_desc_, filter_desc_,
                        conv_desc_, top_desc_,
			//CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
			//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                        free_byte, &fwd_algo_));

    // get convolution forward workspace size
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, bottom_desc_, filter_desc_,
                        conv_desc_, top_desc_, fwd_algo_, &fwd_sizeInBytes_));

    // for bug
    if(fwd_sizeInBytes_ > free_byte){
        fwd_algo_ = (cudnnConvolutionFwdAlgo_t)(1);
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, bottom_desc_, filter_desc_,
                            conv_desc_, top_desc_, fwd_algo_, &fwd_sizeInBytes_));
    }

    CUDA_CHECK(cudaMalloc((void **)&workSpace_, fwd_sizeInBytes_));

    // first run
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha,
                bottom_desc_, d_in, filter_desc_, d_flt,
                conv_desc_, fwd_algo_, workSpace_, fwd_sizeInBytes_, &beta,
                top_desc_, d_out));

    gettimeofday(&stime, NULL);
    cudaEventRecord(scuda, 0);
    for(j = 0; j < CYCLE; j++){
        CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha,
                    bottom_desc_, d_in, filter_desc_, d_flt,
                    conv_desc_, fwd_algo_, workSpace_, fwd_sizeInBytes_, &beta,
                    top_desc_, d_out));
    }
    CUDA_CHECK(cudaEventRecord(ecuda, 0));
    CUDA_CHECK(cudaEventSynchronize(scuda));
    CUDA_CHECK(cudaEventSynchronize(ecuda));
    gettimeofday(&etime ,NULL);

    time = (1.0*(etime.tv_sec - stime.tv_sec)*1000 + 1.0*(etime.tv_usec - stime.tv_usec)/1000)/CYCLE;
    gflops = (1.0*((2*R*S-1)*C+(C-1))*outH/1e3*outW*K/1e3*N)/time;
    
    //fprintf(stdout, "[N C K (inH inW) (R S) (outH outW)] = ");
    fprintf(stdout, "[%4d %4d %4d (%4d %4d) (%3d %3d) (%4d %4d) (%3d %3d) (%3d %3d)] ",
            N, C, K, inH, inW, R, S, outH, outW, padh, padw, strideh, stridew);
    fprintf(stdout, "|| elapsed-time %8.2f ms || Perf %7.2f gflops, %6.2f peak!\n",
            time, gflops, gflops/1e3/PEAK_PERF*100);
    
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outsize*sizeof(DataType), cudaMemcpyDeviceToHost));

#ifdef VERITY_RESULT
    verityConvolutionResult(pcm, h_in, h_flt, h_out);
#endif

    GPUFREE(workSpace_);

    CPUFREE(h_in);
    CPUFREE(h_flt);
    CPUFREE(h_out);
    GPUFREE(d_in);
    GPUFREE(d_flt);
    GPUFREE(d_out);

    CUDA_CHECK(cudaEventDestroy(scuda));
    CUDA_CHECK(cudaEventDestroy(ecuda));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));


}
