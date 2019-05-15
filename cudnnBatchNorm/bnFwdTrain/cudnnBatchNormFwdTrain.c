#include "def.h"

void cudnn_batchNormFwdTrain(batchNormMess bnMess)
{
    int i, j, k;
    int N, C, H, W;
    int indim[4], indim_sd[4], insize;
    int bndim[4], bndim_sd[4], bnsize;
    DataType *h_in, *h_out;
    DataType *h_bnScale, *h_bnBias;
    DataType *h_runMean, *h_runVariance;
    DataType *h_saveMean, *h_saveInvVariance;
    DataType epsilon, alpha, beta, expAvgFactor;
    
    DataType *d_in, *d_out;
    DataType *d_bnScale, *d_bnBias;
    DataType *d_runMean, *d_runVariance;
    DataType *d_saveMean, *d_saveInvVariance;

    struct timeval tv_start, tv_end;
    DataType time_ms, gflop;

    cudnnHandle_t handle;
    cudnnBatchNormMode_t mode;
    cudnnTensorDescriptor_t in_desc;
    cudnnTensorDescriptor_t bn_desc;
    cudaEvent_t scuda, ecuda;

    // configure batchnorm
    N               = bnMess->N_;
    C               = bnMess->C_;
    H               = bnMess->H_;
    W               = bnMess->W_;
    alpha           = bnMess->alpha_;
    beta            = bnMess->beta_;
    epsilon         = bnMess->epsilon_;
    expAvgFactor    = bnMess->expAvgFactor_;
    
    h_in            = bnMess->in_;
    h_out           = bnMess->out_;
    h_bnScale       = bnMess->bnScale_;
    h_bnBias        = bnMess->bnBias_;
    h_runMean       = bnMess->runMean_;
    h_runVariance   = bnMess->runVariance_;
    h_saveMean      = bnMess->saveMean_;
    h_saveInvVariance  = bnMess->saveInvVariance_;

    indim[0] = N; indim[1] = C; indim[2] = H; indim[3] = W;
    indim_sd[0] = C*H*W; indim_sd[1] = H*W; indim_sd[2] = W; indim_sd[3] = 1;
    insize = N*C*H*W;

    cudnnCreate(&handle);
    cudaEventCreate(&scuda);
    cudaEventCreate(&ecuda);

    switch(bnMess->mode_){
        case PER_ACTIVATION:
            mode = CUDNN_BATCHNORM_PER_ACTIVATION;
            bndim[0] = 1; bndim[1] = C; bndim[2] = H; bndim[3] = W;
            bndim_sd[0] = C*H*W; bndim_sd[1] = H*W; bndim_sd[2] = W; bndim_sd[3] = 1;
            bnsize = C*H*W;
            break;
        case SPATIAL:
            mode = CUDNN_BATCHNORM_SPATIAL;
            bndim[0] = 1; bndim[1] = C; bndim[2] = 1; bndim[3] = 1;
            bndim_sd[0] = C; bndim_sd[1] = 1; bndim_sd[2] = 1; bndim_sd[3] = 1;
            bnsize = C;
            break;
        default:
            CODE_MESSAGE("This is a error!");
            break;
    }

    // malloc device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, insize*sizeof(DataType))); // in
    CUDA_CHECK(cudaMalloc((void**)&d_out, insize*sizeof(DataType))); // out
    CUDA_CHECK(cudaMalloc((void**)&d_bnScale, bnsize*sizeof(DataType))); // scale
    CUDA_CHECK(cudaMalloc((void**)&d_bnBias, bnsize*sizeof(DataType))); // bias
    CUDA_CHECK(cudaMalloc((void**)&d_runMean, bnsize*sizeof(DataType))); // run mean
    CUDA_CHECK(cudaMalloc((void**)&d_runVariance, bnsize*sizeof(DataType))); // run variance
    CUDA_CHECK(cudaMalloc((void**)&d_saveMean, bnsize*sizeof(DataType))); // save mean
    CUDA_CHECK(cudaMalloc((void**)&d_saveInvVariance, bnsize*sizeof(DataType))); // save variance
    
    CUDA_CHECK(cudaMemcpy(d_in, h_in, insize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bnScale, h_bnScale, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bnBias, h_bnBias, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_runMean, h_runMean, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_runVariance, h_runVariance, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_saveMean, h_saveMean, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_saveInvVariance, h_saveInvVariance, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));

    // create & set descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(in_desc, CUDNN_DataType,
            4, indim, indim_sd));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(bn_desc, CUDNN_DataType,
            4, bndim, bndim_sd));

    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining( // 第一次预热
            handle, mode,
            &alpha, &beta,
            in_desc, d_in, in_desc, d_out,
            bn_desc, d_bnScale, d_bnBias,
            expAvgFactor, d_runMean, d_runVariance, epsilon,
            d_saveMean, d_saveInvVariance
            ));
#ifdef PERF
    gettimeofday(&tv_start, NULL);
    cudaEventRecord(scuda, 0);
    for(j = 0; j < CYCLE; j++){
        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
                handle, mode,
                &alpha, &beta,
                in_desc, d_in, in_desc, d_out,
                bn_desc, d_bnScale, d_bnBias,
                expAvgFactor, d_runMean, d_runVariance, epsilon,
                d_saveMean, d_saveInvVariance
                ));
    }
    cudaEventRecord(ecuda, 0);
    cudaEventSynchronize(scuda);
    cudaEventSynchronize(ecuda);
    gettimeofday(&tv_end, NULL);
    time_ms = TIME_MS(tv_start, tv_end)/CYCLE;
    fprintf(stdout, "cudnn  bnfwdinf >>> Time: %.4lf ms.\n", time_ms);
    fprintf(stdout, "===================================================\n");
#endif

    // out|runMean|runVariance|saveMean|saveInvVariance form device to host
    cudaMemcpy(h_out, d_out, insize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_runMean, d_runMean, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_runVariance, d_runVariance, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_saveMean, d_saveMean, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_saveInvVariance, d_saveInvVariance, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);

    // clean environment
    GPUFREE(d_in);
    GPUFREE(d_out);
    GPUFREE(d_bnScale);
    GPUFREE(d_bnBias);
    GPUFREE(d_runMean);
    GPUFREE(d_runVariance);
    GPUFREE(d_saveMean);
    GPUFREE(d_saveInvVariance);
    
    cudaEventDestroy(scuda);
    cudaEventDestroy(ecuda);
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}
