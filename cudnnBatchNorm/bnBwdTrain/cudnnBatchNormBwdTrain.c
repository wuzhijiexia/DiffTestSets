#include "def.h"

void cudnn_batchNormBwdTrain(batchNormMess bnMess)
{
    int i, j, k;
    int N, C, H, W;
    int indim[4], indim_sd[4], insize;
    int bndim[4], bndim_sd[4], bnsize;
    
    DataType *h_in, *h_indiff, *h_outdiff;
    DataType *h_bnScale;
    DataType *h_saveMean, *h_saveInvVariance;
    DataType *h_bndiffScale, *h_bndiffBias;
    
    DataType *d_in, *d_indiff, *d_outdiff;
    DataType *d_bnScale;
    DataType *d_saveMean, *d_saveInvVariance;
    DataType *d_bndiffScale, *d_bndiffBias;
    
    DataType alphaDataDiff, betaDataDiff;
    DataType alphaParamDiff, betaParamDiff;
    DataType epsilon;

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
    epsilon         = bnMess->epsilon_;
    alphaDataDiff   = bnMess->alphaDataDiff_;
    betaDataDiff    = bnMess->betaDataDiff_;
    alphaParamDiff  = bnMess->alphaParamDiff_;
    betaParamDiff   = bnMess->betaParamDiff_;
    
    h_in                = bnMess->in_;
    h_indiff            = bnMess->indiff_;
    h_outdiff           = bnMess->outdiff_;
    h_bnScale           = bnMess->bnScale_;
    h_saveMean          = bnMess->saveMean_;
    h_saveInvVariance   = bnMess->saveInvVariance_;
    h_bndiffScale       = bnMess->bndiffScale_;
    h_bndiffBias        = bnMess->bndiffBias_;

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
    CUDA_CHECK(cudaMalloc((void**)&d_indiff, insize*sizeof(DataType))); // indiff
    CUDA_CHECK(cudaMalloc((void**)&d_outdiff, insize*sizeof(DataType))); // outdiff
    CUDA_CHECK(cudaMalloc((void**)&d_bnScale, bnsize*sizeof(DataType))); // scale
    CUDA_CHECK(cudaMalloc((void**)&d_saveMean, bnsize*sizeof(DataType))); // save mean
    CUDA_CHECK(cudaMalloc((void**)&d_saveInvVariance, bnsize*sizeof(DataType))); // save variance
    CUDA_CHECK(cudaMalloc((void**)&d_bndiffScale, bnsize*sizeof(DataType))); // scalediff
    CUDA_CHECK(cudaMalloc((void**)&d_bndiffBias, bnsize*sizeof(DataType))); // biasdiff
    
    CUDA_CHECK(cudaMemcpy(d_in, h_in, insize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indiff, h_indiff, insize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outdiff, h_outdiff, insize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bnScale, h_bnScale, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_saveMean, h_saveMean, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_saveInvVariance, h_saveInvVariance, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bndiffScale, h_bndiffScale, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bndiffBias, h_bndiffBias, bnsize*sizeof(DataType), cudaMemcpyHostToDevice));

    // create & set descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(in_desc, CUDNN_DataType,
            4, indim, indim_sd));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(bn_desc, CUDNN_DataType,
            4, bndim, bndim_sd));

    CUDNN_CHECK(cudnnBatchNormalizationBackward( // 第一次预热
            handle, mode,
            &alphaDataDiff, &betaDataDiff,
            &alphaParamDiff, &betaParamDiff,
            in_desc, d_in, in_desc, d_outdiff, in_desc, d_indiff,
            bn_desc, d_bnScale, d_bndiffScale, d_bndiffBias,
            epsilon, d_saveMean, d_saveInvVariance
            ));
#ifdef PERF
    gettimeofday(&tv_start, NULL);
    cudaEventRecord(scuda, 0);
    for(j = 0; j < CYCLE; j++){
        CUDNN_CHECK(cudnnBatchNormalizationBackward(
                handle, mode,
                &alphaDataDiff, &betaDataDiff,
                &alphaParamDiff, &betaParamDiff,
                in_desc, d_in, in_desc, d_outdiff, in_desc, d_indiff,
                bn_desc, d_bnScale, d_bndiffScale, d_bndiffBias,
                epsilon, d_saveMean, d_saveInvVariance
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
    cudaMemcpy(h_indiff, d_indiff, insize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bndiffScale, d_bndiffScale, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bndiffBias, d_bndiffBias, bnsize*sizeof(DataType), cudaMemcpyDeviceToHost);

    // clean environment
    GPUFREE(d_in);
    GPUFREE(d_indiff);
    GPUFREE(d_outdiff);
    GPUFREE(d_bnScale);
    GPUFREE(d_saveMean);
    GPUFREE(d_saveInvVariance);
    GPUFREE(d_bndiffScale);
    GPUFREE(d_bndiffBias);
    
    cudaEventDestroy(scuda);
    cudaEventDestroy(ecuda);
    cudnnDestroy(handle);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}
