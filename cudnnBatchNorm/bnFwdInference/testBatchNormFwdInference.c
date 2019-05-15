#include <stdlib.h>
#include <stdio.h>
#include "def.h"

int main(int argc, char *argv[])
{
    int i, j, k;
    int N, C, H, W;
    DataType *in, *out;
    DataType *bnScale, *bnBias, *runMean, *runVariance;
    DataType epsilon, alpha, beta;
    int mode;
    int indim[4], bndim[4];
    int insize, bnsize;

#ifdef PERF
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig.txt", "r+");
#else
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig_correctness.txt", "r+");
#endif
    
    FILE *pfdata = fopen("./data.txt", "r+");
    FILE *pfoutput = fopen("./cudnn_BatchNormFwdInf_nchw.txt", "r+");
    if((pfconfig == NULL) || (pfoutput == NULL) || (pfdata == NULL)){
        CODE_MESSAGE("This is a error!");
        exit(1);
    }
    ftruncate(fileno(pfoutput), 0); // 清空输出文件
    fseek(pfoutput, 0, SEEK_SET);

    batchNormMess bnMess;
    bnMess = (struct batchNormMessStruct*)malloc(sizeof(struct batchNormMessStruct));

    alpha = 1.0;
    beta = 0.0;
    epsilon = 0.001;

    fscanf(pfconfig, "%d %d %d %d %d", &N, &C, &H, &W, &mode);
    while(!feof(pfconfig)){
        indim[0]=N; indim[1]=C; indim[2]=H; indim[3]=W;
        insize = N*C*H*W;
        switch(mode){
            case 0:
                bnMess->mode_ = PER_ACTIVATION;
                bndim[0]=1; bndim[1]=C; bndim[2]=H; bndim[3]=W;
                bnsize = C*H*W;
                break;
            case 1:
                bnMess->mode_ = SPATIAL;
                bndim[0]=1; bndim[1]=C; bndim[2]=1; bndim[3]=1;
                bnsize = C;
                break;
            default:
                CODE_MESSAGE("This is a error!");
                break;
        }

        fprintf(stdout, "BatchNorm Message:\n");
        fprintf(stdout, "   [N C H W mode] = [%d %d %d %d %d]\n",
                N, C, H, W, mode);
        fprintf(stdout, "   [alpha beta epsilon] = [%lf %lf %lf]\n",
                alpha, beta, epsilon);

        in      = (DataType*)malloc(insize*sizeof(DataType));
        out     = (DataType*)malloc(insize*sizeof(DataType));
        bnScale = (DataType*)malloc(bnsize*sizeof(DataType));
        bnBias  = (DataType*)malloc(bnsize*sizeof(DataType));
        runMean = (DataType*)malloc(bnsize*sizeof(DataType));
        runVariance = (DataType*)malloc(bnsize*sizeof(DataType));

#if 0
        randomInit4dData(in, insize);
        randomInit4dData(out, insize);
        randomInit4dData(bnScale, bnsize);
        randomInit4dData(bnBias, bnsize);
        randomInit4dData(runMean, bnsize);
        randomInit4dData(runVariance, bnsize);
#else
        fileInit4dData(pfdata, in, indim);
        fileInit4dData(pfdata, out, indim);
        fileInit4dData(pfdata, bnScale, bndim);
        fileInit4dData(pfdata, bnBias, bndim);
        fileInit4dData(pfdata, runMean, bndim);
        fileInit4dData(pfdata, runVariance, bndim);
#endif

        bnMess->N_          = N;
        bnMess->C_          = C;
        bnMess->H_          = H;
        bnMess->W_          = W;
        bnMess->in_         = in;
        bnMess->out_        = out;
        bnMess->bnScale_    = bnScale;
        bnMess->bnBias_     = bnBias;
        bnMess->runMean_ = runMean;
        bnMess->runVariance_ = runVariance;
        bnMess->alpha_      = alpha;
        bnMess->beta_       = beta;
        bnMess->epsilon_    = epsilon;

        // cudnn batchnorm fwdinference
        cudnn_batchNormFwdInference(bnMess);

#ifndef PERF
        // print output data to file
        for(i = 0; i < N; i++)
            for(j = 0; j < C; j++){
                for(k = 0; k < H*W; k++){
                    fprintf(pfoutput, "%lf ", out[i*C*H*W+j*H*W+k]);
                }
                fprintf(pfoutput, "\n");
            }
#endif

        CPUFREE(in);
        CPUFREE(out);
        CPUFREE(bnScale);
        CPUFREE(bnBias);
        CPUFREE(runMean);
        CPUFREE(runVariance);

        fscanf(pfconfig, "%d %d %d %d %d", &N, &C, &H, &W, &mode);
    }

    fclose(pfconfig);
    fclose(pfdata);
    fclose(pfoutput);

    return 0;
}
