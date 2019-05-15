#include <stdlib.h>
#include <stdio.h>
#include "def.h"

int main(int argc, char *argv[])
{
    int i, j, k;
    int N, C, H, W;
    DataType *in, *out;
    DataType *bnScale, *bnBias;
    DataType *runMean, *runVariance, *saveMean, *saveInvVariance;
    DataType epsilon, alpha, beta, expAvgFactor;
    int mode;
    int indim[4], bndim[4];
    int insize, bnsize;

#ifdef PERF
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig.txt", "r+");
#else
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig_correctness.txt", "r+");
#endif

    FILE *pfdata = fopen("./data.txt", "r+");
    FILE *pfoutput = fopen("./cudnn_BatchNormFwdTrain_nchw.txt", "r+");
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
    expAvgFactor = 0.2;

    fscanf(pfconfig, "%d %d %d %d %d", &N, &C, &H, &W, &mode);
    while(!feof(pfconfig)){
        indim[0]=N; indim[1]=C; indim[2]=H; indim[3]=W;
        insize = N*C*H*W;
        switch(mode){
            case PER_ACTIVATION:
                bnMess->mode_ = PER_ACTIVATION;
                bndim[0]=1; bndim[1]=C; bndim[2]=H; bndim[3]=W;
                bnsize = C*H*W;
                break;
            case SPATIAL:
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
        fprintf(stdout, "   [alpha beta epsilon factor] = [%lf %lf %lf %lf]\n",
                alpha, beta, epsilon, expAvgFactor);

        in              = (DataType*)malloc(insize*sizeof(DataType));
        out             = (DataType*)malloc(insize*sizeof(DataType));
        bnScale         = (DataType*)malloc(bnsize*sizeof(DataType));
        bnBias          = (DataType*)malloc(bnsize*sizeof(DataType));
        runMean         = (DataType*)malloc(bnsize*sizeof(DataType));
        runVariance     = (DataType*)malloc(bnsize*sizeof(DataType));
        saveMean        = (DataType*)malloc(bnsize*sizeof(DataType));
        saveInvVariance    = (DataType*)malloc(bnsize*sizeof(DataType));

#if 0
        randomInit4dData(in, insize);
        randomInit4dData(out, insize);
        randomInit4dData(bnScale, bnsize);
        randomInit4dData(bnBias, bnsize);
        randomInit4dData(runMean, bnsize);
        randomInit4dData(runVariance, bnsize);
        randomInit4dData(saveMean, bnsize);
        randomInit4dData(saveInvVariance, bnsize);
#else
        fileInit4dData(pfdata, in, indim);
        fileInit4dData(pfdata, out, indim);
        fileInit4dData(pfdata, bnScale, bndim);
        fileInit4dData(pfdata, bnBias, bndim);
        fileInit4dData(pfdata, runMean, bndim);
        fileInit4dData(pfdata, runVariance, bndim);
        fileInit4dData(pfdata, saveMean, bndim);
        fileInit4dData(pfdata, saveInvVariance, bndim);
#endif

        bnMess->N_              = N;
        bnMess->C_              = C;
        bnMess->H_              = H;
        bnMess->W_              = W;
        bnMess->alpha_          = alpha;
        bnMess->beta_           = beta;
        bnMess->epsilon_        = epsilon;
        bnMess->expAvgFactor_   = expAvgFactor;
        
        bnMess->in_             = in;
        bnMess->out_            = out;
        bnMess->bnScale_        = bnScale;
        bnMess->bnBias_         = bnBias;
        bnMess->runMean_        = runMean;
        bnMess->runVariance_    = runVariance;
        bnMess->saveMean_       = saveMean;
        bnMess->saveInvVariance_   = saveInvVariance;

        // cudnn batchnorm fwdinference
        cudnn_batchNormFwdTrain(bnMess);

#ifndef PERF
        // print output data to file
        for(i = 0; i < insize; i++)
            fprintf(pfoutput, "%lf %lf\n", in[i], out[i]);
        for(i = 0; i < bnsize; i++)
            fprintf(pfoutput, "%lf %lf %lf %lf\n",
                    saveMean[i], saveInvVariance[i],
                    runMean[i], runVariance[i]);
#endif

        CPUFREE(in);
        CPUFREE(out);
        CPUFREE(bnScale);
        CPUFREE(bnBias);
        CPUFREE(runMean);
        CPUFREE(runVariance);
        CPUFREE(saveMean);
        CPUFREE(saveInvVariance);

        fscanf(pfconfig, "%d %d %d %d %d", &N, &C, &H, &W, &mode);
    }

    fclose(pfconfig);
    fclose(pfdata);
    fclose(pfoutput);

    return 0;
}
