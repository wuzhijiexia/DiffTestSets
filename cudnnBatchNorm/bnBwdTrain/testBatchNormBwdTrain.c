#include <stdlib.h>
#include <stdio.h>
#include "def.h"

int main(int argc, char *argv[])
{
    int i, j, k;
    int N, C, H, W;

    DataType *in, *indiff, *outdiff;
    DataType *bnScale, *saveMean, *saveInvVariance;
    DataType *bndiffScale, *bndiffBias;
    DataType alphaDataDiff, betaDataDiff;
    DataType alphaParamDiff, betaParamDiff;
    DataType epsilon;
    
    int mode;
    int indim[4], bndim[4];
    int insize, bnsize;

#ifdef PERF
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig.txt", "r+");
#else
    FILE *pfconfig = fopen("./BatchNorm_nchwConfig_correctness.txt", "r+");
#endif
    
    FILE *pfdata = fopen("./data.txt", "r+");
    FILE *pfoutput = fopen("./cudnn_BatchNormBwdTrain_nchw.txt", "r+");
    if((pfconfig == NULL) || (pfoutput == NULL) || (pfdata == NULL)){
        CODE_MESSAGE("This is a error!");
        exit(1);
    }
    ftruncate(fileno(pfoutput), 0); // 清空输出文件
    fseek(pfoutput, 0, SEEK_SET);

    batchNormMess bnMess;
    bnMess = (struct batchNormMessStruct*)malloc(sizeof(struct batchNormMessStruct));

    alphaDataDiff   = 1.0;
    betaDataDiff    = 0.0;
    alphaParamDiff  = 1.0;
    betaParamDiff   = 0.0;
    epsilon         = 0.001;

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
        fprintf(stdout, "   [alphaDataDiff betaDataDiff alphaParamDiff betaParamDiff epsilon]"
                " = [%lf %lf %lf %lf %lf]\n",
                alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, epsilon);

        in              = (DataType*)malloc(insize*sizeof(DataType));
        indiff          = (DataType*)malloc(insize*sizeof(DataType));
        outdiff         = (DataType*)malloc(insize*sizeof(DataType));
        bnScale         = (DataType*)malloc(bnsize*sizeof(DataType));
        saveMean        = (DataType*)malloc(bnsize*sizeof(DataType));
        saveInvVariance = (DataType*)malloc(bnsize*sizeof(DataType));
        bndiffScale     = (DataType*)malloc(bnsize*sizeof(DataType));
        bndiffBias      = (DataType*)malloc(bnsize*sizeof(DataType));

#if 0
        randomInit4dData(in, insize);
        randomInit4dData(indiff, insize);
        randomInit4dData(outdiff, insize);
        randomInit4dData(bnScale, bnsize);
        randomInit4dData(saveMean, bnsize);
        randomInit4dData(saveInvVariance, bnsize);
        randomInit4dData(bndiffScale, bnsize);
        randomInit4dData(bndiffBias, bnsize);
#else
        fileInit4dData(pfdata, in, indim);
        fileInit4dData(pfdata, indiff, indim);
        fileInit4dData(pfdata, outdiff, indim);
        fileInit4dData(pfdata, bnScale, bndim);
        fileInit4dData(pfdata, saveMean, bndim);
        fileInit4dData(pfdata, saveInvVariance, bndim);
        fileInit4dData(pfdata, bndiffScale, bndim);
        fileInit4dData(pfdata, bndiffBias, bndim);
#endif

        bnMess->N_              = N;
        bnMess->C_              = C;
        bnMess->H_              = H;
        bnMess->W_              = W;
        bnMess->epsilon_        = epsilon;
        bnMess->alphaDataDiff_  = alphaDataDiff;
        bnMess->betaDataDiff_   = betaDataDiff;
        bnMess->alphaParamDiff_ = alphaParamDiff;
        bnMess->betaParamDiff_  = betaParamDiff;
        
        bnMess->in_              = in;
        bnMess->indiff_          = indiff;
        bnMess->outdiff_         = outdiff;
        bnMess->bnScale_         = bnScale;
        bnMess->bndiffScale_     = bndiffScale;
        bnMess->bndiffBias_      = bndiffBias;
        bnMess->saveMean_        = saveMean;
        bnMess->saveInvVariance_ = saveInvVariance;

        // cudnn batchnorm fwdinference
        cudnn_batchNormBwdTrain(bnMess);

#ifndef PERF
        // print data to file
        for(i = 0; i < insize; i++)
            fprintf(pfoutput, "%lf %lf %lf\n",
                    in[i], indiff[i], outdiff[i]);
        for(i = 0; i < bnsize; i++)
            fprintf(pfoutput, "%lf %lf %lf %lf %lf\n",
                    bnScale[i], bndiffScale[i], bndiffBias[i],
                    saveMean[i], saveInvVariance[i]);
#endif

        CPUFREE(in);
        CPUFREE(indiff);
        CPUFREE(outdiff);
        CPUFREE(bnScale);
        CPUFREE(bndiffScale);
        CPUFREE(bndiffBias);
        CPUFREE(saveMean);
        CPUFREE(saveInvVariance);

        fscanf(pfconfig, "%d %d %d %d %d", &N, &C, &H, &W, &mode);
    }

    fclose(pfconfig);
    fclose(pfdata);
    fclose(pfoutput);

    return 0;
}
