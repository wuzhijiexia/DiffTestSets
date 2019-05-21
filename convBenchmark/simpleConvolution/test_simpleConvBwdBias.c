#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "def.h"

int main(int argc, char *argv[]){
    int i, j, k;
    int N, C, inH, inW;
    int K, fltH, fltW;
    int outH, outW;
    int padh, padw;
    int strideh, stridew;

    DataType *outdiff, *bdiff;
    int outdim[4], bdim[4];
    int outsize, bsize;
    DataType alpha, beta;

    FILE *pfconfig = fopen("./data/conv_nchwConfig.txt", "r+");
    FILE *pfdata = fopen("./data/data.txt", "r+");
    FILE *pfoutput = fopen("./data/simple_convBwdBias_nchw.txt", "r+");
    if((pfconfig == NULL) && (pfdata == NULL) && (pfoutput == NULL)) {
        CODE_MESSAGE("This is a error!");
        exit(1);
    }
    ftruncate(fileno(pfoutput), 0); // 清空输出文件
    fseek(pfoutput, 0, SEEK_SET);

    convMess cm = (struct convMessStruct*)malloc(sizeof(struct convMessStruct));
    alpha = 1.0;
    beta = 0.0;
    
    // first get convolution configures
    fscanf(pfconfig, "%d %d %d %d %d %d %d %d %d %d %d",
           &N, &C, &K, &inH, &inW, &fltH, &fltW,
           &padh, &padw, &strideh, &stridew);
    while(!feof(pfconfig)){
        outH = (inH+2*padh-fltH)/strideh + 1;
        outW = (inW+2*padw-fltW)/stridew + 1;

        // print convolution message
        fprintf(stdout, "Simple Convolution Message:\n");
        fprintf(stdout, "input-dim[%d %d %d %d], filter-dim[%d %d %d %d]\n",
                N, C, inH, inW, K, C, fltH, fltW);
        fprintf(stdout, "[padh padw strideh stridew] = [%d %d %d %d]\n",
                padh, padw, strideh, stridew);
        fprintf(stdout, "output-dim[%d %d %d %d]\n", N, K, outH, outW);
        fprintf(stdout, "[alpha beta] = [%lf %lf]\n", alpha, beta);

        outdim[0]=N; outdim[1]=K; outdim[2]=outH; outdim[3]=outW;
        bdim[0]=1; bdim[1]=K; bdim[2]=1; bdim[3]=1;

        outsize = outdim[0]*outdim[1]*outdim[2]*outdim[3];
        bsize = bdim[0]*bdim[1]*bdim[2]*bdim[3];

        outdiff = (DataType*)malloc(outsize*sizeof(DataType));
        bdiff = (DataType*)malloc(bsize*sizeof(DataType));

        fileInit4dData(pfdata, outdiff, outdim); // file init data
        fileInit4dData(pfdata, bdiff, bdim);

        // configure convolution message
        cm->N_      = N;
        cm->C_      = C;
        cm->inH_    = inH;
        cm->inW_    = inW;
        cm->K_      = K;
        cm->fltH_   = fltH;
        cm->fltW_   = fltW;
        cm->outH_   = outH;
        cm->outW_   = outW;
        
        cm->padh_   = padh;
        cm->padw_   = padw;
        cm->strideh_ = strideh;
        cm->stridew_ = stridew;
        
        cm->outdiff_    = outdiff;
        cm->bdiff_    = bdiff;
        cm->alpha_  = alpha;
        cm->beta_   = beta;

        simple_convBwdBias(cm); // simple convolution forward for bias
        
        fprintf(stdout, "=============================================================\n");
        
#ifndef PERF
        for(i = 0; i < N*K*outH; i++){ // output diff
            for(j = 0; j < outW; j++){
                fprintf(pfoutput, "%lf ", outdiff[i*outW+j]);
            }
            fprintf(pfoutput, "\n");
        }

        for(i = 0; i < K; i++){ // bias diff
            fprintf(pfoutput, "%lf\n", bdiff[i]);
        }
#endif

        // next get convolution configures
        fscanf(pfconfig, "%d %d %d %d %d %d %d %d %d %d %d",
               &N, &C, &K, &inH, &inW, &fltH, &fltW,
               &padh, &padw, &strideh, &stridew);

        CPUFREE(outdiff);
        CPUFREE(bdiff);
    }

    // clean environment
    fclose(pfconfig);
    fclose(pfdata);
    fclose(pfoutput);

    return 0;
}
