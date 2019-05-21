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

    DataType *in, *flt, *out;
    int indim[4], fltdim[4], outdim[4];
    int insize, fltsize, outsize;
    DataType alpha, beta;

    FILE *pfconfig = fopen("./data/conv_nchwConfig.txt", "r+");
    FILE *pfdata = fopen("./data/data.txt", "r+");
    FILE *pfoutput = fopen("./data/simple_convFwd_nchw.txt", "r+");
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

        indim[0]=N; indim[1]=C; indim[2]=inH; indim[3]=inW;
        fltdim[0]=K; fltdim[1]=C; fltdim[2]=fltH; fltdim[3]=fltW;
        outdim[0]=N; outdim[1]=K; outdim[2]=outH; outdim[3]=outW;

        insize = indim[0]*indim[1]*indim[2]*indim[3];
        fltsize = fltdim[0]*fltdim[1]*fltdim[2]*fltdim[3];
        outsize = outdim[0]*outdim[1]*outdim[2]*outdim[3];

        in  = (DataType*)malloc(insize*sizeof(DataType));
        flt = (DataType*)malloc(fltsize*sizeof(DataType));
        out = (DataType*)malloc(outsize*sizeof(DataType));

        fileInit4dData(pfdata, in, indim); // file init data
        fileInit4dData(pfdata, flt, fltdim);
        fileInit4dData(pfdata, out, outdim);

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
        
        cm->in_     = in;
        cm->flt_    = flt;
        cm->out_    = out;
        cm->alpha_  = alpha;
        cm->beta_   = beta;

        simple_convFwd(cm); // simple convolution forward
        
        fprintf(stdout, "=============================================================\n");
        
#ifndef PERF
        for(i = 0; i < N*C*inH; i++){
            for(j = 0; j < inW; j++){
                fprintf(pfoutput, "%lf ", in[i*inW+j]);
            }
            fprintf(pfoutput, "\n");
        }

        for(i = 0; i < K*C*fltH; i++){
            for(j = 0; j < fltW; j++){
                fprintf(pfoutput, "%lf ", flt[i*fltW+j]);
            }
            fprintf(pfoutput, "\n");
        }

        for(i = 0; i < N*K*outH; i++){
            for(j = 0; j < outW; j++){
                fprintf(pfoutput, "%lf ", out[i*outW+j]);
            }
            fprintf(pfoutput, "\n");
        }
#endif

        // next get convolution configures
        fscanf(pfconfig, "%d %d %d %d %d %d %d %d %d %d %d",
               &N, &C, &K, &inH, &inW, &fltH, &fltW,
               &padh, &padw, &strideh, &stridew);

        CPUFREE(in);
        CPUFREE(flt);
        CPUFREE(out);
    }

    // clean environment
    fclose(pfconfig);
    fclose(pfdata);
    fclose(pfoutput);

    return 0;
}
