#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "def.h"

void verityConvolutionResult(convMess *pcm,
        DataType* in, DataType* flt, DataType* out)
{
    // Convolution Configure
    int N, C, K, inH, inW;
    int R, S;
    int padh, padw, strideh, stridew;
    int outH, outW;
    int n, c, k, outh, outw, r, s;
    int indx, fltdx, outdx;
    DataType *verity_out;

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

    verity_out = (DataType*)malloc(N*K*outH*outW*sizeof(DataType));
    memset(verity_out, 0, N*K*outH*outW*sizeof(DataType));
    
    int insd[] = {C*inH*inW, inH*inW, inW, 1};
    int fltsd[] = {C*R*S, R*S, S, 1};
    int outsd[] = {K*outH*outW, outH*outW, outW, 1};

    int true_inh, true_inw;
#pragma omp parallel for private(n,k,outh,outw,c,r,s,true_inh,true_inw,indx,fltdx,outdx)
    for(n = 0; n < N; n++){
        for(k = 0; k < K; k++){
            for(outh = 0; outh < outH; outh++){
                for(outw = 0; outw < outW; outw++){
                    outdx = n*outsd[0] + k*outsd[1] + outh*outsd[2] + outw;
                    for(c = 0; c < C; c++){
                        for(r = 0; r < R; r++){
                            for(s = 0; s < S; s++){
                                true_inh = outh*strideh + r - padh;
                                true_inw = outw*stridew + s - padw;
                                if((true_inh >= 0) && (true_inh < inH) && (true_inw >= 0) && (true_inw < inW)){
                                indx = n*insd[0] + c*insd[1] + (outh*strideh+r-padh)*insd[2] + (outw*stridew+s-padw);
                                fltdx = k*fltsd[0] + c*fltsd[1] + r*fltsd[2] + s;
                                verity_out[outdx] += in[indx]*flt[fltdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Verity Result
#pragma omp parallel for private(n)
    for(n = 0; n < N*outsd[0]; n++){
        if(abs(out[n]-verity_out[n]) > 1e-4){
            CODE_MESSAGE();
            fprintf(stdout, "out[%d] = %f, verity_out[%d] = %f!\n",
                    n, out[n], n, verity_out[n]);
            CPUFREE(verity_out);
            exit(1);
        }
    }
    
#if 0
    for(n = 0; n < N*outsd[0]; n++)
        printf("out[%d] = %f, verity_out[%d] = %f!\n",
                n, out[n], n, verity_out[n]);
#endif

    printf("The result is correct!\n");

    CPUFREE(verity_out);
}
