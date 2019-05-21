#include "def.h"

void simple_convBwdBias(convMess cm)
{
    fprintf(stdout, "\033\[31mSimple Convolution Backward for Bias ......\033\[0m\n");

    int N, K, outH, outW;
    int n, k, oh, ow;
    int outidx;

    DataType *outdiff, *bdiff;

    struct timeval stime, etime;
    double gflops, time_ms;
    
    N       = cm->N_;
    K       = cm->K_;
    outH    = cm->outH_;
    outW    = cm->outW_;
    
    outdiff   = cm->outdiff_;
    bdiff     = cm->bdiff_;
    
    int outdim[] = {N, K, outH, outW};
    int outdimInv[] = {K*outH*outW, outH*outW, outW, 1};
    int outsize = N*K*outH*outW;

    // kernel compute for convolution backward bias
    for(k = 0; k < K; k++){
        bdiff[k] = 0.0;
        for(n = 0; n < N; n++){
            for(oh = 0; oh < outH; oh++){
                for(ow = 0; ow < outW; ow++){
                    outidx = n*outdimInv[0]+k*outdimInv[1]+oh*outdimInv[2]+ow;
                    bdiff[k] += outdiff[outidx];
                }
            }
        }
    }
}
