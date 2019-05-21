#include "def.h"

void simple_convBwdData(convMess cm)
{
    fprintf(stdout, "\033\[31mSimple Convolution Backward for Data ......\033\[0m\n");

    int N, C, inH, inW;
    int K, fltH, fltW;
    int outH, outW;

    int n, c, ih, iw;
    int k, fh, fw;
    int oh, ow;

    int inidx, fltidx, outidx;
    int tmp_oh, tmp_ow;
    int true_oh, true_ow;

    DataType *flt, *outdiff, *indiff;
    DataType alpha, beta;

    struct timeval stime, etime;
    double gflops, time_ms;

    N     = cm->N_;
    C     = cm->C_;
    inH   = cm->inH_;
    inW   = cm->inW_;
    K     = cm->K_;
    fltH  = cm->fltH_;
    fltW  = cm->fltW_;
    outH  = cm->outH_;
    outW  = cm->outW_;

    alpha = cm->alpha_;
    beta  = cm->beta_;

    flt     = cm->flt_;
    outdiff = cm->outdiff_;
    indiff  = cm->indiff_;

    int indim[] = {N, C, inH, inW};
    int indimInv[] = {C*inH*inW, inH*inW, inW, 1};
    int insize = N*C*inH*inW;

    int fltdim[] = {K, C, fltH, fltW};
    int fltdimInv[] = {C*fltH*fltW, fltH*fltW, fltW, 1};
    int fltsize = K*C*fltH*fltW;

    int outdim[] = {N, K, outH, outW};
    int outdimInv[] = {K*outH*outW, outH*outW, outW, 1};
    int outsize = N*K*outH*outW;

    int pad[] = {cm->padh_, cm->padw_};
    int stride[] = {cm->strideh_, cm->stridew_};

    // kernel compute for convolution backward for data
#pragma omp parallel for private(n,k,oh,ow,c,fh,fw,true_oh,true_ow,inidx,fltidx,outidx)
    for(n = 0; n < N; n++){
        for(c = 0; c < C; c++){
            for(ih = 0; ih < inH; ih++){
                for(iw = 0; iw < inW; iw++){
                    inidx = n*indimInv[0] + c*indimInv[1] + ih*indimInv[2] + iw;
                    indiff[inidx] = 0.0;
                    for(k = 0; k < K; k++){
                        for(fh = 0; fh < fltH; fh++){
                            for(fw = 0; fw < fltW; fw++){
                                tmp_oh = (ih+pad[0]-fh)%stride[0];
                                tmp_ow = (iw+pad[1]-fw)%stride[1];
                                true_oh = (ih+pad[0]-fh)/stride[0];
                                true_ow = (iw+pad[1]-fw)/stride[1];
                                if((tmp_oh==0) && (tmp_ow==0) && (true_oh>=0) && (true_ow>=0) && (true_oh<outH) && (true_ow<outW)){
                                    outidx = n*outdimInv[0] + k*outdimInv[1] + true_oh*outdimInv[2] + true_ow;
#ifdef ROTATE
                                    fltidx = k*fltdimInv[0] + c*fltdimInv[1] + (fltH-fh-1)*fltdimInv[2] + (fltW-fw-1);
#endif
#ifdef NOROTATE
                                    fltidx = k*fltdimInv[0] + c*fltdimInv[1] + fh*fltdimInv[2] + fw;
#endif
                                    indiff[inidx] += outdiff[outidx]*flt[fltidx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
