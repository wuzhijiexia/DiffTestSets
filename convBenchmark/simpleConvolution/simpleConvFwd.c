#include "def.h"

void simple_convFwd(convMess cm)
{
    fprintf(stdout, "\033\[31mSimple Convolution Forward ......\033\[0m\n");

    int N, C, inH, inW;
    int K, fltH, fltW;
    int outH, outW;

    int n, c, ih, iw;
    int k, fh, fw;
    int oh, ow;

    int inidx, fltidx, outidx;
    int true_ih, true_iw;

    DataType *in, *flt, *out;
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

    in    = cm->in_;
    flt   = cm->flt_;
    out   = cm->out_;

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

    // kernel compute for convolution forward

#pragma omp parallel for private(n,k,oh,ow,c,fh,fw,true_ih,true_iw,inidx,fltidx,outidx)
    for(n = 0; n < N; n++){
        for(k = 0; k < K; k++){
            for(oh = 0; oh < outH; oh++){
                for(ow = 0; ow < outW; ow++){
                    outidx = n*outdimInv[0] + k*outdimInv[1] + oh*outdimInv[2] + ow;
                    out[outidx] = 0.0; // output初始化零
                    for(c = 0; c < C; c++){
                        for(fh = 0; fh < fltH; fh++){
                            for(fw = 0; fw < fltW; fw++){
                                true_ih = oh*stride[0] + fh - pad[0];
                                true_iw = ow*stride[1] + fw - pad[1];
                                if((true_ih >= 0) && (true_ih < inH) && (true_iw >= 0) && (true_iw < inW)){
                                    inidx = n*indimInv[0] + c*indimInv[1] + true_ih*indimInv[2] + true_iw;
#ifdef ROTATE
                                    fltidx = k*fltdimInv[0] + c*fltdimInv[1] + (fltH-fh-1)*fltdimInv[2] + (fltW-fw-1);
#endif
#ifdef NOROTATE
                                    fltidx = k*fltdimInv[0] + c*fltdimInv[1] + fh*fltdimInv[2] + fw;
#endif
                                    out[outidx] += in[inidx]*flt[fltidx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
