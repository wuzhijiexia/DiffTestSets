#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mkldnn_conv.h"

#define PADH         1
#define PADW         1
#define STRIDEH      1
#define STRIDEW      1

int main(int argc, char *argv[]){
    convMess cm;
    FILE *dfile = fopen("../convConfigures-test.txt", "r+");
    if(dfile == NULL) {
        CODE_MESSAGE();
        exit(1);
    }

    cm.padh_ = PADH;
    cm.padw_ = PADW;
    cm.strideh_ = STRIDEH;
    cm.stridew_ = STRIDEW;
    fscanf(dfile, "%d %d %d %d %d %d %d",
            &cm.N_, &cm.C_, &cm.K_, &cm.inH_, &cm.inW_, &cm.R_, &cm.S_);
    cm.outH_ = (cm.inH_ + 2*cm.padh_ - cm.R_) / cm.strideh_ + 1;
    cm.outW_ = (cm.inW_ + 2*cm.padw_ - cm.S_) / cm.stridew_ + 1;
    mkldnnConvolutionTest(&cm);
#if 0
    while(!feof(dfile)){
        cm.outH_ = (cm.inH_ + 2*cm.padh_ - cm.R_) / cm.strideh_ + 1;
        cm.outW_ = (cm.inW_ + 2*cm.padw_ - cm.S_) / cm.stridew_ + 1;
        mkldnnConvolutionTest(&cm);
#if 0
    fprintf(stdout, "%d %d %d %d %d %d %d\n",
            cm.N_, cm.C_, cm.K_, cm.inH_, cm.inW_, cm.R_, cm.S_);
#endif
        fscanf(dfile, "%d %d %d %d %d %d %d",
                &cm.N_, &cm.C_, &cm.K_, &cm.inH_, &cm.inW_, &cm.R_, &cm.S_);
    }
#endif

    fclose(dfile);

    return 0;
}
