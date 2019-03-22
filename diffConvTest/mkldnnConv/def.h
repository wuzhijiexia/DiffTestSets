#ifndef _DEF_H_
#define _DEF_H_

#define false       0
#define true        1

#define DataDim     4

#ifdef DATA_FLOAT
#define DataType    float
#endif

#ifdef DATA_DOUBLE
#define DataType    double
#endif

#define CODE_MESSAGE() \
    do { \
        fprintf(stdout, "%s : %s : %d!\n", __FILE__, __FUNCTION__, __LINE__); \
    }while(false)

#define CPUFREE(pdata) \
    do{ \
        if(pdata != NULL) free(pdata); \
    }while(false)

typedef struct Conv_Message {
    int N_;
    int C_;
    int inH_;
    int inW_;
    int outH_;
    int outW_;
    int K_;
    int R_;
    int S_;
    int padh_;
    int padw_;
    int strideh_;
    int stridew_;
}convMess;

void verityConvolutionResult(convMess*, DataType*, DataType*, DataType*);
#endif
