#include "def.h"

void fileInit4dData(FILE *pfile, DataType *data, int *dim)
{
    int i, j;
    int size = dim[0]*dim[1]*dim[2]*dim[3];

    if(size < 2000){
        fseek(pfile, 0, SEEK_SET);
        for(i = 0; i < size; i++)
            fscanf(pfile, "%lf", data+i);
    }else{
        for(i = 0; i < dim[0]*dim[1]; i++){
            fseek(pfile, i*4, SEEK_SET);
            for(j = 0; j < dim[2]*dim[3]; j++)
                fscanf(pfile, "%lf", data+(i*dim[2]*dim[3]+j));
        }
    }
}

void randomInit4dData(DataType *data, int size)
{
    int i;
    for(i = 0; i < size; i++)
        data[i] = 0.1*(rand()%10);
}

void zeroInit4dData(DataType *data, int size)
{
    int i;
    for(i = 0; i < size; i++)
        data[i] = 0.0;
}

void compare4dData(DataType *out, DataType *out_verify, int size)
{
    int i;
    for(i = 0; i < size; i++){
        if((out[i]-out_verify[i] > 1e-10) || (out[i]-out_verify[i] < -1e-10)){
            fprintf(stdout, "\033[31mError: out[%d]=%lf, out_verify[%d]=%lf.\033[0m\n",
                    i, out[i], i, out_verify[i]);
            break;
        }
    }

    if(i == size){
        fprintf(stdout, "Correctness!\n");
    }
}
