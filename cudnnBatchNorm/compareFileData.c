#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
    double data, data_verify;
    FILE *pfile = fopen("./output.txt", "r+");
    FILE *pfile_verify = fopen(argv[1], "r+");

    if((pfile == NULL) || (pfile_verify == NULL)){
        fprintf(stdout, "This is a error!\n");
        exit(0);
    }

    fscanf(pfile, "%lf", &data);
    fscanf(pfile_verify, "%lf", &data_verify);
    while((!feof(pfile) || (!feof(pfile_verify)))){
        if(fabs(data-data_verify) > 1e-10){
            fprintf(stdout, "data=%lf, data_verify=%lf\n",
                    data, data_verify);
        }
        fscanf(pfile, "%lf", &data);
        fscanf(pfile_verify, "%lf", &data_verify);
    }

    fclose(pfile);
    fclose(pfile_verify);

    return 0;
}
