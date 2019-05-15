# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

# P100: arch=compute_60,code=sm_60
# k20: arch=compute_35,code=sm_35

GPUID = 1
AXIS = BN_SPATIAL # BN_SPATIAL or BN_PER_ACTIVATION
MODE = PERF # PERF or NULL

BNFWDINF_EXE = testBatchNormFwdInference
bnfwdinf:
	@rm -rf $(BNFWDINF_EXE)
	@nvcc -DDATA_DOUBLE -D$(AXIS) -D$(MODE) -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -I./ \
	    -o $(BNFWDINF_EXE) \
	    def.c \
	    ./bnFwdInference/cudnnBatchNormFwdInference.c \
	    ./bnFwdInference/testBatchNormFwdInference.c \
	    -lcuda -lcudnn
	@echo "Begin to run BN Inference Forward......"
	@export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(BNFWDINF_EXE)

BNFWDTRAIN_EXE = testBatchNormFwdTrain
bnfwdtrain:
	@rm -rf $(BNFWDTRAIN_EXE)
	@nvcc -DDATA_DOUBLE -D$(AXIS) -D$(MODE) -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -I./ \
	    -o $(BNFWDTRAIN_EXE) \
	    def.c \
	    ./bnFwdTrain/cudnnBatchNormFwdTrain.c \
	    ./bnFwdTrain/testBatchNormFwdTrain.c \
	    -lcuda -lcudnn
	@echo "Begin to run BN Train Forward......"
	@export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(BNFWDTRAIN_EXE)

BNBWDTRAIN_EXE = testBatchNormBwdTrain
bnbwdtrain:
	@rm -rf $(BNBWDTRAIN_EXE)
	@nvcc -DDATA_DOUBLE -D$(AXIS) -D$(MODE) -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -I./ \
	    -o $(BNBWDTRAIN_EXE) \
	    def.c \
	    ./bnBwdTrain/cudnnBatchNormBwdTrain.c \
	    ./bnBwdTrain/testBatchNormBwdTrain.c \
	    -lcuda -lcudnn
	@echo "Begin to run BN Train Backward......"
	@export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(BNBWDTRAIN_EXE)

COMPARE_EXE = compareFileData
compare:
	@rm -rf $(COMPARE_EXE)
	@gcc -o $(COMPARE_EXE) ./compareFileData.c
	@echo "Begin to run ......"
	@./$(COMPARE_EXE) $(FILE)

.PHONY: clean
clean:
	rm -rf $(BNFWDINF_EXE) $(BNFWDTRAIN_EXE) $(BNBWDTRAIN_EXE) $(COMPARE_EXE)
