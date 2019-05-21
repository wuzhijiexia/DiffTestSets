# -DDATA_FLOAT | -DDATA_DOUBLE (Data type for pragram)
# -DVERITY_RESULT (Whether verity the correctness)

# TITAN V: arch=compute_70,code=sm_70
# P100:    arch=compute_60,code=sm_60
# k20:     arch=compute_35,code=sm_35

GPUID = 1
DATATYPE = DATA_DOUBLE # DATA_DOUBLE or DATA_FLOAT
CONVMODE = ROTATE # ROTATE or NOROTATE
MODE = NULL # PERF or NULL

CUDNN_CFLAGS += -D$(DATATYPE) -D$(CONVMODE) -D$(MODE)
CUDNN_CFLAGS += -gencode arch=compute_70,code=sm_70 -Xcompiler -fopenmp -I./assister/
SIMPLE_CFLAGS += -D$(DATATYPE) -D$(CONVMODE) -D$(MODE)
SIMPLE_CFLAGS += -Wno-unused-result -O3 -fopenmp -I./assister/

# convolution forward
CUDNN_CONVFWD_EXE = ./tools/test_cudnnConvFwd
SIMPLE_CONVFWD_EXE = ./tools/test_simpleConvFwd
convfwd:
	rm -rf $(CUDNN_CONVFWD_EXE) $(SIMPLE_CONVFWD_EXE)
	nvcc $(CUDNN_CFLAGS) -o $(CUDNN_CONVFWD_EXE) \
	    ./assister/def.c ./cudnnConvolution/cudnnConvFwd.c ./cudnnConvolution/test_cudnnConvFwd.c \
	    -lcuda -lcudnn
	gcc $(SIMPLE_CFLAGS) -o $(SIMPLE_CONVFWD_EXE) \
	    ./assister/def.c ./simpleConvolution/simpleConvFwd.c ./simpleConvolution/test_simpleConvFwd.c
	@echo "Begin to run cudnn|simple convolution forward ......"
	@export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(CUDNN_CONVFWD_EXE)
	@./$(SIMPLE_CONVFWD_EXE)
	@echo "End to run cudnn|simple convolution forward ......"

# convolution backward for data
CUDNN_CONVBWDDATA_EXE = ./tools/test_cudnnConvBwdData
SIMPLE_CONVBWDDATA_EXE = ./tools/test_simpleConvBwdData
convbwddata:
	rm -rf $(CUDNN_CONVBWDDATA_EXE) $(SIMPLE_CONVBWDDATA_EXE)
	nvcc $(CUDNN_CFLAGS) -o $(CUDNN_CONVBWDDATA_EXE) \
	    ./assister/def.c ./cudnnConvolution/cudnnConvBwdData.c ./cudnnConvolution/test_cudnnConvBwdData.c \
	    -lcuda -lcudnn
	gcc $(SIMPLE_CFLAGS) -o $(SIMPLE_CONVBWDDATA_EXE) \
	    ./assister/def.c ./simpleConvolution/simpleConvBwdData.c ./simpleConvolution/test_simpleConvBwdData.c
	@echo "Begin to run cudnn|simple convolution backward for data ......"
	export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(CUDNN_CONVBWDDATA_EXE)
	./$(SIMPLE_CONVBWDDATA_EXE)

# convolution backward for filter
CUDNN_CONVBWDFILTER_EXE = ./tools/test_cudnnConvBwdFilter
SIMPLE_CONVBWDFILTER_EXE = ./tools/test_simpleConvBwdFilter
convbwdfilter:
	rm -rf $(CUDNN_CONVBWDFILTER_EXE) $(SIMPLE_CONVBWDFILTER_EXE)
	nvcc $(CUDNN_CFLAGS) -o $(CUDNN_CONVBWDFILTER_EXE) \
	    ./assister/def.c ./cudnnConvolution/cudnnConvBwdFilter.c ./cudnnConvolution/test_cudnnConvBwdFilter.c \
	    -lcuda -lcudnn
	gcc $(SIMPLE_CFLAGS) -o $(SIMPLE_CONVBWDFILTER_EXE) \
	    ./assister/def.c ./simpleConvolution/simpleConvBwdFilter.c ./simpleConvolution/test_simpleConvBwdFilter.c
	@echo "Begin to run cudnn|simple convolution forward ......"
	export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(CUDNN_CONVBWDFILTER_EXE)
	./$(SIMPLE_CONVBWDFILTER_EXE)

# convolution backward for bias
CUDNN_CONVBWDBIAS_EXE = ./tools/test_cudnnConvBwdBias
SIMPLE_CONVBWDBIAS_EXE = ./tools/test_simpleConvBwdBias
convbwdbias:
	rm -rf $(CUDNN_CONVBWDBIAS_EXE) $(SIMPLE_CONVBWDBIAS_EXE)
	nvcc $(CUDNN_CFLAGS) -o $(CUDNN_CONVBWDBIAS_EXE) \
	    ./assister/def.c ./cudnnConvolution/cudnnConvBwdBias.c ./cudnnConvolution/test_cudnnConvBwdBias.c \
	    -lcuda -lcudnn
	gcc $(SIMPLE_CFLAGS) -o $(SIMPLE_CONVBWDBIAS_EXE) \
	    ./assister/def.c ./simpleConvolution/simpleConvBwdBias.c ./simpleConvolution/test_simpleConvBwdBias.c
	@echo "Begin to run cudnn|simple convolution forward ......"
	export CUDA_VISIBLE_DEVICES=$(GPUID); ./$(CUDNN_CONVBWDBIAS_EXE)
	./$(SIMPLE_CONVBWDBIAS_EXE)

COMPARE_EXE = ./tools/compareFileData
compare:
	rm -rf $(COMPARE_EXE)
	gcc $(SIMPLE_CFLAGS) -o $(COMPARE_EXE) ./assister/compareFileData.c
	echo "Begin to run Compare data ......"
	./$(COMPARE_EXE) $(FILE)

.PHONY: clean
clean:
	rm -rf $(CUDNN_CONVFWD_EXE) $(SIMPLE_CONVFWD_EXE)
	rm -rf $(CUDNN_CONVBWDBIAS_EXE) $(SIMPLE_CONVBWDBIAS_EXE)
	rm -rf $(CUDNN_CONVBWDFILTER_EXE) $(SIMPLE_CONVBWDFILTER_EXE)
	rm -rf $(CUDNN_CONVBWDDATA_EXE) $(SIMPLE_CONVBWDDATA_EXE)
