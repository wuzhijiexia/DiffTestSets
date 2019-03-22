export LD_LIBRARY_PATH=/home/xqq
export CPLUS_INCLUDE_PATH=/home/xqq
g++ -O3 -o test easy_mkldnn_conv_bench.cpp -L/home/xqq -lmkldnn -std=c++11 -lmklml_intel
