#include <stdio.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <string>
#include "/public3/home/xqq/anaconda3/envs/conda_xqq/include/mkldnn.hpp"
#include "/public3/home/xqq/cnnBench_intel/kernels/layer_kernels/conv_kernels.h"

#define FWD_CONVOLUTION   0
#define BWD_F_CONVOLUTION 1
#define BWD_D_CONVOLUTION 2

#define PREC_F32       0
#define PREC_U8S8U8    1
#define PREC_S16S16S32 2

#define TRAINING 0
#define INFERENCE_SERVER 1
#define INFERENCE_DEVICE 2

#define ITERS 1000

using namespace mkldnn;		// 可以直接这么用 mkldnn 的名字空间吗？？？

struct conv_para
{
	int n;		// minibatch
	int iw;		// input width
	int ih;		// input height
	int ic;		// input channels
	int oc; 	// output channels
	int fw; 	// filter width
	int fh; 	// filter height
	int sw; 	// stride width
	int sh; 	// stride height
	int pw; 	// padding width
	int ph; 	// padding height
	int iters;	// loop times
};

static inline int get_out_dim(int in_dim, int filter_dim, int pad_dim, int stride_dim)
{
	return (in_dim - filter_dim + 2 * pad_dim) / stride_dim + 1;
}

static double get_ops(const conv_para &para, bool calc_padding = true)
{
	double ops;
	int OW = get_out_dim(para.iw, para.fw, para.pw, para.sw);
	int OH = get_out_dim(para.ih, para.fh, para.ph, para.sh);

	if(calc_padding)	// calc_padding 仅在用 0 填充且可以忽略 padding 部分计算的情况下为 false
	{
		double multi_ops_once = 1.0 * para.fw * para.fh;	// 在单个 ic 上做一次矩阵乘加的乘法操作数
		double add_ops_once = multi_ops_once - 1;		// 在单个 ic 上做一次矩阵乘加的加法操作数
		double ops_once = multi_ops_once + add_ops_once;	// 在单个 ic 上做一次矩阵乘加的乘法和加法操作总次数
		ops = (ops_once * para.ic + para.ic) * para.oc * OW * OH * para.n;
	}
	else
	{
		int x = 0;
		int y = 0;
		ops = 0;
		
		for(int oh = 0; oh < OH; ++oh)
		{
			for(int ow = 0; ow < OW; ++ow)
			{
				int valid_w, valid_h;
				int x_right = x + para.fw - 1;
				int y_down = y + para.fh - 1;
				
				if(x_right < para.pw)
					valid_w = 0;
				else if(x < para.pw && x_right < para.pw + para.iw)
					valid_w = x_right - para.pw + 1;
				else if(x < para.pw && x_right >= para.pw + para.iw)
					valid_w = para.iw;
				else if(x >= para.pw && x_right < para.pw + para.iw)
					valid_w = x_right - x + 1;
				else if(x >= para.pw && x < para.pw + para.iw && x_right >= para.pw + para.iw)
					valid_w = para.pw + para.iw - x;
				else
					valid_w = 0;

				if(y_down < para.ph)
					valid_h = 0;
				else if(y < para.ph && y_down < para.ph + para.ih)
					valid_h = y_down - para.ph + 1;
				else if(y < para.ph && y_down >= para.ph + para.ih)
					valid_h = para.ih;
				else if(y >= para.ph && y_down < para.ph + para.ih)
					valid_h = y_down - y + 1;
				else if(y >= para.ph && y < para.ph + para.ih && y_down >= para.ph + para.ih)
					valid_h = para.ph + para.ih - y;
				else
					valid_h = 0;

				if(valid_w != 0 && valid_h != 0)
				{
					int ops_add = ((valid_w * valid_h * 2 - 1) * para.ic + para.ic) * para.oc;
					ops += ops_add;
				}
				x += para.sw;
			}
			x = 0;
			y += para.sh;
		}
		ops *= para.n;
	}
	return ops;
}

struct conv_performance
{
	double min_ms;
	double max_gflops;
	double avg_ms;
	double avg_gflops;
};

static inline double ms_timer()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1.e3 + tv.tv_usec * 1.e-3;
}

template <typename Func>
static inline conv_performance get_performance(int max_iters, double ops, Func func)
{
	func();
	conv_performance performance = {DBL_MAX, 0, 0, 0};
	for(int iters_done = 0; iters_done < max_iters; ++iters_done)
	{
		double ms = ms_timer();
		func();
		ms = ms_timer() - ms;
		if(ms < performance.min_ms)
			performance.min_ms = ms;
		performance.avg_ms += ms;
	}
	performance.max_gflops = ops / performance.min_ms * 1e-6;
	performance.avg_ms /= max_iters;
	performance.avg_gflops = ops / performance.avg_ms * 1e-6;
	return performance;
}

template <typename T>
static inline void rand_fill(T *data, size_t size)
{
	static bool initialized = false;
	if(!initialized)
	{
		srand48(1);
		initialized = true;
	}
	for(size_t i = 0; i < size / sizeof(T); ++i)
		data[i] = static_cast<T>(drand48());
}

static conv_performance conv_bench(conv_para para, int mode, int precision, bool calc_padding = true)
{
	engine eng(engine::kind::cpu, 0);
	int groups = 1;
	memory::data_type src_dt, dst_dt, filter_dt, bias_dt;
	switch(precision)
	{
		case PREC_U8S8U8:
			src_dt = memory::data_type::u8;
			dst_dt = memory::data_type::u8;
			filter_dt = memory::data_type::s8;
			bias_dt = memory::data_type::s32;
			break;
		case PREC_S16S16S32:
			src_dt = memory::data_type::s16;
			dst_dt = memory::data_type::s32;
			filter_dt = memory::data_type::s16;
			bias_dt = memory::data_type::s32;
			break;
		default:
			src_dt = memory::data_type::f32;
			dst_dt = memory::data_type::f32;
			filter_dt = memory::data_type::f32;
			bias_dt = memory::data_type::f32;
	}
	
	int OW = get_out_dim(para.iw, para.fw, para.pw, para.sw);
	int OH = get_out_dim(para.ih, para.fh, para.ph, para.sh);
	memory::desc src_d({para.n, para.ic, para.ih, para.iw}, src_dt, memory::format::any);
	memory::desc dst_d({para.n, para.oc, OH, OW}, dst_dt, memory::format::any);
	std::vector<int> fsizes = {para.oc / groups, para.ic / groups, para.fh, para.fw};
	if(groups != 1)
		fsizes.insert(fsizes.begin(), groups);
	memory::desc filter_d(fsizes, filter_dt, memory::format::any);
	memory::desc bias_d({para.oc}, bias_dt, memory::format::any);
	memory::dims strides = {para.sh, para.sw};
	memory::dims padding = {para.ph, para.pw};

	std::shared_ptr<primitive> conv;
	std::shared_ptr<memory> src;
	std::shared_ptr<memory> dst;
	std::shared_ptr<memory> filter;
	std::shared_ptr<memory> bias;

	auto fwd_conv_pd = convolution_forward::primitive_desc(
			   {prop_kind::forward_training, algorithm::convolution_direct, src_d, 
			   filter_d, bias_d, dst_d, strides, padding, padding, padding_kind::zero}, eng);

	if(mode == FWD_CONVOLUTION) 
	{
		src.reset(new memory(fwd_conv_pd.src_primitive_desc()));
		dst.reset(new memory(fwd_conv_pd.dst_primitive_desc()));
		filter.reset(new memory(fwd_conv_pd.weights_primitive_desc()));
		bias.reset(new memory(fwd_conv_pd.bias_primitive_desc()));
		conv.reset(new convolution_forward(fwd_conv_pd, *src, *filter, *bias, *dst));
	} 
	else if(mode == BWD_D_CONVOLUTION) {
        auto bwd_d_conv_pd = convolution_backward_data::primitive_desc(
			     {algorithm::convolution_direct, src_d, filter_d, dst_d, strides, 
			     padding, padding, padding_kind::zero}, eng, fwd_conv_pd);
        src.reset(new memory(bwd_d_conv_pd.diff_src_primitive_desc()));
        dst.reset(new memory(bwd_d_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_d_conv_pd.weights_primitive_desc()));
        conv.reset(new convolution_backward_data(bwd_d_conv_pd, *dst, *filter, *src));
    } 
	else if(mode == BWD_F_CONVOLUTION) {
        auto bwd_f_conv_pd = convolution_backward_weights::primitive_desc(
			     {algorithm::convolution_direct, src_d, filter_d,
#if COMPUTE_BWD_BIAS
                	     bias_d,
#endif
                	     dst_d, strides, padding, padding, padding_kind::zero}, eng, fwd_conv_pd);
        src.reset(new memory(bwd_f_conv_pd.src_primitive_desc()));
        dst.reset(new memory(bwd_f_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_f_conv_pd.diff_weights_primitive_desc()));
#if COMPUTE_BWD_BIAS
        bias.reset(new memory(bwd_f_conv_pd.diff_bias_primitive_desc()));
        conv.reset(new convolution_backward_weights(bwd_f_conv_pd, *src, *dst, *filter, *bias));
#else
	conv.reset(new convolution_backward_weights(bwd_f_conv_pd, *src, *dst, *filter));
#endif
	} 
	else
		throw std::runtime_error("Invalid benchmarking mode");

	for (const auto &m : {src, dst, filter, bias}) {
		if (!m.get() || !m->get())
			continue;
		void *data = m->get_data_handle();
		auto pd = m->get_primitive_desc();
		size_t size = pd.get_size();
		switch (pd.desc().data.data_type) {
			case memory::data_type::f32:
				rand_fill(static_cast<float *>(data), size);
				break;
			case memory::data_type::u8:
				rand_fill(static_cast<uint8_t *>(data), size);
				break;
			case memory::data_type::s8:
				rand_fill(static_cast<int8_t *>(data), size);
				break;
			case memory::data_type::s16:
				rand_fill(static_cast<int16_t *>(data), size);
				break;
			case memory::data_type::s32:
				rand_fill(static_cast<int32_t *>(data), size);
				break;
			default:
				assert(!"Unsupported data type!\n");
		}
	}
	stream str(stream::kind::eager);
	str.submit({*conv}).wait();
	return get_performance(para.iters, get_ops(para, calc_padding), [&](){str.rerun().wait();});

}
static void usage()
{
	printf(
			"Usage: <executable> [OPTIONS]\n"
			"\n"
			"Output control:\n"
			"   --csv-output        Produce CSV output\n"
			"   --original-output   Produce output in the original format\n"
			"\n"
			"Control flops calculations:\n"
			"   --no-skip-padding   Count ops with padding zeroes (default)\n"
			"   --skip-padding      Do not count ops with padding zeroes\n"
			"\n"
			"Precision control:\n"
			"   --f32               32-bit floating point (default)\n"
			"   --u8s8u8            8-bit integers (AVX512VL CPUs)\n"
			"   --s16s16s32         16-bit integers with 32-bit output\n"
			"                       (AVX512_4VNNI CPUs)\n"
			"Problem set control:\n"
			"   --training          Training data set (default)\n"
			"   --inference-server  Server inference data set\n"
			"   --inference-device  Device inference data set\n"
			"\n"
			);
	exit(-1);
}
int main(int argc, char **argv)
{
	bool calc_padding = true;
	bool csv_output = false;
	int precision = PREC_F32;
	std::vector<int> modes = {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION};
	int problem_set = TRAINING;
	for(argc--, argv++; argc; argv++, argc--) {
		if (*argv == std::string("--csv-output"))
			csv_output = true;
		else if (*argv == std::string("--original-output"))
			csv_output = false;
		else if (*argv == std::string("--skip-padding"))
			skip_padding = true;
		else if (*argv == std::string("--no-skip-padding"))
			skip_padding = false;
		else if (*argv == std::string("--f32"))
			precision = PREC_F32;
		else if (*argv == std::string("--u8s8u8"))
			precision = PREC_U8S8U8;
		else if (*argv == std::string("--s16s16s32"))
			precision = PREC_S16S16S32;
		else if (*argv == std::string("--inference-device"))
			problem_set = INFERENCE_DEVICE;
		else if (*argv == std::string("--inference-server"))
			problem_set = INFERENCE_SERVER;
		else if (*argv == std::string("--training"))
			problem_set = TRAINING;
		else
			usage();
	}
	if (precision != PREC_F32 || problem_set != TRAINING)
		modes = {FWD_CONVOLUTION};
	const char *conv_mode_strs[] = {"FWD", "BWD_F", "BWD_D"};
	const char *skip_padding_strs[]
		= {"w/ padding in flops", "w/o padding in flops"};
	const auto &problems = (problem_set == TRAINING
			? training_set
			: (problem_set == INFERENCE_DEVICE
				? inference_device_set
				: inference_server_set));
	for (auto m : modes) {
		if (!csv_output)
			printf(" %s Convolution\n", conv_mode_strs[m]);
		for (const auto& problem : problems) {
			conv_para p;
			std::tie(p.iw, p.ih, p.ic, p.n, p.oc, p.fw, p.fh,
					p.pw, p.ph, p.sw, p.sh) = problem;
			p.iters = ITERS;
			auto r = bench_conv(p, m, precision, skip_padding);
			if (csv_output)
				printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e,%e,%e,%e\n",
						conv_mode_strs[m], skip_padding,
						p.minibatch, p.w, p.h, p.ic, p.oc, p.fw, p.fh,
						p.stride_w, p.stride_h, p.pad_w, p.pad_h,
						r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
			else
				printf("W=%d, H=%d, C=%d, N=%d, K=%d, S=%d, R=%d | "
						"%s %s min(ms) %.2f; max(gflop/s) %.2f;"
						"avg(ms) %.2f; avg(gflop/s) %.2f;\n",
						p.w, p.h, p.ic, p.minibatch, p.oc, p.fw, p.fh,
						conv_mode_strs[m], skip_padding_strs[skip_padding],
						r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
			fflush(0);
		}
	}
	return 0;
}
