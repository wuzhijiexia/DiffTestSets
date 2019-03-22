#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include "mkldnn.h"
#include "mkldnn_conv.h"

static size_t product(mkldnn_dim_t *arr, size_t size) {
    size_t i;
    size_t prod = 1;
    for (i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

void mkldnnConvolutionTest(convMess* pcm)
{
    mkldnn_dim_t i, j, k;

    // Convolution Configure
    mkldnn_dim_t N, C, K, inH, inW;
    mkldnn_dim_t R, S;
    mkldnn_dim_t padh, padw, strideh, stridew;
    mkldnn_dim_t outH, outW;
    mkldnn_dim_t insize, fltsize, outsize;

    N = pcm->N_;
    C = pcm->C_;
    K = pcm->K_;
    inH = pcm->inH_;
    inW = pcm->inW_;
    outH = pcm->outH_;
    outW = pcm->outW_;
    R = pcm->R_;
    S = pcm->S_;
    padh = pcm->padh_;
    padw = pcm->padw_;
    strideh = pcm->strideh_;
    stridew = pcm->stridew_;
    
    insize = N*C*inH*inW;
    fltsize = K*C*R*S;
    outsize = N*K*outH*outW;


    const mkldnn_dim_t groups = 1;
    mkldnn_dim_t c3_src_sizes[4] = {N, C, inH, inW};
    mkldnn_dim_t c3_weights_sizes[] = {K, C, R, S};
    //mkldnn_dim_t c3_weights_sizes[] = {groups, K/groups, C/groups, R, S};
    mkldnn_dim_t c3_bias_sizes[1] = {K};
    mkldnn_dim_t strides[] = {strideh, stridew};
    mkldnn_dim_t padding[] = {padh, padw}; // set proper values
    mkldnn_dim_t c3_dst_sizes[4] = {N, K, outH, outW};

    DataType *src = (DataType*)calloc(product(c3_src_sizes, 4), sizeof(DataType));
    DataType *weights = (DataType*)calloc(product(c3_weights_sizes, 4), sizeof(DataType));
    //DataType *weights = (DataType*)calloc(product(c3_weights_sizes, 5), sizeof(DataType));
    DataType *bias = (DataType*)calloc(product(c3_bias_sizes, 1), sizeof(DataType));
    DataType *dst = (DataType*)calloc(product(c3_dst_sizes, 4), sizeof(DataType));
    DataType *out_mem = (DataType*)calloc(product(c3_dst_sizes, 4), sizeof(DataType));
    CHECK_TRUE(src && weights && bias && dst && out_mem);

    CODE_MESSAGE();
    for (i = 0; i < c3_bias_sizes[0]; ++i) bias[i] = 0;
    for(i = 0; i < insize; i++) src[i] = 0.1*(rand()%4);
    for(i = 0; i < fltsize; i++) weights[i] = 0.1*(rand()%5);
    
    CODE_MESSAGE();
    mkldnn_engine_t engine;
    MKLDNN_CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0));

    mkldnn_stream_t stream;
    MKLDNN_CHECK(mkldnn_stream_create(&stream, engine, mkldnn_stream_default_flags));

    /* first describe user data and create data descriptors for future
     * convolution w/ the specified format -- we do not want to do a reorder */
    mkldnn_memory_desc_t c3_src_md, c3_weights_md, c3_bias_md, c3_dst_md, out_md;
    mkldnn_memory_t c3_src, c3_weights, c3_bias, c3_dst, out;

    // src
    {
        MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&c3_src_md, 4, c3_src_sizes, mkldnn_f32, mkldnn_nChw8c));
        MKLDNN_CHECK(mkldnn_memory_create(&c3_src, &c3_src_md, engine, src));
    }

    CODE_MESSAGE();
    // weights
    {
        MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&c3_weights_md, 4,
                    c3_weights_sizes, mkldnn_f32,
                    mkldnn_OIhw8i8o));
        //MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&c3_weights_md, 4 + (groups != 1),
          //          c3_weights_sizes + (groups == 1), mkldnn_f32,
            //        groups == 1 ? mkldnn_OIhw8i8o : mkldnn_gOIhw8i8o));
        MKLDNN_CHECK(mkldnn_memory_create(&c3_weights, &c3_weights_md, engine, weights));
    }

    CODE_MESSAGE();
    // bias
    {
        MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&c3_bias_md, 1, c3_bias_sizes, mkldnn_f32, mkldnn_x));
        MKLDNN_CHECK(mkldnn_memory_create(&c3_bias, &c3_bias_md, engine, bias));
    }

    // c3_dst
    {
        MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&c3_dst_md, 4, c3_dst_sizes, mkldnn_f32, mkldnn_nChw8c));
        MKLDNN_CHECK(mkldnn_memory_create(&c3_dst, &c3_dst_md, engine, dst));
    }

    // out
    {
        MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(&out_md, 4, c3_dst_sizes, mkldnn_f32, mkldnn_nchw));
        MKLDNN_CHECK(mkldnn_memory_create(&out, &out_md, engine, out_mem));
    }

    /* create a convolution primitive descriptor */
    mkldnn_convolution_desc_t c3_desc;
    mkldnn_primitive_desc_t c3_pd;
    mkldnn_primitive_t c3;

    MKLDNN_CHECK(mkldnn_convolution_forward_desc_init(&c3_desc,
                mkldnn_forward_inference, mkldnn_convolution_direct,
                &c3_src_md, &c3_weights_md, &c3_bias_md, &c3_dst_md,
                strides, padding, NULL, mkldnn_padding_zero));
    MKLDNN_CHECK(mkldnn_primitive_desc_create(&c3_pd, &c3_desc, NULL, engine, NULL));

    CHECK_TRUE(mkldnn_memory_desc_equal(&c3_src_md,
                mkldnn_primitive_desc_query_md(c3_pd, mkldnn_query_src_md, 0)));
    CHECK_TRUE(mkldnn_memory_desc_equal(&c3_weights_md,
                mkldnn_primitive_desc_query_md(c3_pd, mkldnn_query_weights_md, 0)));
    CHECK_TRUE(mkldnn_memory_desc_equal(&c3_bias_md,
                mkldnn_primitive_desc_query_md(c3_pd, mkldnn_query_weights_md, 1)));
    CHECK_TRUE(mkldnn_memory_desc_equal(&c3_dst_md,
                mkldnn_primitive_desc_query_md(c3_pd, mkldnn_query_dst_md, 0)));

    /* create a convolution and execute it */
    MKLDNN_CHECK(mkldnn_primitive_create(&c3, c3_pd));
    MKLDNN_CHECK(mkldnn_primitive_desc_destroy(c3_pd));

    mkldnn_exec_arg_t c3_args[4] = {
        {MKLDNN_ARG_SRC, c3_src},
        {MKLDNN_ARG_WEIGHTS, c3_weights},
        {MKLDNN_ARG_BIAS, c3_bias},
        {MKLDNN_ARG_DST, c3_dst},
    };
    MKLDNN_CHECK(mkldnn_primitive_execute(c3, stream, 4, c3_args));
#ifdef VERITY_RESULT
    verityConvolutionResult(pcm, src, weights, dst);
#endif
    MKLDNN_CHECK(mkldnn_primitive_destroy(c3));

    /* create a reorder primitive descriptor */
    mkldnn_primitive_desc_t r_pd;
    MKLDNN_CHECK(mkldnn_reorder_primitive_desc_create(
                &r_pd, engine, &c3_dst_md, engine, &out_md, NULL));

    /* create a reorder and execute it */
    mkldnn_primitive_t r;
    MKLDNN_CHECK(mkldnn_primitive_create(&r, r_pd));
    MKLDNN_CHECK(mkldnn_primitive_desc_destroy(r_pd));

    mkldnn_exec_arg_t r_args[2] = {
        {MKLDNN_ARG_FROM, c3_dst},
        {MKLDNN_ARG_TO, out},
    };
    MKLDNN_CHECK(mkldnn_primitive_execute(r, stream, 2, r_args));
    MKLDNN_CHECK(mkldnn_primitive_destroy(r));

    /* clean-up */
    MKLDNN_CHECK(mkldnn_memory_destroy(c3_src));
    MKLDNN_CHECK(mkldnn_memory_destroy(c3_weights));
    MKLDNN_CHECK(mkldnn_memory_destroy(c3_bias));
    MKLDNN_CHECK(mkldnn_memory_destroy(c3_dst));
    MKLDNN_CHECK(mkldnn_memory_destroy(out));
    MKLDNN_CHECK(mkldnn_stream_destroy(stream));
    MKLDNN_CHECK(mkldnn_engine_destroy(engine));

#if 0
    const mkldnn_dim_t N = c3_dst_sizes[0], C = c3_dst_sizes[1],
          H = c3_dst_sizes[2], W = c3_dst_sizes[3];
    for (mkldnn_dim_t n = 0; n < N; ++n)
        for (mkldnn_dim_t c = 0; c < C; ++c)
            for (mkldnn_dim_t h = 0; h < H; ++h)
                for (mkldnn_dim_t w = 0; w < W; ++w)
                {
                    mkldnn_dim_t off = ((n*C + c)*H + h)*W + w;
                    CHECK_TRUE(out_mem[off] == bias[c]);
                }
#endif

    CPUFREE(src);
    CPUFREE(weights);
    CPUFREE(bias);
    CPUFREE(dst);
    CPUFREE(out_mem);
}
