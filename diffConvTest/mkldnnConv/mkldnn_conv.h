#ifndef _MKLDNN_CONV_H_
#define _MKLDNN_CONV_H_

#include "def.h"

#define MKLDNN_CHECK(flag) do { \
    mkldnn_status_t status = flag; \
    if (status != mkldnn_success) { \
        CODE_MESSAGE(); \
        exit(1); \
    } \
} while(false)

#define CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        CODE_MESSAGE(); \
        exit(1); \
    } \
} while(false)

#endif
