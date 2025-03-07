#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define data_t float
#define EPSILON 1e-5  // 避免除零错误
#define PI 3.141592653589793

#define GELU_SCALING_FACTOR sqrtf(2.0f / PI)

// #define ENABLE_CASUAL
#define ENABLE_PREFILL_PRINT
#define ENABLE_DECODING_PRINT

void func_vdot(int n, const data_t *x, const data_t *y, data_t *out);
void get_col(int h, int w, const data_t* data, int col, data_t* out);
data_t get_max(int n, const data_t* in);
void gelu_forward(float* inp, int N);
void func_embedding(data_t* Inp, int N, int d, int embedding);

void layernorm_forward(data_t* out, data_t* mean, data_t* rstd,
    const data_t* inp, 
    const data_t* gamma, const data_t* beta, // they are 1 x d for each layer
    int N, int d
);

int forward_layer_prefill(
    const data_t* WQ, const data_t* WK, const data_t* WV, // d x d
    const data_t* W1, const data_t* W2, // d x 4d
    const data_t* gamma, const data_t* beta, // 2 x d 
    const data_t* Input, // N x d
    const int N, const int d,
    data_t* K, data_t* V, // store to KV Cache
    data_t* O // N x d
);

int forward_layer_decoding(
    const data_t* WQ, const data_t* WK, const data_t* WV, // d x d
    const data_t* W1, const data_t* W2, // d x 4d
    const data_t* gamma, const data_t* beta, // 2 x d 
    const data_t* Input, // 1 x d
    const int N, const int d,
    data_t* K_cache, data_t* V_cache, // KV Cache
    data_t* O // 1 x d
);