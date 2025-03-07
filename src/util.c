#include "attention.h"

void func_vdot(int n, const data_t *x, const data_t *y, data_t *out) {
    data_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *out = sum;
}

void get_col(int h, int w, const data_t* data, int col, data_t* out) {
    if (col < 0 || col >= w) {
        fprintf(stderr, "Column index out of bounds\n");
        exit(-1);
    }
    for (int i = 0; i < h; i++) {
        out[i] = data[i * w + col];
    }
}

data_t get_max(int n, const data_t* in){
    data_t max = in[0];
    for(int i = 1; i < n; i++){
        if(in[i] > max){
            max = in[i];
        }
    }
    return max;
}

void gelu_forward(data_t* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        data_t x = inp[i];
        data_t cube = 0.044715f * x * x * x;
        inp[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}


void layernorm_forward(data_t* out, data_t* mean, data_t* rstd,
    const data_t* inp, 
    const data_t* gamma, const data_t* beta, // they are 1 x d for each layer
    int N, int d
) {
    data_t eps = 1e-5f;
    for (int t = 0; t < N; t++) {
        data_t* x = inp + t * d;

        // calculate the mean
        data_t m = 0.0f;
        for (int i = 0; i < d; i++) {
            m += x[i];
        }
        m /= d;

        // calculate the variance (without any bias correction)
        data_t v = 0.0f;
        for (int i = 0; i < d; i++) {
        data_t xshift = x[i] - m;
        v += xshift * xshift;
        }
        v /= d;

        // calculate the rstd (reciprocal standard deviation)
        data_t s = 1.0f / sqrtf(v + eps);

        // seek to the output position in out[b,t,:]
        data_t* out_bt = out + t * d;
        for (int i = 0; i < d; i++) {
            data_t n = (s * (x[i] - m)); // normalize
            data_t o = n * gamma[i] + beta[i]; // scale and shift
            out_bt[i] = o; // write
        }

        // cache the mean and rstd for the backward pass later
        mean[t] = m;
        rstd[t] = s;
    }
}

void func_embedding(
    data_t* Inp,
    int N, int d, int embedding
){
    data_t* Input = (data_t*)malloc(N * d * sizeof(data_t));
    if (Input == NULL) {
        fprintf(stderr, "Memory allocation failed for Input\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < N * d; i++){
        Input[i] = (data_t)rand() / RAND_MAX;
    }

    data_t* W_embedding = (data_t*)malloc(d * embedding * sizeof(data_t));
    if (W_embedding == NULL) {
        fprintf(stderr, "Memory allocation failed for W_embedding\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < d * embedding; i++){
        W_embedding[i] = (data_t)rand() / RAND_MAX;
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    for(int i = 0; i < N; i++){
        for(int j = 0; j < d; j++){
            func_vdot(embedding, &W_embedding[j * embedding], &Input[i * embedding], &Inp[i * d + j]);
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\tTime taken to compute word embedding: %f seconds\n", cpu_time_used);

    return;
}