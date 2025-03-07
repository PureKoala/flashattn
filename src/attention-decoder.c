#include "attention.h"

// Q, K, V: N x d
int attention_block_decoder(
    const data_t* Q, // 1 x d
    const data_t* K, const data_t* V, // N x d
    const int N, const int d, 
    data_t* O // d x 1
) {
    clock_t start, end;
    double cpu_time_used;

    data_t *QK = (data_t*)malloc(N * sizeof(data_t)); // aka P, S
    if (QK == NULL) {
        fprintf(stderr, "Memory allocation failed for QK\n");
        exit(EXIT_FAILURE);
    }
    data_t scale_factor = 1.0 / sqrt(d);

    // compute QK
    start = clock();
    for(int k = 0; k < N; k++){
        const data_t* Kj = K + k * d;
        func_vdot(d, Q, Kj, QK + k);
        QK[k] = QK[k] * scale_factor;     
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute S: %f seconds\n", cpu_time_used);
#endif
    // compute S
    start = clock();
    data_t l = 0;
    data_t m = get_max(N, QK);

    for(int j = 0; j < N; j++){
        l += exp(QK[j] - m);
    }

    for (int j = 0; j < N; j++){
        QK[j] = exp(QK[j] - m) / l;
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute P: %f seconds\n", cpu_time_used);
#endif
    // compute O
    start = clock();
    for(int j = 0; j < d; j++){
        data_t *Vj = (data_t*)malloc(N * sizeof(data_t));
        if (Vj == NULL) {
            fprintf(stderr, "Memory allocation failed for Vj\n");
            exit(EXIT_FAILURE);
        }
        get_col(N, d, V, j, Vj);
        func_vdot(N, QK, Vj, O + j);
        free(Vj);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute O: %f seconds\n", cpu_time_used);
#endif
    free(QK);

    return 0;
}



// Q, K, V: 1 x d
int forward_layer_decoding(
    const data_t* WQ, const data_t* WK, const data_t* WV, // d x d
    const data_t* W1, const data_t* W2, // d x 4d
    const data_t* gamma, const data_t* beta, // 2 x d 
    const data_t* Input, // 1 x d
    const int N, const int d,
    data_t* K_cache, data_t* V_cache, // KV Cache
    data_t* O // 1 x d
){
    clock_t start, end;
    double cpu_time_used;

    // compute attention layer norm
    data_t* normed_attention = (data_t*)malloc(d * sizeof(data_t));
    data_t* mean = (data_t*)malloc(sizeof(data_t));
    data_t* rstd = (data_t*)malloc(sizeof(data_t));
    
    if (normed_attention == NULL || mean == NULL || rstd == NULL) {
        fprintf(stderr, "Memory allocation failed for pre-norm \n");
        exit(EXIT_FAILURE);
    }

    start = clock();
    layernorm_forward(normed_attention, mean, rstd, Input, gamma, beta, 1, d);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute attention pre layer norm: %f seconds\n", cpu_time_used);
#endif
    free(mean);
    free(rstd);

    // ============================================================================

    // compute Q from input
    data_t* Q = (data_t*)malloc(d * sizeof(data_t));

    start = clock();
    for(int j = 0; j < d; j++){
        func_vdot(d, &WQ[j * d], normed_attention, &Q[j]);
        func_vdot(d, &WK[j * d], normed_attention, &K_cache[N * d + j]);
        func_vdot(d, &WV[j * d], normed_attention, &V_cache[N * d + j]);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute Q, K, V: %f seconds\n", cpu_time_used);
#endif
    free(normed_attention);

    // ============================================================================

    // compute attention block
    data_t* attn_res = (data_t*)malloc(d * sizeof(data_t));
    if (attn_res == NULL) {
        fprintf(stderr, "Memory allocation failed for attention result\n");
        exit(EXIT_FAILURE);
    }

    start = clock();
    attention_block_decoder(Q, K_cache, V_cache, N, d, attn_res);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute attention block: %f seconds\n", cpu_time_used);
#endif
    free(Q);

    // ============================================================================

    // compute residual connection
    start = clock();
    for(int j = 0; j < d; j++){
        attn_res[j] = attn_res[j] + Input[j];
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute attention residual connection: %f seconds\n", cpu_time_used);
#endif
    // ============================================================================

    // compute MLP layer norm
    data_t* normed_mlp = (data_t*)malloc(d * sizeof(data_t));
    mean = (data_t*)malloc(sizeof(data_t));
    rstd = (data_t*)malloc(sizeof(data_t));
    
    if (normed_mlp == NULL || mean == NULL || rstd == NULL) {
        fprintf(stderr, "Memory allocation failed for pre-norm \n");
        exit(EXIT_FAILURE);
    }

    start = clock();
    layernorm_forward(normed_mlp, mean, rstd, attn_res, gamma + d, beta + d, 1, d);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute MLP pre layer norm: %f seconds\n", cpu_time_used);
#endif
    free(mean);
    free(rstd);

    // ============================================================================

    // compute MLP block
    data_t* H1 = (data_t*)malloc(4 * d * sizeof(data_t));
    data_t* H2 = (data_t*)malloc(d * sizeof(data_t));

    if (H1 == NULL || H2 == NULL) {
        fprintf(stderr, "Memory allocation failed for MLP block\n");
        exit(EXIT_FAILURE);
    }

    start = clock();
    for(int j = 0; j < 4 * d; j++){
        func_vdot(d, &W1[j * d], normed_mlp, &H1[j]);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute H1: %f seconds\n", cpu_time_used);
#endif
    start = clock();
    gelu_forward(H1, 4 * d);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute Gelu: %f seconds\n", cpu_time_used);
#endif
    start = clock();
    for(int j = 0; j < d; j++){
        func_vdot(4 * d, &W2[j * 4 * d], H1, &H2[j]);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute H2: %f seconds\n", cpu_time_used);
#endif
    // ============================================================================

    // compute MLP residual connection
    start = clock();
    for(int j = 0; j < d; j++){
        O[j] = H2[j] + attn_res[j];
    }
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute MLP residual connection: %f seconds\n", cpu_time_used);
#endif
    free(H1);
    free(H2);
    free(attn_res);
    free(normed_mlp);

    // ============================================================================

    return 0;
}



// int main(int argc, char *argv[]) {

//     int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192};
//     int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

//     for (int s = 0; s < num_sizes; s++) {
//         int N = sizes[s];
//         int d = 1024;
//         int embedding = 768;

//         data_t *WQ    = (data_t*)malloc(d * embedding * sizeof(data_t));
//         data_t *WK    = (data_t*)malloc(d * embedding * sizeof(data_t));
//         data_t *WV    = (data_t*)malloc(d * embedding * sizeof(data_t));
//         data_t *Input = (data_t*)malloc(1 * embedding * sizeof(data_t));

//         if (WQ == NULL || WK == NULL || WV == NULL || Input == NULL) {
//             fprintf(stderr, "Memory weight or input init allocation failed\n");
//             exit(EXIT_FAILURE);
//         }

//         // random init WQ, WK, WV
//         for(int i = 0; i < d * embedding; i++){
//             WQ[i] = (data_t)rand() / RAND_MAX;
//             WK[i] = (data_t)rand() / RAND_MAX;
//             WV[i] = (data_t)rand() / RAND_MAX;
//         }

//         data_t *Q_cache = (data_t*)malloc((N + 1) * d * sizeof(data_t));
//         data_t *K_cache = (data_t*)malloc((N + 1) * d * sizeof(data_t));
//         data_t *V_cache = (data_t*)malloc((N + 1) * d * sizeof(data_t));

//         if (Q_cache == NULL || K_cache == NULL || V_cache == NULL) {
//             fprintf(stderr, "Memory Q | K | V cache allocation failed\n");
//             exit(EXIT_FAILURE);
//         }

//         // random init Q, K, V cache
//         for(int i = 0; i < N * d; i++){
//             Q_cache[i] = (data_t)rand() / RAND_MAX;
//             K_cache[i] = (data_t)rand() / RAND_MAX;
//             V_cache[i] = (data_t)rand() / RAND_MAX;
//         }


//         printf("================ Q K V compute init N = %d, d = %d ================\n", N, d);

//         data_t *O = (data_t*)malloc(d * sizeof(data_t));

//         if (O == NULL) {
//             fprintf(stderr, "Memory O init allocation failed\n");
//             exit(EXIT_FAILURE);
//         }

//         // compute Q, K, V from input
//         clock_t start, end;
//         double cpu_time_used;

//         start = clock();
//         for(int j = 0; j < d; j++){
//             func_vdot(embedding, &WQ[j * embedding], Input, &Q_cache[N * d + j]);
//             func_vdot(embedding, &WK[j * embedding], Input, &K_cache[N * d + j]);
//             func_vdot(embedding, &WV[j * embedding], Input, &V_cache[N * d + j]);
//         }
//         end = clock();
//         cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//         printf("\tTime taken to compute Q, K, V: %f seconds\n", cpu_time_used);
    

//         printf("================ Attention block compute init ===============\n");

//         attention_block_decoder(Q_cache, K_cache, V_cache, N + 1, d, O);

//         printf("=============================================================\n\n\n");

//         free(O);
//         free(WQ);
//         free(WK);
//         free(WV);
//         free(Input);
//         free(Q_cache);
//         free(K_cache);
//         free(V_cache);
//     }

//     return 0;
// }