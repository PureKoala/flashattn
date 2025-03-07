#include "attention.h"

// Decoder attention with block-wise processing
int flashattention_block_decoder(
    const data_t* Q,  // 1 x d (single query vector)
    const data_t* K, const data_t* V,  // N x d (cached keys and values)
    const int N, const int d, const int Bc,
    data_t* O  // d x 1 (output)
) {
    clock_t start, end;
    double cpu_time_used;
    
    data_t scale_factor = 1.0 / sqrt(d);
    
    // For softmax stability
    data_t m = -INFINITY;
    data_t l = 0.0f;
    
    // Temporary storage for softmax values
    data_t* P = (data_t*)malloc(N * sizeof(data_t));
    if (P == NULL) {
        fprintf(stderr, "Memory allocation failed for P\n");
        exit(EXIT_FAILURE);
    }
    
    // Zero initialize output
    for (int j = 0; j < d; j++) {
        O[j] = 0.0f;
    }
    
    // Begin processing
    start = clock();
    
    // First pass: compute QK and find max for softmax stability
    for (int j_block = 0; j_block < N; j_block += Bc) {
        int j_end = (j_block + Bc < N) ? j_block + Bc : N;
        
        // Compute QK for this block
        for (int j = j_block; j < j_end; j++) {
            const data_t* Kj = K + j * d;
            data_t qk_val;
            func_vdot(d, Q, Kj, &qk_val);
            qk_val *= scale_factor;
            P[j] = qk_val;
            
            // Track max value for softmax stability
            if (qk_val > m) {
                m = qk_val;
            }
        }
    }
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute QK: %f seconds\n", cpu_time_used);
#endif
    
    // Second pass: compute softmax denominators
    start = clock();
    for (int j = 0; j < N; j++) {
        l += exp(P[j] - m);
    }
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute softmax denominator: %f seconds\n", cpu_time_used);
#endif
    
    // Third pass: block-wise attention computation
    start = clock();
    for (int j_block = 0; j_block < N; j_block += Bc) {
        int j_end = (j_block + Bc < N) ? j_block + Bc : N;
        
        // Process this block of keys and values
        for (int j = j_block; j < j_end; j++) {
            // Compute normalized attention weight
            data_t p_j = exp(P[j] - m) / l;
            
            // Update output with weighted value
            for (int k = 0; k < d; k++) {
                O[k] += p_j * V[j * d + k];
            }
        }
    }
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
#ifdef ENABLE_DECODING_PRINT
    printf("\tTime taken to compute attention output: %f seconds\n", cpu_time_used);
#endif
    
    free(P);
    
    return 0;
}