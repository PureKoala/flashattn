#include "attention.h"

int flashattention_block_prefill(
    const data_t* Q, const data_t* K, const data_t* V,
    const int N, const int d, const int Br, const int Bc,
    data_t* O
) {
    clock_t start, end;
    double cpu_time_used;
    
    // Allocate memory for O_temp, m, l
    data_t *O_temp = (data_t*)calloc(N * d, sizeof(data_t));
    data_t *m = (data_t*)malloc(N * sizeof(data_t));
    data_t *l = (data_t*)malloc(N * sizeof(data_t));
    
    if (O_temp == NULL || m == NULL || l == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize m to -inf and l to 0
    for (int i = 0; i < N; i++) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    
    data_t scale_factor = 1.0 / sqrt(d);
    
    // Computing attention with blocking
    start = clock();
    
    // Process Q in blocks of Br rows
    for (int i_block = 0; i_block < N; i_block += Br) {
        int i_end = (i_block + Br < N) ? i_block + Br : N;
        
        // Process K,V in blocks of Bc columns
        for (int j_block = 0; j_block < N; j_block += Bc) {
            int j_end = (j_block + Bc < N) ? j_block + Bc : N;
            
            // Allocate temporary QK block
            data_t *QK_block = (data_t*)malloc((i_end - i_block) * (j_end - j_block) * sizeof(data_t));
            if (QK_block == NULL) {
                fprintf(stderr, "Memory allocation failed for QK_block\n");
                exit(EXIT_FAILURE);
            }
            
            // Compute QK for this block
            for (int i = i_block; i < i_end; i++) {
                const data_t* Qi = Q + i * d;
                
                for (int j = j_block; j < j_end; j++) {
                    #ifdef ENABLE_CASUAL
                    if (i >= j) {
                    #endif
                        const data_t* Kj = K + j * d;
                        data_t qk_val;
                        func_vdot(d, Qi, Kj, &qk_val);
                        qk_val *= scale_factor;
                        QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)] = qk_val;
                    #ifdef ENABLE_CASUAL
                    } else {
                        QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)] = -INFINITY;
                    }
                    #endif
                }
                
                // Find max in this block row for stable softmax
                for (int j = j_block; j < j_end; j++) {
                    data_t qk_val = QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)];
                    if (qk_val > m[i]) {
                        // Rescale sums when max changes
                        if (m[i] != -INFINITY) {
                            l[i] *= exp(m[i] - qk_val);
                        }
                        m[i] = qk_val;
                    }
                }
                
                // Compute exponentials and row sums
                for (int j = j_block; j < j_end; j++) {
                    data_t qk_val = QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)];
                    data_t exp_val = exp(qk_val - m[i]);
                    QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)] = exp_val;
                    l[i] += exp_val;
                }
                
                // Update output with this block
                for (int j = j_block; j < j_end; j++) {
                    data_t p_ij = QK_block[(i - i_block) * (j_end - j_block) + (j - j_block)] / l[i];
                    
                    // Update O with V values
                    for (int k = 0; k < d; k++) {
                        O_temp[i * d + k] += p_ij * V[j * d + k];
                    }
                }
            }
            
            free(QK_block);
        }
    }
    
    // Copy results to output
    memcpy(O, O_temp, N * d * sizeof(data_t));
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    #ifdef ENABLE_PREFILL_PRINT
    printf("\tTime taken for FlashAttention: %f seconds\n", cpu_time_used);
    #endif
    
    // Free allocated memory
    free(O_temp);
    free(m);
    free(l);
    
    return 0;
}