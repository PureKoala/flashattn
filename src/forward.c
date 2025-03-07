#include "attention.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int num = atoi(argv[1]);
    if (num <= 0) {
        fprintf(stderr, "Invalid value for num: %d\n", num);
        return EXIT_FAILURE;
    }


    int d = 1024;
    int embedding = 768;
    int layernum = 6;
    int decoder_num = 6;


    // for (int s = 0; s < num; s++) {
        int N = 64 * num;

        data_t *WQ = (data_t*)malloc(layernum * d * d * sizeof(data_t));
        data_t *WK = (data_t*)malloc(layernum * d * d * sizeof(data_t));
        data_t *WV = (data_t*)malloc(layernum * d * d * sizeof(data_t));

        data_t *W1 = (data_t*)malloc(layernum * d * 4 * d * sizeof(data_t));
        data_t *W2 = (data_t*)malloc(layernum * d * 4 * d * sizeof(data_t));

        data_t *gamma = (data_t*)malloc(layernum * 2 * d * sizeof(data_t));
        data_t *beta  = (data_t*)malloc(layernum * 2 * d * sizeof(data_t));

        data_t *Input = (data_t*)malloc(N * d * sizeof(data_t));

        // data_t *Q_cache = (data_t*)malloc(N * d * sizeof(data_t));
        data_t *K_cache = (data_t*)malloc(layernum * (N + decoder_num) * d * sizeof(data_t));
        data_t *V_cache = (data_t*)malloc(layernum * (N + decoder_num) * d * sizeof(data_t));
        data_t *O  = (data_t*)malloc(N * d * sizeof(data_t));

        if (WQ == NULL || WK == NULL || WV == NULL || W1 == NULL || W2 == NULL || gamma == NULL || beta == NULL) {
            fprintf(stderr, "Weight memory init allocation failed\n");
            exit(EXIT_FAILURE);
        }

        if (K_cache == NULL || V_cache == NULL || O == NULL) {
            fprintf(stderr, "KV Cache memory init allocation failed\n");
            exit(EXIT_FAILURE);
        }

        if (Input == NULL) {
            fprintf(stderr, "Input memory init allocation failed\n");
            exit(EXIT_FAILURE);
        }

        // random init weights
        for(int i = 0; i < layernum * d * d; i++){
            WQ[i] = (data_t)rand() / RAND_MAX;
            WK[i] = (data_t)rand() / RAND_MAX;
            WV[i] = (data_t)rand() / RAND_MAX;
        }

        for(int i = 0; i < layernum * d * 4 * d; i++){
            W1[i] = (data_t)rand() / RAND_MAX;
            W2[i] = (data_t)rand() / RAND_MAX;
        }

        for(int i = 0; i < layernum * 2 * d; i++){
            gamma[i] = (data_t)rand() / RAND_MAX;
            beta[i] = (data_t)rand() / RAND_MAX;
        }

        // word embedding
        func_embedding(Input, N, d, embedding);

        printf("================ Begin to prefill N = %d, d = %d ================\n", N, d);

        // prefill model
        for(int layer = 0; layer < layernum; layer++){
            printf("======== Prefill layer %d ========\n", layer);
            forward_layer_prefill(
                WQ + layer * d * d, WK + layer * d * d, WV + layer * d * d, 
                W1 + layer * d * 4 * d, W2 + layer * d * 4 * d,
                gamma + layer * 2 * d, beta + layer * 2 * d,
                Input, N, d,
                K_cache + layer * (N + decoder_num) * d, V_cache + layer * (N + decoder_num) * d,
                O
            );

            memcpy(Input, O, N * d * sizeof(data_t));
        }

        data_t* decoder_input  = (data_t*)malloc(1 * d * sizeof(data_t));
        data_t* decoder_output = (data_t*)malloc(1 * d * sizeof(data_t));

        if (decoder_input == NULL || decoder_output == NULL) {
            fprintf(stderr, "Decoder memory init allocation failed\n");
            exit(EXIT_FAILURE);
        }

        printf("================ Begin to decoding ================\n");

        // decoder rounds
        for (int n = 0; n < decoder_num; n++){
            printf("======== Decoding round %d ========\n", n);
            func_embedding(decoder_input, 1, d, embedding);
            for(int layer = 0; layer < layernum; layer++){

                printf("======== Decoding layer %d ========\n", layer);

                forward_layer_decoding(
                    WQ + layer * d * d, WK + layer * d * d, WV + layer * d * d, 
                    W1 + layer * d * 4 * d, W2 + layer * d * 4 * d,
                    gamma + layer * 2 * d, beta + layer * 2 * d,
                    decoder_input, N + n, d,
                    K_cache + layer * (N + decoder_num) * d, V_cache + layer * (N + decoder_num) * d,
                    decoder_output
                );

                memcpy(decoder_input, decoder_output, 1 * d * sizeof(data_t));
            }
        }

        free(WQ);
        free(WK);
        free(WV);
        free(W1);
        free(W2);
        free(gamma);
        free(beta);
        free(Input);
        free(K_cache);
        free(V_cache);
        free(O);
        free(decoder_input);
        free(decoder_output);    

    // }

}