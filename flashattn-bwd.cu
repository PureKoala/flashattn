#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define data_t float

// backward kernel
__global__
void flash2_backward_kernel(
    const data_t* Q, const data_t* K, const data_t* V, const data_t* O, const data_t* dO, const data_t* L,
    const int N, const int d, const int Tc, const int Tr, const int Bc, const int Br,
    const data_t* dQ, const data_t* dK, const data_t* dV
){ 
    int tx  = threadIdx.x;
    int ntx = blockDim.x;

    int bx = blockIdx.x; // batch index
    int by = blockIdx.y; // head  index

    // offset into Q,K,V,O,l,m
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int L_offset   = (bx * gridDim.y * N) + (by * N);

    // Define SRAM for Q, K, V, S
    extern __shared__ data_t sram[];
    int tile_size_Q  = Br * d;
    int tile_size_KV = Bc * d;
    int tile_size_SP = Br * Bc;

    data_t* Qi  = sram;
    data_t* dQj = sram + tile_size_Q;

    data_t* Kj  = dQj + tile_size_Q;
    data_t* dKj = Kj  + tile_size_KV;

    data_t* Vj  = dKj + tile_size_KV;
    data_t* dVj = Vj  + tile_size_KV;

    data_t* Oi  = dVj + tile_size_KV;
    data_t* dOi = Oi  + tile_size_Q;

    // for both Sij and Pij
    data_t* Si  = dOi + tile_size_Q;
    data_t* dSi = Si  + tile_size_SP;

    data_t* Li  = dSi + Br;

    data_t* D   = dSi + tile_size_SP;

    for (int j = 0; j < Tc; j++){
        
        // Load Kj, Vj (Bc x d)
        for (int c = 0; c < Bc; c += ntx){
            if (tx + c < Bc){
                for (int x = 0; x < d; x++){
                    Kj[(tx + c) * d + x] = K[qkv_offset + tile_size_KV * j + (tx + c) * d + x];
                    Vj[(tx + c) * d + x] = V[qkv_offset + tile_size_KV * j + (tx + c) * d + x];
                }
            }
        }

        // __syncthreads();

        for (int i = 0; i < Tr; i++){
            
            // Load Qi, dQj, Oi, dOi, Li (Br x d), and conpute Di
            for (int r = 0; r < Br; r += ntx){
                if (tx + r < Br){
                    for (int x = 0; x < d; x++){
                        Qi[(tx + r) * d + x]  = Q[qkv_offset + tile_size_Q * i + (tx + r) * d + x];
                        dQj[(tx + r) * d + x] = dQ[qkv_offset + tile_size_Q * i + (tx + r) * d + x];
                        Oi[(tx + r) * d + x]  = O[qkv_offset + tile_size_Q * i + (tx + r) * d + x];
                        dOi[(tx + r) * d + x] = dO[qkv_offset + tile_size_Q * i + (tx + r) * d + x];
                    }
                    Li[tx + r] = L[L_offset + i * Br + tx + r];
                }
            }

            for (int r = 0; r < Br; r += ntx){
                if (tx + r < Br){
                    D[(tx + r) * d + x] = 0;
                    for (int x = 0; x < d; x++){
                        D[(tx + r) * d + x] += Oi[(tx + r) * d + x] * dOi[(tx + r) * d + x];
                    }
                }
            }

            __syncthreads();

            // compute Sij, Pij
            for (int rc = 0; rc < Bc * Br; rc += ntx){
                if (tx + rc < Bc * Br){
                    int r = (tx + rc) / Bc;
                    int c = (tx + rc) % Bc;

                    // for Sij
                    data_t sum = 0;
                    for (int x = 0; x < d; x++){
                        sum += Qi[r * d + x] * Kj[c * d + x];
                    }

                    // for Pij
                    Si[r * Bc + c] = __expf(sum - Li[r]);
                }
            }

            __syncthreads();

            // compute dVj
            for (int cd = 0; cd < Bc * d; cd += ntx){
                if (tx + cd < Bc * d){
                    int c = (tx + cd) / d;
                    int x = (tx + cd) % d;

                    // dVj[(tx + cd) * d + x] = 0;
                    for (int r = 0; r < Br; r++){
                        dVj[(tx + cd) * d + x] += Si[r * Bc + c] * dOi[r * d + x];
                    }
                }
            }
            
            // compute dPij
            for (int rc = 0; rc < Bc * Br; rc += ntx){
                if (tx + rc < Bc * Br){
                    int r = (tx + rc) / Bc;
                    int c = (tx + rc) % Bc;

                    dSi[r * Bc + c] = 0;
                    for (int x = 0; x < d; x++){
                        dSi[r * Bc + c] += Qi[r * d + x] * dVj[c * d + x];
                    }
                }
            }

            __syncthreads();

            // compute dSij
            for (int rc = 0; rc < Bc * Br; rc += ntx){
                if (tx + rc < Bc * Br){
                    int r = (tx + rc) / Bc;
                    int c = (tx + rc) % Bc;

                    dSi[r * Bc + c] = dSi[r * Bc + c] * (Si[r * Bc + c] - D[r]);
                }
            }

            __syncthreads();

            // compute dQi
            for (int rd = 0; rd < Br * d; rd += ntx){
                if (tx + rd < Br * d){
                    int r = (tx + rd) / d;
                    int x = (tx + rd) % d;

                    dQj[(tx + rd) * d + x] = 0;
                    for (int c = 0; c < Bc; c++){
                        dQj[(tx + rd) * d + x] += dSi[r * Bc + c] * Kj[c * d + x];
                    }
                }
            }

            // compute dKj
            for (int cd = 0; cd < Bc * d; cd += ntx){
                if (tx + cd < Bc * d){
                    int c = (tx + cd) / d;
                    int x = (tx + cd) % d;

                    dKj[(tx + cd) * d + x] = 0;
                    for (int r = 0; r < Br; r++){
                        dKj[(tx + cd) * d + x] += dSi[r * Bc + c] * Qi[r * d + x];
                    }
                }
            }
        }

    }


}



// backward function
std::vector<torch::Tensor> backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L
) {
    const int Bc = 32;
    const int Br = 32;

    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    torch::Device device(torch::kCUDA);
    dQ = dQ.to(device);
    dK = dK.to(device);
    dV = dV.to(device);

    const int sram_size = 2 * Bc * d * sizeof(data_t) + Br * d * sizeof(data_t) + Br * Bc * sizeof(data_t);
    int max_sram_size;

    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory per block: %d, Requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid(Tr, nh);
    dim3 block(Br);

    flash2_backward_kernel<<<grid, block, sram_size>>>(
        Q.data_ptr<data_t>(), K.data_ptr<data_t>(), V.data_ptr<data_t>(), O.data_ptr<data_t>(), dO.data_ptr<data_t>(), L.data_ptr<data_t>(),
        N, d, Tc, Tr, Bc, Br,
        dQ.data_ptr<data_t>(), dK.data_ptr<data_t>(), dV.data_ptr<data_t>()
    );

    return {dQ, dK, dV};
}




