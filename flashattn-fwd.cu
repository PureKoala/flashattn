#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define data_t float


// forward kernel
__global__
void flash2_forward_kernel(
    const data_t* Q, const data_t* K, const data_t* V, const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br, const data_t softmax_scale,
    data_t* L, data_t* O
){
    int tx = threadIdx.x;

    int bx = blockIdx.x; // batch index
    int by = blockIdx.y; // head  index

    // offset into Q,K,V,O,l,m
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int L_offset   = (bx * gridDim.y * N) + (by * N);

    // Define SRAM for Q, K, V, S
    extern __shared__ data_t sram[];
    int tile_size_Q  = Br * d;
    int tile_size_KV = Bc * d;
    data_t* Qi = sram;
    data_t* Kj = Qi + tile_size_Q;
    data_t* Vj = Kj + tile_size_KV;
    data_t* SP = Vj + tile_size_KV;

    data_t m[Br];
    data_t l[Br];

    for (int i = 0; i < Tr; i++){
        for (int x = 0; x < d; x++){
            Qi[tx * d + x] = Q[qkv_offset + tile_size_Q * i + tx * d + x];   
        }

        m[tx] = -INFINITY;
        l[tx] = 0;

        __syncthreads();

        for (int j = 0; j < Tc; j++){

            // load Kj, Vj, m, l
            for (int x = 0; x < d; x++){
                for(int y = 0; y < Bc; y += Br){
                    if (tx + y < Bc){
                        Kj[(tx + y) * d + x] = K[qkv_offset + tile_size_KV * j + (tx + y) * d + x];
                        Vj[(tx + y) * d + x] = V[qkv_offset + tile_size_KV * j + (tx + y) * d + x];
                    }
                }
            }

            __syncthreads();

            data_t row_m_prev = m[tx];
            data_t row_l_prev = l[tx];

            // compute SP = QK^T, row_m = rowmax(S)
            data_t row_m = -INFINITY;
            for (int y = 0; y < Bc; y++){
                data_t sum = 0;
                for (int x = 0; x < d; x++){
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale;
                SP[tx * Bc + y] = sum;

                if (sum > row_m){
                    row_m = sum;
                }
            }

            // compute row_l, P
            data_t row_l = 0;
            for(int y = 0; y < Bc; y++){
                SP[tx * Bc + y] = __expf(SP[tx * Bc + y] - row_m);
                row_l += SP[tx * Bc + y];
            }

            // compute new m, l
            data_t row_m_new = max(row_m, row_m_prev);
            data_t row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev + __expf(row_m - row_m_new) * row_l;

            // compute O, l, m
            for (int x = 0; x < d; x++){
                data_t pv = 0; // Pij * Vj
                for (int y = 0; y < Bc; y++){
                    pv += SP[tx * Bc + y] * Vj[y * d + x];
                }
                O[qkv_offset + tile_size_Q * i + tx * d + x] = O[qkv_offset + tile_size_Q * i + tx * d + x] * __expf(row_m_prev - row_m_new) + pv;
            }

            m[tx] = row_m_new;
            l[tx] = row_l_new;

        }

        // write O, L to HBM
        for (int x = 0; x < d; x++){
            O[qkv_offset + tile_size_Q * i + tx * d + x] = O[qkv_offset + tile_size_Q * i + tx * d + x] / l[tx];
        }
        L[L_offset + i * Br + tx] = l[tx];
        
        __syncthreads();
    }
}



// forward function
torch :: Tensor forward(
    torch :: Tensor Q, torch :: Tensor K, torch :: Tensor V
) {
    const int Bc = 32;
    const int Br = 32;

    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N + Bc - 1) / Bc;
    const float softmax_scale = 1.0f / sqrtf(d);

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, nh, N});

    torch::Device device(torch::kCUDA);
    L = L.to(device);

    const int sram_size = 2* Bc * d * sizeof(data_t) + Br * d * sizeof(data_t) + Br * Bc * sizeof(data_t);
    int max_sram_size;

    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory per block: %d, Requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid(Tr, nh);
    dim3 block(Br);

    flash2_forward_kernel<<<grid, block, sram_size>>>(Q.data_ptr<data_t>(), K.data_ptr<data_t>(), V.data_ptr<data_t>(), N, d, Tc, Tr, Bc, Br, softmax_scale, L.data_ptr<data_t>(), O.data_ptr<data_t>());

    return O;

}




