#include <vector>
#include <iostream>
#include "../tester/utils.h"
#include <cassert>

// *********************************************************************
//
// * kthLargest Part
//
// *********************************************************************

/**
 * @brief Merge kernel for merge sort.
 *
 * @tparam T int or float.
 * @param input Input array.
 * @param output Output array that has been sorted partially.
 * @param width Width of the subarrays to merge.
 * @param size Array size, for bound check.
 * @return No return.
 */
template <typename T>
__global__ void mergeKernel(const T *input, T *output, int width, int size)
{
    // Get the thread global id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the start index for this thread
    int start = tid * (2 * width);
    // Bound check
    if (start < size)
    {
        // Get the mid index for this thread
        int mid = min(start + width, size);
        // Get the end index for this thread
        int end = min(start + 2 * width, size);
        int i = start, j = mid, k = start;
        // Merge the two parts into output
        while (i < mid && j < end)
        {
            if (input[i] > input[j])
                output[k++] = input[i++];
            else
                output[k++] = input[j++];
        }
        // Copy remaining elements
        while (i < mid)
            output[k++] = input[i++];
        while (j < end)
            output[k++] = input[j++];
    }
}

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 *
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed.
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
T kthLargest(const std::vector<T> &h_input, size_t k)
{
    // Check for valid input
    if (h_input.empty() || k == 0 || k > h_input.size())
    {
        return T(-100); // Invalid case
    }
    const unsigned int input_size = h_input.size();
    const unsigned int input_bytes = input_size * sizeof(T);

    // Allocate device memory and copy input vector to device
    T *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    // Invoke kernel function to find the k-th largest element
    int threads = 256;
    for (int width = 1; width < input_size; width *= 2)
    {
        // Per thread handles 2*width elements, merge them into output
        int blocks = (input_size + threads * 2 * width - 1) / (threads * 2 * width);
        mergeKernel<<<blocks, threads>>>(d_input, d_output, width, input_size);
        std::swap(d_input, d_output);
    }

    // Copy the k-th largest element back to host
    // The sorted output is swapped to d_input
    T h_output;
    CUDA_CHECK(cudaMemcpy(&h_output, &d_input[k - 1], sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return T(h_output);
}

// *********************************************************************
//
// * flashAttention Part
//
// *********************************************************************

template <typename T>
__global__ void flashAttnKernel(
    const T *d_q, // [batch_size, target_seq_len, query_heads, head_dim]
    const T *d_k, // [batch_size, src_seq_len, kv_heads, head_dim]
    const T *d_v, // [batch_size, src_seq_len, kv_heads, head_dim]
    T *d_o,       // [batch_size, target_seq_len, query_heads, head_dim]
    const int Tr, const int Tc, const int Br, const int Bc,
    const int batch_size, const int target_seq_len, const int src_seq_len,
    const int query_heads, const int kv_heads, const int head_dim,
    const int group_size, const bool is_causal, const float softmax_scale,
    float *d_l, float *d_m)
{
    extern __shared__ float smem[]; // dynamic shared memory
    int q_tile_size = Br * head_dim;
    int k_tile_size = Bc * head_dim;
    int v_tile_size = Bc * head_dim;

    float *Qi = smem;             // Q [Br, head_dim]
    float *Kj = Qi + q_tile_size; // K [Bc, head_dim]
    float *Vj = Kj + k_tile_size; // V [Bc, head_dim]
    float *S = Vj + v_tile_size;  // S [Br, Bc]

    /********************************************************
     *  Check Shared Memory Addr
     ********************************************************/

    int tx = threadIdx.x; // thread within block
    int bx = blockIdx.x;  // batch index
    int by = blockIdx.y;  // query head index
    // int Br = blockDim.x

    // Compute Q/K/V/O/l/m offsets
    int q_offset = bx * target_seq_len * query_heads * head_dim + by * head_dim;          // [batch_size, target_seq_len, query_heads, head_dim]
    int k_offset = bx * src_seq_len * kv_heads * head_dim + (by / group_size) * head_dim; // [batch_size, src_seq_len, kv_heads, head_dim]
    int v_offset = k_offset;                                                              // [batch_size, src_seq_len, kv_heads, head_dim]
    int o_offset = q_offset;                                                              // [batch_size, target_seq_len, query_heads, head_dim]
    int lm_offset = bx * target_seq_len * query_heads + by;                               // [batch_size, target_seq_len, query_heads]

    for (int j = 0; j < Tc; ++j) // Outer loop
    {
        // 1. Load K and V into shared memory
        if (tx < Bc)
        {
            int k_row = j * Bc + tx;
            for (int x = 0; x < head_dim; ++x)
            {
                float kval = 0.f, vval = 0.f;
                if (k_row < src_seq_len)
                {
                    kval = d_k[k_offset + k_row * kv_heads * head_dim + x]; // [batch_size, src_seq_len, kv_heads, head_dim]
                    vval = d_v[v_offset + k_row * kv_heads * head_dim + x]; // [batch_size, src_seq_len, kv_heads, head_dim]
                }
                Kj[(tx * head_dim + x)] = kval; // [batch_size, src_seq_len, kv_heads, head_dim]
                Vj[(tx * head_dim + x)] = vval; // [batch_size, src_seq_len, kv_heads, head_dim]
            }
        }
        __syncthreads(); // Ensure all threads have loaded Kj and Vj

        /********************************************************
         *  Check Load Kj and Vj to Shared Memory. Ok!
         ********************************************************/

        for (int i = 0; i < Tr; ++i) // Inner loop
        {
            int q_row = i * Br + tx; // global index
            bool q_valid = (q_row < target_seq_len);

            // 2. Load Q into shared memory
            for (int x = 0; x < head_dim; ++x) // Q [Br, d]
            {
                Qi[(tx * head_dim + x)] = q_valid ? d_q[q_offset + q_row * query_heads * head_dim + x] : 0.f; // [batch_size, target_seq_len, query_heads, head_dim]
            }

            // Initially, row_m_prev = -INFINITY, row_l_prev = 0
            float row_m_prev = -INFINITY;
            float row_l_prev = 0.f;
            if (j != 0 && q_valid)
            {
                row_m_prev = d_m[lm_offset + q_row * query_heads]; // [batch_size, target_seq_len, query_heads]
                row_l_prev = d_l[lm_offset + q_row * query_heads];
            }

            /********************************************************
             *  Check Load Qi to Shared Memory. Ok!
             ********************************************************/

            // 3. S = Q * K^T * softmax_scale, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; ++y) // [Br, d] * [d, Bc] -> [Br, Bc]
            {
                int k_pos = j * Bc + y;
                float sum = 0.f;
                for (int x = 0; x < head_dim; ++x)
                {
                    sum += Qi[tx * head_dim + x] * Kj[y * head_dim + x]; // Q * K^T
                }
                sum *= softmax_scale;
                bool valid = q_valid && (k_pos < src_seq_len);
                if (!valid) // Bound check
                    sum = -INFINITY;
                
                /********************************************************************************
                 * Note: I think this condition has error !!!
                 * it should be k_pos > q_row + src_seq_len - target_seq_len, considering kvcache
                 ********************************************************************************/
                if (valid && is_causal && (k_pos > q_row)) // k_pos > q_row + src_seq_len - target_seq_len
                    sum = -INFINITY;

                S[tx * Bc + y] = sum; // [Br, Bc]
                row_m = max(row_m, sum);
            }
            /********************************************************
             *  Check Compute S = Q * K^T, Record row_m
             ********************************************************/

            // 4. P = exp(S - row_m), row_l = row_sum(P)
            float row_l = 0.f;
            for (int y = 0; y < Bc; ++y)
            {
                S[tx * Bc + y] = expf(S[tx * Bc + y] - row_m);
                row_l += S[tx * Bc + y];
            }
            /********************************************************
             *  Check Tile Exp, Calculate row_l
             ********************************************************/

            // 5. Compute new m and l
            float row_m_new = max(row_m, row_m_prev);
            float row_l_new = (expf(row_m_prev - row_m_new) * row_l_prev) + expf(row_m - row_m_new) * row_l;

            // 6. Write O, l, m to HBM
            for (int x = 0; x < head_dim; ++x) // P * V
            {
                float pv = 0.f;
                for (int y = 0; y < Bc; ++y) // [Br, Bc] * [Bc, d] -> [Br, d]
                {
                    pv += S[tx * Bc + y] * Vj[y * head_dim + x];
                }
                // Update output
                if (q_valid)
                    d_o[o_offset + q_row * query_heads * head_dim + x] = (1 / row_l_new) * ((row_l_prev * expf(row_m_prev - row_m_new) * d_o[o_offset + q_row * query_heads * head_dim + x]) + (expf(row_m - row_m_new) * pv));
            }
            if (q_valid)
            {
                d_m[lm_offset + q_row * query_heads] = row_m_new;
                d_l[lm_offset + q_row * query_heads] = row_l_new;
            }
            /********************************************************
             *  Check Calc O, l, m
             ********************************************************/
        }
        __syncthreads();
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal)
{
    // Check Input
    static_assert(std::is_same_v<T, float>, "Template parameter T must be float now.");
    assert(h_q.size() == batch_size * target_seq_len * query_heads * head_dim);
    assert(h_k.size() == batch_size * src_seq_len * kv_heads * head_dim);
    assert(h_v.size() == batch_size * src_seq_len * kv_heads * head_dim);
    assert(h_o.size() == batch_size * target_seq_len * query_heads * head_dim);
    // Grouped Query Attention
    assert(query_heads % kv_heads == 0);

    // h_q: [batch_size, target_seq_len, query_heads, head_dim]
    // h_k: [batch_size, src_seq_len, kv_heads, head_dim]
    // h_v: [batch_size, src_seq_len, kv_heads, head_dim]
    // h_o: [batch_size, target_seq_len, query_heads, head_dim]

    float *d_q, *d_k, *d_v, *d_o, *d_m, *d_l;

    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * h_q.size()));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(float) * h_k.size()));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float) * h_v.size()));
    CUDA_CHECK(cudaMalloc(&d_o, sizeof(float) * h_o.size()));
    CUDA_CHECK(cudaMalloc(&d_m, sizeof(float) * batch_size * target_seq_len * query_heads));
    CUDA_CHECK(cudaMalloc(&d_l, sizeof(float) * batch_size * target_seq_len * query_heads));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * h_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(float) * h_k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(float) * h_v.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_o, 0, sizeof(float) * h_o.size()));
    // CUDA_CHECK(cudaMemset(d_m, 0, sizeof(float) * batch_size * target_seq_len * query_heads));   // Done in kernel later.
    // CUDA_CHECK(cudaMemset(d_l, 0, sizeof(float) * batch_size * target_seq_len * query_heads));   // Done in kernel later.

    /*******************************************************************
     *  Check GPU Global Memory.
     ******************************************************************/

    // TODO: Dynamic shared memory size: Br*Dh + 2*Bc*Dh + Br*Bc floats
    // Bc = ceil(M/4d), Br = min(Bc, d)
    int Br = 16, Bc = 16;

    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    // int Bc = (max_smem + 4*head_dim - 1) / (4 * head_dim);  // ceil(M / (4*d))
    // int Br = std::min(Br, head_dim);

    const int Tr = std::ceil((float)(target_seq_len) / Br);
    const int Tc = std::ceil((float)(src_seq_len) / Bc);

    // Calculate SRAM size needed per block, Q + K + V + S
    int smem_bytes = (Br * head_dim + 2 * Bc * head_dim + Br * Bc) * sizeof(float);
    if (smem_bytes > max_smem)
    {
        fprintf(stderr, "[FlashAttention] Requested shared mem %d > max %d. Reduce Br/Bc or Dh.\n", smem_bytes, max_smem);
        CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_k));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_o));
        CUDA_CHECK(cudaFree(d_l));
        CUDA_CHECK(cudaFree(d_m));
        throw std::runtime_error("Shared memory too large for this configuration");
    }

    float softmax_scale = 1.0f / std::sqrt(head_dim);
    int group_size = query_heads / kv_heads;

    dim3 grid(batch_size, query_heads);
    dim3 block(Br);

    /*******************************************************************
     *  Check Shared Memory and Arguments.
     ******************************************************************/

    flashAttnKernel<<<grid, block, smem_bytes>>>(
        d_q, d_k, d_v, d_o,
        Tr, Tc, Br, Bc, batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim,
        group_size, is_causal, softmax_scale,
        d_l, d_m);

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, sizeof(float) * h_o.size(), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_l));
    CUDA_CHECK(cudaFree(d_m));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int> &, size_t);
template float kthLargest<float>(const std::vector<float> &, size_t);
template void flashAttention<float>(const std::vector<float> &, const std::vector<float> &,
                                    const std::vector<float> &, std::vector<float> &,
                                    int, int, int, int, int, int, bool);
