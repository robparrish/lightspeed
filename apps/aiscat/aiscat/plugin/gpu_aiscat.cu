#include "gpu_aiscat.hpp"
#include <lightspeed/resource_list.hpp>
#include <lightspeed/gpu_context.hpp>
#include <lightspeed/tensor.hpp>

// Place CUERR; after any interesting CUDA API line to check for errors
#include <cstdio>
#include <cstdlib>
#define CUERR { \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
}
    
namespace lightspeed {

// => Type-Templated Math <= //

template<typename T> T __device__ __inline__ sin_R(T x);
double __device__ __inline__ sin_R(double x) { return sin(x); }
float __device__ __inline__ sin_R(float x) { return sinf(x); }

template<typename T> T __device__ __inline__ cos_R(T x);
double __device__ __inline__ cos_R(double x) { return cos(x); }
float __device__ __inline__ cos_R(float x) { return cosf(x); }

template<typename T> T __device__ __inline__ sqrt_R(T x);
double __device__ __inline__ sqrt_R(double x) { return sqrt(x); }
float __device__ __inline__ sqrt_R(float x) { return sqrtf(x); }

template<typename T> void __device__ __inline__ sincos_R(T x, T* s, T* c);
void __device__ __inline__ sincos_R(double x, double* s, double* c) { sincos(x, s, c); }
void __device__ __inline__ sincos_R(float x, float* s, float* c) { sincosf(x, s, c); }

// => Warp/Block Reductions <= //

template <typename T>
__inline__ __device__
T warpReduceSum(T val) 
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// => Perpendicular Moments <= //

template <typename T>
__inline__ __device__
T blockReduceSum(T val) 
{
    // TODO: This block reduction is dangerous:
    // (1) It assumes that the warp size is ~32
    // (2) It assumes that the max block size is 1024 threads
    static __shared__ T shared[32]; // Max of 1024 threads in a block (32 x 32)
    // General case
    int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads(); // Make sure all warps check data into shared before 
    val = (tid < blockDim.x * blockDim.y * blockDim.z / warpSize) ? shared[tid] : static_cast<T>(0);
    if (wid==0) val = warpReduceSum(val);
    __syncthreads(); // Make sure all threads that need to can read the shared data before proceeding to next shared op
    return val; // Only correct on the 0-th warp.
}

template<typename T, typename T4, typename A, int blocksize>
__global__ void perpendicular_moments_kernel(
    int nP,            // Number of points
    T L,               // DeBroglie wavelength
    const T4* xyzq_G,  // Point charges (nxyz, each element (x,y,z,w))
    const T* s_G,      // Momentum transfer (ns,)
    A* M0_G,           // Moments for each P point (nS * nQblock * nPblock)
    A* M2_G            // Moments for each P point (nS * nQblock * nPblock)
    )
{
    // 3D grid (PblockO, QblockO, S) of 1D blocks (PQlane)
    T s = s_G[blockIdx.z]; // LD (1-off)

    // Shared set of P points
    __shared__ T4 xyzqP_S[blocksize];

    // Grid stride loop
    A M0 = static_cast<A>(0.0);
    A M2 = static_cast<A>(0.0);
    for (int P0 = blockIdx.x * blocksize; P0 < nP; P0 += gridDim.x * blocksize) {
        if (P0 + threadIdx.x < nP) {
            xyzqP_S[threadIdx.x] = xyzq_G[P0 + threadIdx.x]; // LD Coalesced
        }
        __syncthreads(); 
    for (int Q = threadIdx.x + blockIdx.y * blocksize; Q < nP; Q += gridDim.y * blocksize) {
        T4 xyzqQ = xyzq_G[Q]; // LD Coalesced
        // #pragma unroll 8 Does not affect performance
        for (int dP = 0; dP < blocksize; dP++) {
            int P = P0 + dP;
            if (P >= nP) break;
            if (P > Q) continue;
            T4 xyzqP = xyzqP_S[dP]; 
            T dx = xyzqQ.x - xyzqP.x;     
            T dy = xyzqQ.y - xyzqP.y;     
            T dz = xyzqQ.z - xyzqP.z;     
            T wPQ = xyzqP.w * xyzqQ.w;
            if (P != Q) wPQ *= static_cast<T>(2.0);
            T r2 = dx*dx + dy*dy + dz*dz;
            T r = sqrt_R(r2);
            T sr = s * r;
            if (sr == static_cast<T>(0.0)) {
                // K0 = 1.0/3.0, K2 = 0.0
                M0 += wPQ / static_cast<T>(3.0);
            } else {
                T sg2 = (dx*dx + dy*dy) / r2;
                T J1_sr = sin_R(sr) / (sr * sr * sr) - cos_R(sr) / (sr * sr);
                T J2 = (static_cast<T>(3.0) / (sr * sr) - static_cast<T>(1.0)) * sin_R(sr) / sr - static_cast<T>(3.0) * cos_R(sr) / (sr * sr);
                T del = s*L / static_cast<T>(4.0 * M_PI);
                T A2 = static_cast<T>(0.5) * (static_cast<T>(2.0) - static_cast<T>(3.0) * sg2) * (static_cast<T>(1.0) - del * del);
                M0 += (A) (wPQ * (J1_sr - static_cast<T>(0.5) * (sg2 + A2) * J2));
                M2 += (A) (wPQ * (-static_cast<T>(0.5) * A2 * J2));
            } 
        }
    }
    __syncthreads();
    }
    
    M0 = blockReduceSum(M0);
    M2 = blockReduceSum(M2);
    // ST (1-off)
    if (threadIdx.x == 0) {
        M0_G[blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z))] = M0;
        M2_G[blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z))] = M2;
    }
}

template<typename T, typename T4, typename A>
std::shared_ptr<Tensor> perpendicular_moments_gpu(
    const std::shared_ptr<ResourceList>& resources,
    T L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    )
{
    // Validity checks
    s->ndim_error(1);
    xyzq->ndim_error(2);
    xyzq->shape_error({xyzq->shape()[0], 4});
    
    // Sizes
    size_t nS = s->shape()[0];
    size_t nP = xyzq->shape()[0];

    // Pointers
    const double* sp = s->data().data();
    const double* xyzqp = xyzq->data().data();

    // GPU Setup
    const std::shared_ptr<GPUContext>& gpu = resources->gpus()[0];
    cudaSetDevice(gpu->id()); 

    // Sizing
    constexpr int blocksize4 = 512;
    constexpr int gridsize4 = 4;
    dim3 block_size(blocksize4);
    dim3 grid_size(gridsize4, gridsize4, nS);

    // Host copy of momentum transfers
    T* s_H = new T[nS]; 
    for (size_t S = 0; S < nS; S++) {
        s_H[S] = sp[S];
    }

    // Device copy of momentum transfers
    T* s_D;
    cudaMalloc(&s_D, nS * sizeof(T));
    cudaMemcpy(s_D, s_H, nS * sizeof(T), cudaMemcpyHostToDevice);
    CUERR;

    // Host copy of point charges
    T4* xyzq_H = new T4[nP];
    ::memset(xyzq_H, '\0', nP * sizeof(T4));
    for (size_t P = 0; P < nP; P++) {
        xyzq_H[P].x = xyzqp[4*P + 0];
        xyzq_H[P].y = xyzqp[4*P + 1];
        xyzq_H[P].z = xyzqp[4*P + 2];
        xyzq_H[P].w = xyzqp[4*P + 3];
    }

    // Device copy of point charges
    T4* xyzq_D;
    cudaMalloc(&xyzq_D, nP * sizeof(T4));
    cudaMemcpy(xyzq_D, xyzq_H, nP * sizeof(T4), cudaMemcpyHostToDevice);
    CUERR;

    // Device copy of detector moments
    A* M0_D;
    cudaMalloc(&M0_D, grid_size.x * grid_size.y * grid_size.z * sizeof(A));
    CUERR;
    A* M2_D;
    cudaMalloc(&M2_D, grid_size.x * grid_size.y * grid_size.z * sizeof(A));
    CUERR;

    // Launch kernel
    perpendicular_moments_kernel<T, T4, A, blocksize4><<<grid_size,block_size>>>(nP, L, xyzq_D, s_D, M0_D, M2_D);
    CUERR;

    // Host copy of results
    A* M0_H = new A[grid_size.x * grid_size.y * grid_size.z];
    cudaMemcpy(M0_H, M0_D, (grid_size.x * grid_size.y * grid_size.z) * sizeof(A), cudaMemcpyDeviceToHost);
    CUERR;
    A* M2_H = new A[grid_size.x * grid_size.y * grid_size.z];
    cudaMemcpy(M2_H, M2_D, (grid_size.x * grid_size.y * grid_size.z) * sizeof(A), cudaMemcpyDeviceToHost);
    CUERR;
    
    // Target
    std::shared_ptr<Tensor> M(new Tensor({nS, 2}));
    double* Mp = M->data().data();
    for (size_t S = 0; S < nS; S++) {
        for (size_t PQ = 0; PQ < grid_size.x * grid_size.y; PQ++) {
            Mp[2*S + 0] += (double) M0_H[S * grid_size.x * grid_size.y + PQ];
            Mp[2*S + 1] += (double) M2_H[S * grid_size.x * grid_size.y + PQ];
        }
    }

    // Free memory
    cudaFree(s_D);
    cudaFree(xyzq_D);
    cudaFree(M0_D);
    cudaFree(M2_D);
    
    delete[] s_H;
    delete[] xyzq_H;
    delete[] M0_H;
    delete[] M2_H;

    return M;
}

std::shared_ptr<Tensor> perpendicular_moments_gpu_f_f(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    )
{
    return perpendicular_moments_gpu<float, float4, float>(resources, L, s, xyzq);
}

std::shared_ptr<Tensor> perpendicular_moments_gpu_f_d(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    )
{
    return perpendicular_moments_gpu<float, float4, double>(resources, L, s, xyzq);
}

std::shared_ptr<Tensor> perpendicular_moments_gpu_d_d(
    const std::shared_ptr<ResourceList>& resources,
    double L,
    const std::shared_ptr<Tensor>& s,
    const std::shared_ptr<Tensor>& xyzq
    )
{
    return perpendicular_moments_gpu<double, double4, double>(resources, L, s, xyzq);
}

// => Aligned Detector Signals <= //

template<typename T, typename T3, typename T4, typename A, int blocksize>
__global__ void aligned_diffraction_kernel(
    int nP,
    const T3* sxyz_G,
    const T4* xyzq_G,
    A* FR_G,
    A* FI_G
    )
{
    // 2D grid (S, Pblock) of 1D blocks (Plane)
    T3 s = sxyz_G[blockIdx.x];

    A FR = static_cast<A>(0.0);
    A FI = static_cast<A>(0.0);
    for (int P = threadIdx.x + blockIdx.y * blocksize; P < nP; P += gridDim.y * blocksize) {
        T4 xyzq = xyzq_G[P];
        T arg = s.x * xyzq.x + s.y * xyzq.y + s.z * xyzq.z;
        T c, s;
        sincos_R(arg, &c, &s);
        FR += xyzq.w * c;
        FI += xyzq.w * s;
        // FR += xyzq.w * cos_R(arg);
        // FI += xyzq.w * sin_R(arg);
    }
    
    FR = blockReduceSum(FR);
    FI = blockReduceSum(FI);
    
    if (threadIdx.x == 0) {
        FR_G[blockIdx.x + gridDim.x * blockIdx.y] = FR;
        FI_G[blockIdx.x + gridDim.x * blockIdx.y] = FI;
    } 
}


template<typename T, typename T3, typename T4, typename A>
std::shared_ptr<Tensor> aligned_diffraction_gpu(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq
    )
{
    // Validity checks
    sxyz->ndim_error(2);
    sxyz->shape_error({sxyz->shape()[0], 3});
    xyzq->ndim_error(2);
    xyzq->shape_error({xyzq->shape()[0], 4});
    
    // Sizes
    size_t nS = sxyz->shape()[0];
    size_t nP = xyzq->shape()[0];

    // Pointers
    const double* sxyzp = sxyz->data().data();
    const double* xyzqp = xyzq->data().data();

    // GPU Setup
    const std::shared_ptr<GPUContext>& gpu = resources->gpus()[0];
    cudaSetDevice(gpu->id()); 

    // Sizing
    constexpr int blocksizeA = 512;
    constexpr int gridsizeA = 4;
    dim3 block_size(blocksizeA);
    dim3 grid_size(nS, gridsizeA);

    // Host copy of momentum transfers
    T3* sxyz_H = new T3[nS];
    for (size_t S = 0; S < nS; S++) {
        sxyz_H[S].x = sxyzp[3*S + 0];
        sxyz_H[S].y = sxyzp[3*S + 1];
        sxyz_H[S].z = sxyzp[3*S + 2];
    }

    // Device copy of momentum transfers
    T3* sxyz_D;
    cudaMalloc(&sxyz_D, nS * sizeof(T3));
    cudaMemcpy(sxyz_D, sxyz_H, nS * sizeof(T3), cudaMemcpyHostToDevice);
    CUERR;

    // Host copy of point charges
    T4* xyzq_H = new T4[nP];
    ::memset(xyzq_H, '\0', nP * sizeof(T4));
    for (size_t P = 0; P < nP; P++) {
        xyzq_H[P].x = xyzqp[4*P + 0];
        xyzq_H[P].y = xyzqp[4*P + 1];
        xyzq_H[P].z = xyzqp[4*P + 2];
        xyzq_H[P].w = xyzqp[4*P + 3];
    }

    // Device copy of point charges
    T4* xyzq_D;
    cudaMalloc(&xyzq_D, nP * sizeof(T4));
    cudaMemcpy(xyzq_D, xyzq_H, nP * sizeof(T4), cudaMemcpyHostToDevice);
    CUERR;

    // Device copy of detector moments
    A* FR_D;
    A* FI_D;
    cudaMalloc(&FR_D, gridsizeA * nS * sizeof(A));
    cudaMalloc(&FI_D, gridsizeA * nS * sizeof(A));
    CUERR;

    // Launch kernel
    aligned_diffraction_kernel<T, T3, T4, A, blocksizeA><<<grid_size,block_size>>>(nP, sxyz_D, xyzq_D, FR_D, FI_D);
    CUERR;

    // Host copy of results
    A* FR_H = new A[gridsizeA * nS];
    A* FI_H = new A[gridsizeA * nS];
    cudaMemcpy(FR_H, FR_D, gridsizeA * nS * sizeof(A), cudaMemcpyDeviceToHost);
    cudaMemcpy(FI_H, FI_D, gridsizeA * nS * sizeof(A), cudaMemcpyDeviceToHost);
    CUERR;
    
    // Target
    std::shared_ptr<Tensor> M(new Tensor({nS}));
    double* Mp = M->data().data(); 
    for (int S = 0; S < nS; S++) {
        double FR = 0.0;
        double FI = 0.0; 
        for (int Q = 0; Q < gridsizeA; Q++) {
            FR += (double) FR_H[Q * nS + S];
            FI += (double) FI_H[Q * nS + S];
        }
        Mp[S] = FR * FR + FI * FI;
    }

    // Free memory
    cudaFree(sxyz_D);
    cudaFree(xyzq_D);
    cudaFree(FR_D);
    cudaFree(FI_D);
    
    delete[] sxyz_H;
    delete[] xyzq_H;
    delete[] FR_H;
    delete[] FI_H;

    return M;
}

std::shared_ptr<Tensor> aligned_diffraction_gpu_f_f(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq)
{
    return aligned_diffraction_gpu<float, float3, float4, float>(resources, sxyz, xyzq);
}

std::shared_ptr<Tensor> aligned_diffraction_gpu_f_d(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq)
{
    return aligned_diffraction_gpu<float, float3, float4, double>(resources, sxyz, xyzq);
}

std::shared_ptr<Tensor> aligned_diffraction_gpu_d_d(
    const std::shared_ptr<ResourceList>& resources,
    const std::shared_ptr<Tensor>& sxyz,
    const std::shared_ptr<Tensor>& xyzq)
{
    return aligned_diffraction_gpu<double, double3, double4, double>(resources, sxyz, xyzq);
}

} // namespace lightspeed
