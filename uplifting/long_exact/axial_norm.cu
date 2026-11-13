// Width-128 row-transpose specialization of PyTorch v2.3.1 CUDA LayerNorm.
// Arithmetic follows the pinned vectorized Welford schedule and affine.
// https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Moments {
  float mean;
  float m2;
};

template <int Count>
__device__ __forceinline__ Moments observe(Moments prior, float value) {
  const float delta = __fsub_rn(value, prior.mean);
  const float mean = __fmaf_rn(delta, 1.0f / Count, prior.mean);
  return {mean, __fmaf_rn(delta, __fsub_rn(value, mean), prior.m2)};
}

template <int Count, int PeerCount>
__device__ __forceinline__ Moments combine(Moments current, Moments peer) {
  constexpr float current_fraction = float(Count) / (Count + PeerCount);
  constexpr float peer_fraction = float(PeerCount) / (Count + PeerCount);
  const float delta = __fsub_rn(current.mean, peer.mean);
  const float current_mean = __fmul_rn(current.mean, current_fraction);
  const float weighted_delta = __fmul_rn(__fmul_rn(delta, delta), float(PeerCount));
  return {__fmaf_rn(peer.mean, peer_fraction, current_mean),
          __fmaf_rn(weighted_delta, current_fraction, __fadd_rn(peer.m2, current.m2))};
}

template <int Delta>
__device__ __forceinline__ Moments shuffle_down(Moments value) {
  return {__shfl_down_sync(0xffffffffu, value.mean, Delta),
          __shfl_down_sync(0xffffffffu, value.m2, Delta)};
}

__device__ __forceinline__ float affine(float x, float gamma, float beta,
                                        float mean, float rstd) {
  const float norm = __fmul_rn(rstd, __fsub_rn(x, mean));
  return __fmaf_rn(gamma, norm, beta);
}

constexpr unsigned warp_size = 32;
constexpr unsigned block_threads = 128;
constexpr unsigned block_rows = block_threads / warp_size;

__global__ void axial_norm_kernel(const float4* __restrict__ x,
                                  const float4* __restrict__ gamma,
                                  const float4* __restrict__ beta,
                                  float4* __restrict__ y,
                                  float* __restrict__ mean,
                                  float* __restrict__ rstd,
                                  uint32_t length, float eps) {
  const unsigned lane = threadIdx.x % warp_size;
  const unsigned row = blockIdx.x * block_rows + threadIdx.x / warp_size;
  if (row >= length * length) {
    return;  // Uniform for the entire warp.
  }
  const unsigned source_row = (row % length) * length + row / length;
  const float4 value = x[source_row * warp_size + lane];
  Moments moments = observe<1>({0.0f, 0.0f}, value.x);
  moments = observe<2>(moments, value.y);
  moments = observe<3>(moments, value.z);
  moments = observe<4>(moments, value.w);

  moments = combine<4, 4>(moments, shuffle_down<16>(moments));
  moments = combine<8, 8>(moments, shuffle_down<8>(moments));
  moments = combine<16, 16>(moments, shuffle_down<4>(moments));
  moments = combine<32, 32>(moments, shuffle_down<2>(moments));
  moments = combine<64, 64>(moments, shuffle_down<1>(moments));

  // Stock assigns four warps to a row, with only the first warp reading.
  // Reproduce its two merges with empty peer moments before broadcasting.
  if (lane == 0) {
    const Moments empty{0.0f, 0.0f};
    moments = combine<128, 0>(moments, empty);
    moments = combine<128, 0>(moments, empty);
  }
  const float row_mean = __shfl_sync(0xffffffffu, moments.mean, 0);
  const float row_m2 = __shfl_sync(0xffffffffu, moments.m2, 0);
  const float inverse = rsqrtf(__fadd_rn(__fmul_rn(row_m2, 1.0f / 128.0f), eps));
  const float4 scale = gamma[lane];
  const float4 shift = beta[lane];
  y[row * warp_size + lane] = make_float4(
      affine(value.x, scale.x, shift.x, row_mean, inverse),
      affine(value.y, scale.y, shift.y, row_mean, inverse),
      affine(value.z, scale.z, shift.z, row_mean, inverse),
      affine(value.w, scale.w, shift.w, row_mean, inverse));
  if (lane == 0 && mean != nullptr) {
    mean[row] = row_mean;
    rstd[row] = inverse;
  }
}

}  // namespace

extern "C" int axial_norm_launch(const void* x, const void* gamma, const void* beta,
                                 void* y, void* mean, void* rstd, uint32_t length,
                                 float eps, void* stream) {
  const unsigned rows = length * length;
  axial_norm_kernel<<<(rows + block_rows - 1) / block_rows, block_threads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float4*>(x), static_cast<const float4*>(gamma),
      static_cast<const float4*>(beta), static_cast<float4*>(y),
      static_cast<float*>(mean), static_cast<float*>(rstd), length, eps);
  return static_cast<int>(cudaPeekAtLastError());
}
