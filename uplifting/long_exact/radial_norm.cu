// Width-32 specialization of the arithmetic in PyTorch v2.3.1's vectorized
// CUDA LayerNorm kernel. The source and SM86 instruction schedule are pinned
// separately from this implementation. The original PyTorch license applies
// to its LayerNorm implementation.
// https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
// https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/cuda/TensorCompare.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Moments {
  float mean;
  float m2;
};

template <int Count>
__device__ __forceinline__ Moments observe(Moments prior, float x) {
  const float delta = __fsub_rn(x, prior.mean);
  const float mean = __fmaf_rn(delta, 1.0f / Count, prior.mean);
  return {mean, __fmaf_rn(delta, __fsub_rn(x, mean), prior.m2)};
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
  return {__shfl_down_sync(0xffffffffu, value.mean, Delta, 8),
          __shfl_down_sync(0xffffffffu, value.m2, Delta, 8)};
}

__device__ __forceinline__ float affine(float x, float gamma, float beta,
                                        float mean, float rstd, bool activate) {
  const float norm = __fmul_rn(rstd, __fsub_rn(x, mean));
  const float y = __fmaf_rn(gamma, norm, beta);
  // ATen CUDA relu uses clamp_min: pass NaNs through, then CUDA max.
  return activate && !isnan(y) ? ::max(y, 0.0f) : y;
}

constexpr int block_threads = 128;
constexpr int group_size = 8;
constexpr int block_rows = block_threads / group_size;

__global__ void radial_norm_kernel(const float4* x, const float4* gamma,
                                   const float4* beta, float4* y, float* mean,
                                   float* rstd, uint32_t rows, float eps,
                                   bool activate) {
  const unsigned group_lane = threadIdx.x % group_size;
  const unsigned row = blockIdx.x * block_rows + threadIdx.x / group_size;
  const unsigned offset = row * group_size + group_lane;
  const bool active = row < rows;
  // Tail rows still take part in warp shuffles.
  const float4 value = active ? x[offset] : make_float4(0, 0, 0, 0);
  Moments moments = observe<1>({0.0f, 0.0f}, value.x);
  moments = observe<2>(moments, value.y);
  moments = observe<3>(moments, value.z);
  moments = observe<4>(moments, value.w);

  // In stock, offsets 16 and 8 see empty peers. Then offsets 4, 2,
  // and 1 join the eight live vectors. The two inter-warp merges see
  // empty peers again. Each live group uses the same float operations.
  const Moments empty{0.0f, 0.0f};
  moments = combine<4, 0>(moments, empty);
  moments = combine<4, 0>(moments, empty);
  moments = combine<4, 4>(moments, shuffle_down<4>(moments));
  moments = combine<8, 8>(moments, shuffle_down<2>(moments));
  moments = combine<16, 16>(moments, shuffle_down<1>(moments));
  moments = combine<32, 0>(moments, empty);
  moments = combine<32, 0>(moments, empty);
  const float row_mean = __shfl_sync(0xffffffffu, moments.mean, 0, group_size);
  const float row_m2 = __shfl_sync(0xffffffffu, moments.m2, 0, group_size);
  const float inverse = rsqrtf(__fadd_rn(__fmul_rn(row_m2, 0.03125f), eps));

  if (active) {
    const float4 scale = gamma[group_lane];
    const float4 shift = beta[group_lane];
    y[offset] = make_float4(affine(value.x, scale.x, shift.x, row_mean, inverse, activate),
                            affine(value.y, scale.y, shift.y, row_mean, inverse, activate),
                            affine(value.z, scale.z, shift.z, row_mean, inverse, activate),
                            affine(value.w, scale.w, shift.w, row_mean, inverse, activate));
    if (group_lane == 0 && mean != nullptr) {
      mean[row] = row_mean;
      rstd[row] = inverse;
    }
  }
}

}  // namespace

extern "C" int radial_norm_launch(const void* x, const void* gamma, const void* beta,
                                  void* y, void* mean, void* rstd, uint32_t rows,
                                  float eps, void* stream, int activate) {
  radial_norm_kernel<<<(rows + block_rows - 1) / block_rows, block_threads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float4*>(x), static_cast<const float4*>(gamma),
      static_cast<const float4*>(beta), static_cast<float4*>(y),
      static_cast<float*>(mean), static_cast<float*>(rstd), rows, eps, activate != 0);
  return static_cast<int>(cudaPeekAtLastError());
}
