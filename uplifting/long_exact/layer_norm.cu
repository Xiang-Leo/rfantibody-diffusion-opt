// Contiguous width-32/64/128 specialization of the FP32 Welford schedule
// in PyTorch v2.3.1, using the rounded arithmetic tested in P7.
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

template <int Group, int Offset>
__device__ __forceinline__ Moments merge(Moments value) {
  if (Offset >= Group) {
    // A live lane's peer was empty in the stock warp.
    return combine<4, 0>(value, {0.0f, 0.0f});
  }
  constexpr int count = Offset < Group ? 2 * Group / Offset : 4;
  Moments peer{__shfl_down_sync(0xffffffffu, value.mean, Offset, Group),
               __shfl_down_sync(0xffffffffu, value.m2, Offset, Group)};
  return combine<count, count>(value, peer);
}

__device__ __forceinline__ float affine(float x, float gamma, float beta,
                                        float mean, float rstd) {
  const float norm = __fmul_rn(rstd, __fsub_rn(x, mean));
  return __fmaf_rn(gamma, norm, beta);
}

constexpr unsigned block_threads = 128;

template <int Width>
__global__ void layer_norm_kernel(const float4* __restrict__ x,
                                  const float4* __restrict__ gamma,
                                  const float4* __restrict__ beta,
                                  float4* __restrict__ y,
                                  float* __restrict__ mean,
                                  float* __restrict__ rstd,
                                  uint32_t rows, float eps) {
  constexpr unsigned group = Width / 4;
  constexpr unsigned rows_per_block = block_threads / group;
  const unsigned lane = threadIdx.x % group;
  const unsigned row = blockIdx.x * rows_per_block + threadIdx.x / group;
  const unsigned offset = row * group + lane;
  const bool active = row < rows;
  // Inactive groups in the tail still participate in warp shuffles.
  const float4 value = active ? x[offset] : make_float4(0, 0, 0, 0);
  Moments moments = observe<1>({0.0f, 0.0f}, value.x);
  moments = observe<2>(moments, value.y);
  moments = observe<3>(moments, value.z);
  moments = observe<4>(moments, value.w);

  moments = merge<group, 16>(moments);
  moments = merge<group, 8>(moments);
  moments = merge<group, 4>(moments);
  moments = merge<group, 2>(moments);
  moments = merge<group, 1>(moments);
  if (lane == 0) {
    // Stock assigned three additional empty warps to the row.
    const Moments empty{0.0f, 0.0f};
    moments = combine<Width, 0>(moments, empty);
    moments = combine<Width, 0>(moments, empty);
  }
  const float row_mean = __shfl_sync(0xffffffffu, moments.mean, 0, group);
  const float row_m2 = __shfl_sync(0xffffffffu, moments.m2, 0, group);
  const float inverse = rsqrtf(__fadd_rn(__fmul_rn(row_m2, 1.0f / Width), eps));
  if (active) {
    const float4 scale = gamma[lane];
    const float4 shift = beta[lane];
    y[offset] = make_float4(
        affine(value.x, scale.x, shift.x, row_mean, inverse),
        affine(value.y, scale.y, shift.y, row_mean, inverse),
        affine(value.z, scale.z, shift.z, row_mean, inverse),
        affine(value.w, scale.w, shift.w, row_mean, inverse));
    if (lane == 0 && mean != nullptr) {
      mean[row] = row_mean;
      rstd[row] = inverse;
    }
  }
}

template <int Width>
void launch(const void* x, const void* gamma, const void* beta,
            void* y, void* mean, void* rstd, uint32_t rows,
            float eps, cudaStream_t stream) {
  constexpr unsigned rows_per_block = block_threads / (Width / 4);
  layer_norm_kernel<Width><<<(rows + rows_per_block - 1) / rows_per_block, block_threads, 0, stream>>>(
      static_cast<const float4*>(x), static_cast<const float4*>(gamma),
      static_cast<const float4*>(beta), static_cast<float4*>(y),
      static_cast<float*>(mean), static_cast<float*>(rstd), rows, eps);
}

}  // namespace

extern "C" int layer_norm_launch(const void* x, const void* gamma, const void* beta,
                                 void* y, void* mean, void* rstd, uint32_t rows,
                                 uint32_t width, float eps, void* stream_ptr) {
  if (rows == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  switch (width) {
    case 32: launch<32>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    case 64: launch<64>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    case 128: launch<128>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
