// Materialize the FP32 pair residual and normalize the same rounded values.
// Welford, inverse standard deviation and affine operations are the verified
// width-128 schedule from layer_norm.cu and PyTorch v2.3.1.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Moments {
  float mean;
  float m2;
};

template <int Count>
__device__ __forceinline__ Moments observe(Moments current, float x) {
  const float delta = __fsub_rn(x, current.mean);
  const float mean = __fmaf_rn(delta, 1.0f / Count, current.mean);
  return {mean, __fmaf_rn(delta, __fsub_rn(x, mean), current.m2)};
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

template <int Offset>
__device__ __forceinline__ Moments warp_peer(Moments value) {
  return {__shfl_down_sync(0xffffffffu, value.mean, Offset),
          __shfl_down_sync(0xffffffffu, value.m2, Offset)};
}

__device__ __forceinline__ float affine(float value, float scale, float shift,
                                        float mean, float inverse) {
  const float centered = __fmul_rn(inverse, __fsub_rn(value, mean));
  return __fmaf_rn(scale, centered, shift);
}

constexpr unsigned threads = 128;
constexpr unsigned warp = 32;
constexpr unsigned rows_per_block = threads / warp;

template <bool Transposed>
__global__ void residual_norm(const float4* __restrict__ lhs,
                              const float4* __restrict__ rhs,
                              const float4* __restrict__ gamma,
                              const float4* __restrict__ beta,
                              float4* __restrict__ residual,
                              float4* __restrict__ normalized,
                              float* __restrict__ mean_out,
                              float* __restrict__ inverse_out,
                              unsigned length, float eps) {
  const unsigned lane = threadIdx.x % warp;
  const unsigned row = blockIdx.x * rows_per_block + threadIdx.x / warp;
  if (row >= length * length) {
    return;  // Every lane of the warp takes the same branch.
  }
  const unsigned offset = row * warp + lane;
  const unsigned other_row = Transposed ? (row % length) * length + row / length : row;
  const float4 a = lhs[offset];
  const float4 b = rhs[other_row * warp + lane];
  const float4 value = make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                                   __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
  residual[offset] = value;

  Moments moments = observe<1>({0.0f, 0.0f}, value.x);
  moments = observe<2>(moments, value.y);
  moments = observe<3>(moments, value.z);
  moments = observe<4>(moments, value.w);
  moments = combine<4, 4>(moments, warp_peer<16>(moments));
  moments = combine<8, 8>(moments, warp_peer<8>(moments));
  moments = combine<16, 16>(moments, warp_peer<4>(moments));
  moments = combine<32, 32>(moments, warp_peer<2>(moments));
  moments = combine<64, 64>(moments, warp_peer<1>(moments));
  if (lane == 0) {
    const Moments empty{0.0f, 0.0f};
    moments = combine<128, 0>(moments, empty);
    moments = combine<128, 0>(moments, empty);
  }
  const float mean = __shfl_sync(0xffffffffu, moments.mean, 0);
  const float m2 = __shfl_sync(0xffffffffu, moments.m2, 0);
  const float inverse = rsqrtf(__fadd_rn(__fmul_rn(m2, 1.0f / 128.0f), eps));
  const float4 scale = gamma[lane];
  const float4 shift = beta[lane];
  normalized[offset] = make_float4(
      affine(value.x, scale.x, shift.x, mean, inverse),
      affine(value.y, scale.y, shift.y, mean, inverse),
      affine(value.z, scale.z, shift.z, mean, inverse),
      affine(value.w, scale.w, shift.w, mean, inverse));
  if (lane == 0 && mean_out != nullptr) {
    mean_out[row] = mean;
    inverse_out[row] = inverse;
  }
}

}  // namespace

extern "C" int rfd_residual_norm_launch(const void* lhs, const void* rhs,
                                         const void* gamma, const void* beta,
                                         void* residual, void* normalized,
                                         void* mean, void* inverse,
                                         uint32_t length, float eps, int transposed,
                                         void* stream_ptr) {
  if (length == 0 || length > 5792) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto a = static_cast<const float4*>(lhs);
  const auto b = static_cast<const float4*>(rhs);
  const auto g = static_cast<const float4*>(gamma);
  const auto h = static_cast<const float4*>(beta);
  auto r = static_cast<float4*>(residual);
  auto n = static_cast<float4*>(normalized);
  auto m = static_cast<float*>(mean);
  auto s = static_cast<float*>(inverse);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const unsigned blocks = (length * length + rows_per_block - 1) / rows_per_block;
  if (transposed) {
    residual_norm<true><<<blocks, threads, 0, stream>>>(a, b, g, h, r, n, m, s, length, eps);
  } else {
    residual_norm<false><<<blocks, threads, 0, stream>>>(a, b, g, h, r, n, m, s, length, eps);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
