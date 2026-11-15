// kvserve_kernels.cu -- benchmarking harness around KVServe's quantize/dequantize CUDA kernels.
//
// The four __global__ kernels below (quantize_channel_kernel, quantize_token_kernel,
// dequantize_channel_kernel, dequantize_token_kernel) and their small device helpers
// (to_f32, rn, QuantParams, quant_params, quant_value) are VENDORED VERBATIM from
// KVServe (github.com/hpdps-group/KVServe), file
// kvserve_v1/compression/runtime/csrc/kvs_compress.cu, Apache-2.0 license, copyright
// contributors to the vLLM project. Not a single arithmetic line is changed. They are
// copied here, instead of imported, because in the original file they sit in an
// anonymous namespace (internal linkage) inside a 736-line translation unit whose
// surrounding code builds a flat int64 metadata blob describing vLLM paged-attention
// block layouts, slot arenas and an ANS codec workspace (kvs_compress.h /
// compression/runtime/{layout,pipeline,manager}.py) -- machinery this project has no
// use for and that pulls nothing from vLLM only because the metadata blob is prepared
// by kvserve_v1's own paged-cache bookkeeping, which is not needed to call the kernels
// on a single contiguous [T, H, D] tensor. Pulling just the kernels out, with the host
// launchers below written fresh for this benchmark, is the Apache-2.0-permitted use
// "for comparison" recorded in feasibility/results/C2_kvserve.md and DECISIONS.md D36.
//
// What the two quantize kernels actually compute (see feasibility/results/C2_kvserve.md
// and DECISIONS.md D36 for the full discussion):
//   quantize_channel_kernel: one (min, scale) pair per (head, dim) over the WHOLE token
//     axis passed in (no 32-token grouping -- coarser than KVPick's per-channel-per-32-
//     token-group scale). Output is one uint8 PER VALUE (not bit-packed): the byte
//     reduction in KVServe's shipped system comes from a separate entropy-codec stage
//     (nvCOMP ANS / LC) that this benchmark does not run.
//   quantize_token_kernel: one (min, scale) pair per token, over ALL heads and dims of
//     the given head range (coarser than KVPick's per-token-per-32-dim-group scale).
//
// `levels` is passed as 16 in this benchmark so the quantizer's code range (0..15)
// matches KVPick's 4-bit codes exactly, for a same-bit-depth kernel-speed comparison.
// The output being 1 B/value instead of 2 values/byte means KVServe's quantize kernel
// here writes 2x the bytes KVPick's does for the same input; this is reported as bytes
// processed (read + written), not "per the same wire format", and the byte-count
// asymmetry is called out next to every number derived from this file.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace kvserve_bench {

using bf16 = __nv_bfloat16;

// ---- vendored verbatim from kvs_compress.cu (KVServe, Apache-2.0) -----------------
namespace vendored {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kChannelTile = 8;
constexpr int kTokensPerBlock = kThreads / kWarpSize;

__device__ __forceinline__ float to_f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 rn(float x) { return __float2bfloat16_rn(x); }

struct QuantParams {
  bf16 minimum;
  bf16 maximum;
  bf16 scale;
  bf16 inverse;
};

__device__ __forceinline__ QuantParams quant_params(float mn, float mx, int levels) {
  const bf16 minimum = rn(mn);
  const bf16 maximum = rn(mx);
  float raw = to_f32(rn(to_f32(maximum) - to_f32(minimum)));
  if (raw < 1e-5f) raw = to_f32(rn(1e-5f));
  const bf16 scale = rn(raw / static_cast<float>(levels - 1));
  return {minimum, maximum, scale, rn(1.0f / to_f32(scale))};
}

__device__ __forceinline__ uint8_t quant_value(bf16 value, bf16 minimum, bf16 inverse, int levels) {
  float v = to_f32(value) - to_f32(minimum);
  v = to_f32(rn(v)) * to_f32(inverse);
  float q = rintf(to_f32(rn(v)));
  if (q < 0.0f) q = 0.0f;
  if (q > static_cast<float>(levels - 1)) q = static_cast<float>(levels - 1);
  return static_cast<uint8_t>(q);
}

// Eight adjacent channels share a block; 32 token lanes reduce each channel.
__global__ void quantize_channel_kernel(
    const bf16* __restrict__ src, uint8_t* __restrict__ out,
    bf16* __restrict__ aux_min, bf16* __restrict__ aux_scale,
    bf16* __restrict__ tmp_max, bf16* __restrict__ tmp_inv, int tokens,
    int heads, int head_start, int head_count, int head_size, int levels) {
  __shared__ float minima[kThreads / kWarpSize][kChannelTile];
  __shared__ float maxima[kThreads / kWarpSize][kChannelTile];
  const int c = threadIdx.x % kChannelTile;
  const int token_lane = threadIdx.x / kChannelTile;
  const int column = blockIdx.x * kChannelTile + c;
  const bool valid = column < head_count * head_size;
  const int h = head_start + column / head_size;
  const int d = column % head_size;
  float mn = INFINITY;
  float mx = -INFINITY;
  if (valid) {
    for (int t = token_lane; t < tokens; t += kThreads / kChannelTile) {
      const float v = to_f32(src[(static_cast<int64_t>(t) * heads + h) * head_size + d]);
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
  }
  for (int offset = kChannelTile; offset < kWarpSize; offset <<= 1) {
    mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, offset));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, offset));
  }
  if (threadIdx.x % kWarpSize < kChannelTile) {
    minima[threadIdx.x / kWarpSize][c] = mn;
    maxima[threadIdx.x / kWarpSize][c] = mx;
  }
  __syncthreads();
  if (threadIdx.x < kChannelTile && valid) {
    for (int warp = 1; warp < kThreads / kWarpSize; ++warp) {
      mn = fminf(mn, minima[warp][c]);
      mx = fmaxf(mx, maxima[warp][c]);
    }
    const QuantParams p = quant_params(mn, mx, levels);
    aux_min[column] = p.minimum;
    tmp_max[column] = p.maximum;
    aux_scale[column] = p.scale;
    tmp_inv[column] = p.inverse;
    minima[0][c] = to_f32(p.minimum);
    maxima[0][c] = to_f32(p.inverse);
  }
  __syncthreads();
  if (valid) {
    const bf16 minimum = rn(minima[0][c]);
    const bf16 inverse = rn(maxima[0][c]);
    for (int t = token_lane; t < tokens; t += kThreads / kChannelTile) {
      const int64_t idx = (static_cast<int64_t>(t) * heads + h) * head_size + d;
      out[idx] = quant_value(src[idx], minimum, inverse, levels);
    }
  }
}

// A warp cooperates on one token, reading consecutive head/dimension values.
__global__ void quantize_token_kernel(
    const bf16* __restrict__ src, uint8_t* __restrict__ out,
    bf16* __restrict__ aux_min, bf16* __restrict__ aux_scale,
    bf16* __restrict__ tmp_max, bf16* __restrict__ tmp_inv, int tokens,
    int heads, int head_start, int head_count, int head_size, int levels) {
  const int t = blockIdx.x * kTokensPerBlock + threadIdx.x / kWarpSize;
  if (t >= tokens) return;
  const int lane = threadIdx.x % kWarpSize;
  const int columns = head_count * head_size;
  const int64_t row_base = static_cast<int64_t>(t) * heads * head_size;
  const int first = head_start * head_size;
  float mn = INFINITY;
  float mx = -INFINITY;
  for (int c = lane; c < columns; c += kWarpSize) {
    const float v = to_f32(src[row_base + first + c]);
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, offset));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, offset));
  }
  const QuantParams p = quant_params(mn, mx, levels);
  if (lane == 0) {
    aux_min[t] = p.minimum;
    tmp_max[t] = p.maximum;
    aux_scale[t] = p.scale;
    tmp_inv[t] = p.inverse;
  }
  for (int c = lane; c < columns; c += kWarpSize) {
    const int64_t idx = row_base + first + c;
    out[idx] = quant_value(src[idx], p.minimum, p.inverse, levels);
  }
}

__global__ void dequantize_channel_kernel(const uint8_t* __restrict__ u8,
                                          bf16* __restrict__ out,
                                          const bf16* __restrict__ aux_min,
                                          const bf16* __restrict__ aux_scale,
                                          int tokens, int heads, int head_start,
                                          int head_count, int head_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= head_count * head_size) return;
  const int h = head_start + column / head_size;
  const int d = column % head_size;
  const bf16 mn = aux_min[column];
  const bf16 scale = aux_scale[column];
  for (int t = 0; t < tokens; ++t) {
    const int64_t idx = (static_cast<int64_t>(t) * heads + h) * head_size + d;
    float v = static_cast<float>(u8[idx]);
    v = to_f32(rn(v * to_f32(scale)));
    out[idx] = rn(v + to_f32(mn));
  }
}

__global__ void dequantize_token_kernel(const uint8_t* __restrict__ u8,
                                        bf16* __restrict__ out,
                                        const bf16* __restrict__ aux_min,
                                        const bf16* __restrict__ aux_scale,
                                        int tokens, int heads, int head_start,
                                        int head_count, int head_size) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens) return;
  const bf16 mn = aux_min[t];
  const bf16 scale = aux_scale[t];
  const int64_t row_base = static_cast<int64_t>(t) * heads * head_size;
  for (int h = head_start; h < head_start + head_count; ++h) {
    for (int d = 0; d < head_size; ++d) {
      const int64_t idx = row_base + static_cast<int64_t>(h) * head_size + d;
      float v = static_cast<float>(u8[idx]);
      v = to_f32(rn(v * to_f32(scale)));
      out[idx] = rn(v + to_f32(mn));
    }
  }
}

}  // namespace vendored

// ---- host launchers (written for this benchmark; not part of KVServe) -------------
static void check_input(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 3,
              "x must be contiguous bf16 CUDA [T, H, D]");
}
static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Returns (payload_u8 [T,H,D], aux_min [H*D], aux_scale [H*D]) -- one scale per (head, dim)
// over the WHOLE T axis (see the file header: no 32-token grouping in KVServe's kernel).
std::vector<at::Tensor> quantize_channel(const at::Tensor& x, int64_t levels) {
  check_input(x);
  const int64_t T = x.size(0), H = x.size(1), D = x.size(2);
  const c10::cuda::CUDAGuard guard(x.device()); auto st = at::cuda::getCurrentCUDAStream();
  auto payload = at::empty({T, H, D}, x.options().dtype(at::kByte));
  auto aux_min = at::empty({H * D}, x.options()); auto aux_scale = at::empty({H * D}, x.options());
  auto tmp_max = at::empty({H * D}, x.options()); auto tmp_inv = at::empty({H * D}, x.options());
  const int64_t blocks = cdiv(H * D, vendored::kChannelTile);
  vendored::quantize_channel_kernel<<<static_cast<unsigned>(blocks), vendored::kThreads, 0, st>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), payload.data_ptr<uint8_t>(),
      reinterpret_cast<bf16*>(aux_min.data_ptr()), reinterpret_cast<bf16*>(aux_scale.data_ptr()),
      reinterpret_cast<bf16*>(tmp_max.data_ptr()), reinterpret_cast<bf16*>(tmp_inv.data_ptr()),
      static_cast<int>(T), static_cast<int>(H), 0, static_cast<int>(H), static_cast<int>(D), static_cast<int>(levels));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {payload, aux_min, aux_scale};
}

// Returns (payload_u8 [T,H,D], aux_min [T], aux_scale [T]) -- one scale per token over
// ALL heads and dims (see the file header: no 32-dim grouping in KVServe's kernel).
std::vector<at::Tensor> quantize_token(const at::Tensor& x, int64_t levels) {
  check_input(x);
  const int64_t T = x.size(0), H = x.size(1), D = x.size(2);
  const c10::cuda::CUDAGuard guard(x.device()); auto st = at::cuda::getCurrentCUDAStream();
  auto payload = at::empty({T, H, D}, x.options().dtype(at::kByte));
  auto aux_min = at::empty({T}, x.options()); auto aux_scale = at::empty({T}, x.options());
  auto tmp_max = at::empty({T}, x.options()); auto tmp_inv = at::empty({T}, x.options());
  const int64_t blocks = cdiv(T, vendored::kTokensPerBlock);
  vendored::quantize_token_kernel<<<static_cast<unsigned>(blocks), vendored::kThreads, 0, st>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), payload.data_ptr<uint8_t>(),
      reinterpret_cast<bf16*>(aux_min.data_ptr()), reinterpret_cast<bf16*>(aux_scale.data_ptr()),
      reinterpret_cast<bf16*>(tmp_max.data_ptr()), reinterpret_cast<bf16*>(tmp_inv.data_ptr()),
      static_cast<int>(T), static_cast<int>(H), 0, static_cast<int>(H), static_cast<int>(D), static_cast<int>(levels));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {payload, aux_min, aux_scale};
}

at::Tensor dequantize_channel(const at::Tensor& payload, const at::Tensor& aux_min, const at::Tensor& aux_scale) {
  const int64_t T = payload.size(0), H = payload.size(1), D = payload.size(2);
  const c10::cuda::CUDAGuard guard(payload.device()); auto st = at::cuda::getCurrentCUDAStream();
  auto out = at::empty({T, H, D}, aux_min.options());
  const int64_t blocks = cdiv(H * D, 256);
  vendored::dequantize_channel_kernel<<<static_cast<unsigned>(blocks), 256, 0, st>>>(
      payload.data_ptr<uint8_t>(), reinterpret_cast<bf16*>(out.data_ptr()),
      reinterpret_cast<const bf16*>(aux_min.data_ptr()), reinterpret_cast<const bf16*>(aux_scale.data_ptr()),
      static_cast<int>(T), static_cast<int>(H), 0, static_cast<int>(H), static_cast<int>(D));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor dequantize_token(const at::Tensor& payload, const at::Tensor& aux_min, const at::Tensor& aux_scale) {
  const int64_t T = payload.size(0), H = payload.size(1), D = payload.size(2);
  const c10::cuda::CUDAGuard guard(payload.device()); auto st = at::cuda::getCurrentCUDAStream();
  auto out = at::empty({T, H, D}, aux_min.options());
  const int64_t blocks = cdiv(T, 256);
  vendored::dequantize_token_kernel<<<static_cast<unsigned>(blocks), 256, 0, st>>>(
      payload.data_ptr<uint8_t>(), reinterpret_cast<bf16*>(out.data_ptr()),
      reinterpret_cast<const bf16*>(aux_min.data_ptr()), reinterpret_cast<const bf16*>(aux_scale.data_ptr()),
      static_cast<int>(T), static_cast<int>(H), 0, static_cast<int>(H), static_cast<int>(D));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace kvserve_bench

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Benchmark harness around KVServe's quantize_channel_kernel / quantize_token_kernel "
            "(vendored verbatim, Apache-2.0; see file header and DECISIONS.md D36)";
  m.def("quantize_channel", &kvserve_bench::quantize_channel, py::arg("x"), py::arg("levels") = 16);
  m.def("quantize_token", &kvserve_bench::quantize_token, py::arg("x"), py::arg("levels") = 16);
  m.def("dequantize_channel", &kvserve_bench::dequantize_channel, py::arg("payload"), py::arg("aux_min"), py::arg("aux_scale"));
  m.def("dequantize_token", &kvserve_bench::dequantize_token, py::arg("payload"), py::arg("aux_min"), py::arg("aux_scale"));
}
