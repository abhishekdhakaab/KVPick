// kvq_kernels.cu -- CUDA kernels for the KV4 wire format (see kvq.h for layout and arithmetic).
//
// Versions (selected by the `version` argument so each optimisation is measured separately):
//   v0  naive: one thread per output byte, scales from a separate one-thread-per-group kernel with global-memory loops
//   v1  K: one 256-thread block per (layer, 32-token group, 4 heads); each thread owns 2 adjacent channels and keeps the
//          32 tokens in registers (one global read), writes scale/zero as half2 and one packed byte per token.
//       V: one warp per (token, head) row; each lane owns 4 adjacent dims; group min/max via 8-lane shuffles.
//   v2  the v1 bodies fused into ONE launch covering all K tiles and all V rows of the layer batch (the transfer-buffer
//       producer), so the whole cache is compressed with a single kernel launch.
// Dequantize mirrors v0/v1; v2 dequantize == v1 (the compiler already reaches the bandwidth ceiling there, G4).
// Numerics: fp32, no FMA (--fmad=false and explicit __fmul_rn/__fadd_rn), rintf = round half to even, f16/bf16 RN.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "kvq.h"

namespace kvq {

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
// zf = float(zero16), inv = correctly rounded 1/float(scale16)
__device__ __forceinline__ void make_params(float mn, float mx, __half& z16, __half& s16, float& zf, float& inv) {
  z16 = __float2half_rn(mn); zf = __half2float(z16);
  s16 = __float2half_rn(scale_from_range(mx, zf)); inv = inv_scale(__half2float(s16));
}
__device__ __forceinline__ bf16 dq_bf16(uint8_t c, float sf, float zf) {
  float p = __fmul_rn(static_cast<float>(c), sf);
  float y = __fadd_rn(p, zf);
  return __float2bfloat16_rn(y);
}

struct Dims { int64_t L, T, H, Tg; };

// =====================================================================================================================
// v0: naive
// =====================================================================================================================
// K scales: one thread per (l, g, h, d)
__global__ void k4_scales_v0(const bf16* __restrict__ k, __half* __restrict__ scale, __half* __restrict__ zero, Dims dm) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = dm.L * dm.Tg * dm.H * HEAD_DIM;
  if (i >= n) return;
  const int d = static_cast<int>(i % HEAD_DIM); int64_t r = i / HEAD_DIM;
  const int64_t h = r % dm.H; r /= dm.H; const int64_t g = r % dm.Tg; const int64_t l = r / dm.Tg;
  float mn = INFINITY, mx = -INFINITY;
  const int64_t t1 = (dm.T < (g + 1) * GROUP) ? dm.T : (g + 1) * GROUP;
  for (int64_t t = g * GROUP; t < t1; ++t) {
    float x = bf2f(k[((l * dm.T + t) * dm.H + h) * HEAD_DIM + d]); mn = fminf(mn, x); mx = fmaxf(mx, x);
  }
  __half z16, s16; float zf, inv; make_params(mn, mx, z16, s16, zf, inv);
  scale[i] = s16; zero[i] = z16;
}
// K codes: one thread per output byte (l, t, h, d/2)
__global__ void k4_codes_v0(const bf16* __restrict__ k, const __half* __restrict__ scale, const __half* __restrict__ zero,
                            uint8_t* __restrict__ payload, Dims dm) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = dm.L * dm.T * dm.H * (HEAD_DIM / 2);
  if (i >= n) return;
  const int j = static_cast<int>(i % (HEAD_DIM / 2)); int64_t r = i / (HEAD_DIM / 2);
  const int64_t h = r % dm.H; r /= dm.H; const int64_t t = r % dm.T; const int64_t l = r / dm.T;
  const int64_t m = ((l * dm.Tg + t / GROUP) * dm.H + h) * HEAD_DIM + 2 * j;
  const int64_t e = ((l * dm.T + t) * dm.H + h) * HEAD_DIM + 2 * j;
  uint8_t c0 = quant_code(bf2f(k[e]), __half2float(zero[m]), inv_scale(__half2float(scale[m])));
  uint8_t c1 = quant_code(bf2f(k[e + 1]), __half2float(zero[m + 1]), inv_scale(__half2float(scale[m + 1])));
  payload[i] = pack2(c0, c1);
}
// V scales: one thread per (row, group)
__global__ void v4_scales_v0(const bf16* __restrict__ v, __half* __restrict__ scale, __half* __restrict__ zero, int64_t rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * V_GROUPS) return;
  const int64_t r = i / V_GROUPS; const int g = static_cast<int>(i % V_GROUPS);
  float mn = INFINITY, mx = -INFINITY;
  for (int d = g * GROUP; d < (g + 1) * GROUP; ++d) { float x = bf2f(v[r * HEAD_DIM + d]); mn = fminf(mn, x); mx = fmaxf(mx, x); }
  __half z16, s16; float zf, inv; make_params(mn, mx, z16, s16, zf, inv);
  scale[i] = s16; zero[i] = z16;
}
__global__ void v4_codes_v0(const bf16* __restrict__ v, const __half* __restrict__ scale, const __half* __restrict__ zero,
                            uint8_t* __restrict__ payload, int64_t rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * (HEAD_DIM / 2)) return;
  const int64_t r = i / (HEAD_DIM / 2); const int j = static_cast<int>(i % (HEAD_DIM / 2));
  const int g = (2 * j) / GROUP; const int64_t m = r * V_GROUPS + g;
  const float zf = __half2float(zero[m]), inv = inv_scale(__half2float(scale[m]));
  payload[i] = pack2(quant_code(bf2f(v[r * HEAD_DIM + 2 * j]), zf, inv), quant_code(bf2f(v[r * HEAD_DIM + 2 * j + 1]), zf, inv));
}
__global__ void k4_dequant_v0(const uint8_t* __restrict__ payload, const __half* __restrict__ scale, const __half* __restrict__ zero,
                              bf16* __restrict__ k, Dims dm) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t n = dm.L * dm.T * dm.H * (HEAD_DIM / 2);
  if (i >= n) return;
  const int j = static_cast<int>(i % (HEAD_DIM / 2)); int64_t r = i / (HEAD_DIM / 2);
  const int64_t h = r % dm.H; r /= dm.H; const int64_t t = r % dm.T; const int64_t l = r / dm.T;
  const int64_t m = ((l * dm.Tg + t / GROUP) * dm.H + h) * HEAD_DIM + 2 * j;
  const int64_t e = ((l * dm.T + t) * dm.H + h) * HEAD_DIM + 2 * j;
  const uint8_t b = payload[i];
  k[e] = dq_bf16(b & 0xF, __half2float(scale[m]), __half2float(zero[m]));
  k[e + 1] = dq_bf16(b >> 4, __half2float(scale[m + 1]), __half2float(zero[m + 1]));
}
__global__ void v4_dequant_v0(const uint8_t* __restrict__ payload, const __half* __restrict__ scale, const __half* __restrict__ zero,
                              bf16* __restrict__ v, int64_t rows) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * (HEAD_DIM / 2)) return;
  const int64_t r = i / (HEAD_DIM / 2); const int j = static_cast<int>(i % (HEAD_DIM / 2));
  const int64_t m = r * V_GROUPS + (2 * j) / GROUP;
  const float zf = __half2float(zero[m]), sf = __half2float(scale[m]); const uint8_t b = payload[i];
  v[r * HEAD_DIM + 2 * j] = dq_bf16(b & 0xF, sf, zf); v[r * HEAD_DIM + 2 * j + 1] = dq_bf16(b >> 4, sf, zf);
}

// =====================================================================================================================
// v1 / v2 bodies (device functions, so v2 can fuse them into one launch)
// =====================================================================================================================
constexpr int THREADS = 256;
constexpr int HEADS_PER_BLOCK = 4;                 // K tile block covers 4 heads x 32 tokens x 128 dims
constexpr int THREADS_PER_HEAD = HEAD_DIM / 2;     // 64 threads, 2 adjacent dims each
constexpr int ROWS_PER_BLOCK = THREADS / 32;       // V v1: one warp per row -> 8 rows per block
constexpr int ROWS_PER_BLOCK_V2 = 2 * ROWS_PER_BLOCK; // V v2: two rows per warp -> 16 rows per block

// K tile: block index -> (l, g, head quad). Each thread: head hh = tid/64, dims (2j, 2j+1) with j = tid%64.
__device__ __forceinline__ void k4_tile_quant(const bf16* __restrict__ k, uint8_t* __restrict__ payload,
                                              __half* __restrict__ scale, __half* __restrict__ zero, Dims dm, int64_t tile) {
  const int64_t nq = (dm.H + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
  const int64_t q = tile % nq; int64_t r = tile / nq; const int64_t g = r % dm.Tg; const int64_t l = r / dm.Tg;
  const int hh = threadIdx.x / THREADS_PER_HEAD, j = threadIdx.x % THREADS_PER_HEAD;
  const int64_t h = q * HEADS_PER_BLOCK + hh;
  if (h >= dm.H) return;                                        // H not a multiple of 4: idle threads (no shared-memory sync used)
  const int64_t t0 = g * GROUP; const int nt = static_cast<int>((dm.T - t0 < GROUP) ? (dm.T - t0) : GROUP);
  bf162 vals[GROUP];
  float mn0 = INFINITY, mx0 = -INFINITY, mn1 = INFINITY, mx1 = -INFINITY;
  const bf162* base = reinterpret_cast<const bf162*>(k + ((l * dm.T + t0) * dm.H + h) * HEAD_DIM) + j;
  const int64_t row_stride2 = dm.H * HEAD_DIM / 2;              // in bf162 units
#pragma unroll
  for (int tt = 0; tt < GROUP; ++tt) {
    if (tt < nt) {
      vals[tt] = base[tt * row_stride2];
      const float x0 = __low2float(vals[tt]), x1 = __high2float(vals[tt]);
      mn0 = fminf(mn0, x0); mx0 = fmaxf(mx0, x0); mn1 = fminf(mn1, x1); mx1 = fmaxf(mx1, x1);
    }
  }
  __half z0, s0, z1, s1; float zf0, inv0, zf1, inv1;
  make_params(mn0, mx0, z0, s0, zf0, inv0); make_params(mn1, mx1, z1, s1, zf1, inv1);
  const int64_t m = ((l * dm.Tg + g) * dm.H + h) * HEAD_DIM + 2 * j;
  *reinterpret_cast<__half2*>(scale + m) = __halves2half2(s0, s1);
  *reinterpret_cast<__half2*>(zero + m) = __halves2half2(z0, z1);
  uint8_t* out = payload + ((l * dm.T + t0) * dm.H + h) * (HEAD_DIM / 2) + j;
  const int64_t out_stride = dm.H * (HEAD_DIM / 2);
#pragma unroll
  for (int tt = 0; tt < GROUP; ++tt) {
    if (tt < nt) {
      out[tt * out_stride] = pack2(quant_code(__low2float(vals[tt]), zf0, inv0), quant_code(__high2float(vals[tt]), zf1, inv1));
    }
  }
}

__device__ __forceinline__ void k4_tile_dequant(const uint8_t* __restrict__ payload, const __half* __restrict__ scale,
                                                const __half* __restrict__ zero, bf16* __restrict__ k, Dims dm, int64_t tile) {
  const int64_t nq = (dm.H + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK;
  const int64_t q = tile % nq; int64_t r = tile / nq; const int64_t g = r % dm.Tg; const int64_t l = r / dm.Tg;
  const int hh = threadIdx.x / THREADS_PER_HEAD, j = threadIdx.x % THREADS_PER_HEAD;
  const int64_t h = q * HEADS_PER_BLOCK + hh;
  if (h >= dm.H) return;
  const int64_t t0 = g * GROUP; const int nt = static_cast<int>((dm.T - t0 < GROUP) ? (dm.T - t0) : GROUP);
  const int64_t m = ((l * dm.Tg + g) * dm.H + h) * HEAD_DIM + 2 * j;
  const __half2 s2 = *reinterpret_cast<const __half2*>(scale + m), z2 = *reinterpret_cast<const __half2*>(zero + m);
  const float sf0 = __low2float(s2), sf1 = __high2float(s2), zf0 = __low2float(z2), zf1 = __high2float(z2);
  const uint8_t* in = payload + ((l * dm.T + t0) * dm.H + h) * (HEAD_DIM / 2) + j;
  bf162* out = reinterpret_cast<bf162*>(k + ((l * dm.T + t0) * dm.H + h) * HEAD_DIM) + j;
  const int64_t in_stride = dm.H * (HEAD_DIM / 2), out_stride2 = dm.H * HEAD_DIM / 2;
#pragma unroll 8
  for (int tt = 0; tt < nt; ++tt) {
    const uint8_t b = in[tt * in_stride];
    out[tt * out_stride2] = __halves2bfloat162(dq_bf16(b & 0xF, sf0, zf0), dq_bf16(b >> 4, sf1, zf1));
  }
}

// V row: warp per row; lane owns dims 4*lane .. 4*lane+3; lanes 8g..8g+7 form group g.
__device__ __forceinline__ void v4_row_quant(const bf16* __restrict__ v, uint8_t* __restrict__ payload,
                                             __half* __restrict__ scale, __half* __restrict__ zero, int64_t rows, int64_t row) {
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;
  const uint2 raw = *reinterpret_cast<const uint2*>(v + row * HEAD_DIM + 4 * lane);     // 4 bf16 = 8 B
  const bf162 p0 = *reinterpret_cast<const bf162*>(&raw.x), p1 = *reinterpret_cast<const bf162*>(&raw.y);
  const float x0 = __low2float(p0), x1 = __high2float(p0), x2 = __low2float(p1), x3 = __high2float(p1);
  float mn = fminf(fminf(x0, x1), fminf(x2, x3)), mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) {                                                     // reduce within 8-lane groups
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  __half z16, s16; float zf, inv; make_params(mn, mx, z16, s16, zf, inv);
  const int g = lane / 8;
  if (lane % 8 == 0) { scale[row * V_GROUPS + g] = s16; zero[row * V_GROUPS + g] = z16; }
  const uint8_t b0 = pack2(quant_code(x0, zf, inv), quant_code(x1, zf, inv));
  const uint8_t b1 = pack2(quant_code(x2, zf, inv), quant_code(x3, zf, inv));
  *reinterpret_cast<uint16_t*>(payload + row * (HEAD_DIM / 2) + 2 * lane) = static_cast<uint16_t>(b0 | (b1 << 8));
}

__device__ __forceinline__ void v4_row_dequant(const uint8_t* __restrict__ payload, const __half* __restrict__ scale,
                                               const __half* __restrict__ zero, bf16* __restrict__ v, int64_t rows, int64_t row) {
  const int lane = threadIdx.x % 32;
  if (row >= rows) return;
  const int g = lane / 8;
  const float sf = __half2float(scale[row * V_GROUPS + g]), zf = __half2float(zero[row * V_GROUPS + g]);
  const uint16_t w = *reinterpret_cast<const uint16_t*>(payload + row * (HEAD_DIM / 2) + 2 * lane);
  const uint8_t b0 = w & 0xFF, b1 = w >> 8;
  bf162 o0 = __halves2bfloat162(dq_bf16(b0 & 0xF, sf, zf), dq_bf16(b0 >> 4, sf, zf));
  bf162 o1 = __halves2bfloat162(dq_bf16(b1 & 0xF, sf, zf), dq_bf16(b1 >> 4, sf, zf));
  uint2 raw; raw.x = *reinterpret_cast<uint32_t*>(&o0); raw.y = *reinterpret_cast<uint32_t*>(&o1);
  *reinterpret_cast<uint2*>(v + row * HEAD_DIM + 4 * lane) = raw;
}


// V row pair (v2 path): warp handles two rows (lanes 0-15 row A, 16-31 row B); lane owns 8 adjacent dims (16 B load);
// a 32-dim group = 4 lanes -> two shuffle steps. Twice the bytes in flight per warp compared with v4_row_quant.
__device__ __forceinline__ void v4_rowpair_quant(const bf16* __restrict__ v, uint8_t* __restrict__ payload,
                                                 __half* __restrict__ scale, __half* __restrict__ zero, int64_t rows, int64_t row0) {
  const int lane = threadIdx.x % 32;
  const int64_t row = row0 + (lane >> 4);
  if (row >= rows) return;                                        // rows are processed in pairs; the second may be past the end
  const int c = lane & 15;                                        // 16 lanes x 8 dims = 128
  const uint4 raw = *reinterpret_cast<const uint4*>(v + row * HEAD_DIM + 8 * c);
  const bf162 p0 = *reinterpret_cast<const bf162*>(&raw.x), p1 = *reinterpret_cast<const bf162*>(&raw.y);
  const bf162 p2 = *reinterpret_cast<const bf162*>(&raw.z), p3 = *reinterpret_cast<const bf162*>(&raw.w);
  float xs[8] = {__low2float(p0), __high2float(p0), __low2float(p1), __high2float(p1), __low2float(p2), __high2float(p2), __low2float(p3), __high2float(p3)};
  float mn = xs[0], mx = xs[0];
#pragma unroll
  for (int i = 1; i < 8; ++i) { mn = fminf(mn, xs[i]); mx = fmaxf(mx, xs[i]); }
  // the row's 4 lanes of this group are lanes (c & ~3) .. +3 within the same 16-lane half: xor 1, 2 stays inside
  mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
  mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, 2)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  __half z16, s16; float zf, inv; make_params(mn, mx, z16, s16, zf, inv);
  const int g = c >> 2;
  if ((c & 3) == 0) { scale[row * V_GROUPS + g] = s16; zero[row * V_GROUPS + g] = z16; }
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < 8; i += 2) w |= static_cast<uint32_t>(pack2(quant_code(xs[i], zf, inv), quant_code(xs[i + 1], zf, inv))) << (4 * i);
  *reinterpret_cast<uint32_t*>(payload + row * (HEAD_DIM / 2) + 4 * c) = w;
}

// v1 launches
__global__ void k4_quant_v1(const bf16* __restrict__ k, uint8_t* __restrict__ p, __half* __restrict__ s, __half* __restrict__ z, Dims dm) {
  k4_tile_quant(k, p, s, z, dm, blockIdx.x);
}
__global__ void v4_quant_v1(const bf16* __restrict__ v, uint8_t* __restrict__ p, __half* __restrict__ s, __half* __restrict__ z, int64_t rows) {
  v4_row_quant(v, p, s, z, rows, static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.x / 32);
}
__global__ void v4_quant_v2(const bf16* __restrict__ v, uint8_t* __restrict__ p, __half* __restrict__ s, __half* __restrict__ z, int64_t rows) {
  v4_rowpair_quant(v, p, s, z, rows, static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK_V2 + 2 * (threadIdx.x / 32));
}
__global__ void k4_dequant_v1(const uint8_t* __restrict__ p, const __half* __restrict__ s, const __half* __restrict__ z, bf16* __restrict__ k, Dims dm) {
  k4_tile_dequant(p, s, z, k, dm, blockIdx.x);
}
__global__ void v4_dequant_v1(const uint8_t* __restrict__ p, const __half* __restrict__ s, const __half* __restrict__ z, bf16* __restrict__ v, int64_t rows) {
  v4_row_dequant(p, s, z, v, rows, static_cast<int64_t>(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.x / 32);
}
// v2: one launch for the whole layer batch, K tiles first then V row-blocks
__global__ void __launch_bounds__(THREADS, 3) kv4_quant_v2(const bf16* __restrict__ k, const bf16* __restrict__ v,
                             uint8_t* __restrict__ kp, __half* __restrict__ ks, __half* __restrict__ kz,
                             uint8_t* __restrict__ vp, __half* __restrict__ vs, __half* __restrict__ vz, Dims dm, int64_t n_k_tiles, int64_t rows) {
  const int64_t b = blockIdx.x;
  if (b < n_k_tiles) k4_tile_quant(k, kp, ks, kz, dm, b);
  else v4_rowpair_quant(v, vp, vs, vz, rows, (b - n_k_tiles) * ROWS_PER_BLOCK_V2 + 2 * (threadIdx.x / 32));
}

// =====================================================================================================================
// host side
// =====================================================================================================================
static Dims dims_of(const at::Tensor& x) { return Dims{x.size(0), x.size(1), x.size(2), k_groups(x.size(1))}; }
static void check_k(const at::Tensor& k, const at::Tensor& p, const at::Tensor& s, const at::Tensor& z) {
  TORCH_CHECK(k.is_cuda() && k.scalar_type() == at::kBFloat16 && k.is_contiguous() && k.dim() == 4 && k.size(3) == HEAD_DIM, "k: contiguous bf16 CUDA [L,T,H,128]");
  const Dims dm = dims_of(k);
  TORCH_CHECK(p.is_cuda() && p.scalar_type() == at::kByte && p.is_contiguous() && p.sizes() == at::IntArrayRef({dm.L, dm.T, dm.H, HEAD_DIM / 2}), "payload: u8 [L,T,H,64]");
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kHalf && s.is_contiguous() && s.sizes() == at::IntArrayRef({dm.L, dm.Tg, dm.H, HEAD_DIM}), "scale: f16 [L,Tg,H,128]");
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kHalf && z.is_contiguous() && z.sizes() == s.sizes(), "zero must match scale");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(s.data_ptr()) % 4 == 0, "alignment");
}
static void check_v(const at::Tensor& v, const at::Tensor& p, const at::Tensor& s, const at::Tensor& z) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kBFloat16 && v.is_contiguous() && v.dim() == 4 && v.size(3) == HEAD_DIM, "v: contiguous bf16 CUDA [L,T,H,128]");
  const Dims dm = dims_of(v);
  TORCH_CHECK(p.is_cuda() && p.scalar_type() == at::kByte && p.is_contiguous() && p.sizes() == at::IntArrayRef({dm.L, dm.T, dm.H, HEAD_DIM / 2}), "payload: u8 [L,T,H,64]");
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kHalf && s.is_contiguous() && s.sizes() == at::IntArrayRef({dm.L, dm.T, dm.H, V_GROUPS}), "scale: f16 [L,T,H,4]");
  TORCH_CHECK(z.is_cuda() && z.scalar_type() == at::kHalf && z.is_contiguous() && z.sizes() == s.sizes(), "zero must match scale");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 8 == 0, "alignment");
}
static inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline unsigned grid1(int64_t blocks) { TORCH_CHECK(blocks < (1LL << 31), "grid too large"); return static_cast<unsigned>(blocks); }
#define KVQ_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()

void quantize_k4_cuda(const at::Tensor& k, at::Tensor& p, at::Tensor& s, at::Tensor& z, int version) {
  check_k(k, p, s, z); const c10::cuda::CUDAGuard guard(k.device()); auto st = at::cuda::getCurrentCUDAStream();
  const Dims dm = dims_of(k);
  auto kp = reinterpret_cast<const bf16*>(k.data_ptr()); auto pp = p.data_ptr<uint8_t>();
  auto sp = reinterpret_cast<__half*>(s.data_ptr()); auto zp = reinterpret_cast<__half*>(z.data_ptr());
  if (version == 0) {
    k4_scales_v0<<<grid1(cdiv(dm.L * dm.Tg * dm.H * HEAD_DIM, THREADS)), THREADS, 0, st>>>(kp, sp, zp, dm); KVQ_LAUNCH_CHECK();
    k4_codes_v0<<<grid1(cdiv(dm.L * dm.T * dm.H * (HEAD_DIM / 2), THREADS)), THREADS, 0, st>>>(kp, sp, zp, pp, dm); KVQ_LAUNCH_CHECK();
  } else {
    const int64_t tiles = dm.L * dm.Tg * cdiv(dm.H, HEADS_PER_BLOCK);
    k4_quant_v1<<<grid1(tiles), THREADS, 0, st>>>(kp, pp, sp, zp, dm); KVQ_LAUNCH_CHECK();
  }
}
void quantize_v4_cuda(const at::Tensor& v, at::Tensor& p, at::Tensor& s, at::Tensor& z, int version) {
  check_v(v, p, s, z); const c10::cuda::CUDAGuard guard(v.device()); auto st = at::cuda::getCurrentCUDAStream();
  const int64_t rows = v.size(0) * v.size(1) * v.size(2);
  auto vp = reinterpret_cast<const bf16*>(v.data_ptr()); auto pp = p.data_ptr<uint8_t>();
  auto sp = reinterpret_cast<__half*>(s.data_ptr()); auto zp = reinterpret_cast<__half*>(z.data_ptr());
  if (version == 0) {
    v4_scales_v0<<<grid1(cdiv(rows * V_GROUPS, THREADS)), THREADS, 0, st>>>(vp, sp, zp, rows); KVQ_LAUNCH_CHECK();
    v4_codes_v0<<<grid1(cdiv(rows * (HEAD_DIM / 2), THREADS)), THREADS, 0, st>>>(vp, sp, zp, pp, rows); KVQ_LAUNCH_CHECK();
  } else if (version == 1) {
    v4_quant_v1<<<grid1(cdiv(rows, ROWS_PER_BLOCK)), THREADS, 0, st>>>(vp, pp, sp, zp, rows); KVQ_LAUNCH_CHECK();
  } else {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "v must be 16-byte aligned for v2");
    v4_quant_v2<<<grid1(cdiv(rows, ROWS_PER_BLOCK_V2)), THREADS, 0, st>>>(vp, pp, sp, zp, rows); KVQ_LAUNCH_CHECK();
  }
}
void quantize_kv4_cuda(const at::Tensor& k, const at::Tensor& v, at::Tensor& kp, at::Tensor& ks, at::Tensor& kz,
                       at::Tensor& vp, at::Tensor& vs, at::Tensor& vz, int version) {
  if (version < 2) { quantize_k4_cuda(k, kp, ks, kz, version); quantize_v4_cuda(v, vp, vs, vz, version); return; }
  check_k(k, kp, ks, kz); check_v(v, vp, vs, vz);
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
  const c10::cuda::CUDAGuard guard(k.device()); auto st = at::cuda::getCurrentCUDAStream();
  const Dims dm = dims_of(k); const int64_t rows = dm.L * dm.T * dm.H;
  const int64_t n_k = dm.L * dm.Tg * cdiv(dm.H, HEADS_PER_BLOCK), n_v = cdiv(rows, ROWS_PER_BLOCK_V2);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "v must be 16-byte aligned for v2");
  kv4_quant_v2<<<grid1(n_k + n_v), THREADS, 0, st>>>(
      reinterpret_cast<const bf16*>(k.data_ptr()), reinterpret_cast<const bf16*>(v.data_ptr()),
      kp.data_ptr<uint8_t>(), reinterpret_cast<__half*>(ks.data_ptr()), reinterpret_cast<__half*>(kz.data_ptr()),
      vp.data_ptr<uint8_t>(), reinterpret_cast<__half*>(vs.data_ptr()), reinterpret_cast<__half*>(vz.data_ptr()), dm, n_k, rows);
  KVQ_LAUNCH_CHECK();
}
void dequantize_k4_cuda(const at::Tensor& p, const at::Tensor& s, const at::Tensor& z, at::Tensor& k, int version) {
  check_k(k, p, s, z); const c10::cuda::CUDAGuard guard(k.device()); auto st = at::cuda::getCurrentCUDAStream();
  const Dims dm = dims_of(k);
  auto kp = reinterpret_cast<bf16*>(k.data_ptr()); auto pp = p.data_ptr<uint8_t>();
  auto sp = reinterpret_cast<const __half*>(s.data_ptr()); auto zp = reinterpret_cast<const __half*>(z.data_ptr());
  if (version == 0) { k4_dequant_v0<<<grid1(cdiv(dm.L * dm.T * dm.H * (HEAD_DIM / 2), THREADS)), THREADS, 0, st>>>(pp, sp, zp, kp, dm); }
  else { k4_dequant_v1<<<grid1(dm.L * dm.Tg * cdiv(dm.H, HEADS_PER_BLOCK)), THREADS, 0, st>>>(pp, sp, zp, kp, dm); }
  KVQ_LAUNCH_CHECK();
}
void dequantize_v4_cuda(const at::Tensor& p, const at::Tensor& s, const at::Tensor& z, at::Tensor& v, int version) {
  check_v(v, p, s, z); const c10::cuda::CUDAGuard guard(v.device()); auto st = at::cuda::getCurrentCUDAStream();
  const int64_t rows = v.size(0) * v.size(1) * v.size(2);
  auto vp = reinterpret_cast<bf16*>(v.data_ptr()); auto pp = p.data_ptr<uint8_t>();
  auto sp = reinterpret_cast<const __half*>(s.data_ptr()); auto zp = reinterpret_cast<const __half*>(z.data_ptr());
  if (version == 0) { v4_dequant_v0<<<grid1(cdiv(rows * (HEAD_DIM / 2), THREADS)), THREADS, 0, st>>>(pp, sp, zp, vp, rows); }
  else { v4_dequant_v1<<<grid1(cdiv(rows, ROWS_PER_BLOCK)), THREADS, 0, st>>>(pp, sp, zp, vp, rows); }
  KVQ_LAUNCH_CHECK();
}

}  // namespace kvq
