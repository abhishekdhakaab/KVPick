// kvq_ref.cpp -- slow, obvious C++ CPU reference for the KV4 wire format. Ground truth for the CUDA kernels.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include "kvq.h"

namespace kvq {

static void check_k_shapes(const at::Tensor& k, const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero) {
  TORCH_CHECK(k.dim() == 4 && k.size(3) == HEAD_DIM, "k must be [L, T, H, 128]");
  TORCH_CHECK(k.scalar_type() == at::kBFloat16 && k.is_contiguous(), "k must be contiguous bf16");
  const int64_t L = k.size(0), T = k.size(1), H = k.size(2), Tg = k_groups(T);
  TORCH_CHECK(payload.scalar_type() == at::kByte && payload.is_contiguous() && payload.sizes() == at::IntArrayRef({L, T, H, HEAD_DIM / 2}), "payload must be u8 [L,T,H,64]");
  TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.is_contiguous() && scale.sizes() == at::IntArrayRef({L, Tg, H, HEAD_DIM}), "scale must be f16 [L,Tg,H,128]");
  TORCH_CHECK(zero.scalar_type() == at::kHalf && zero.is_contiguous() && zero.sizes() == scale.sizes(), "zero must match scale");
}

static void check_v_shapes(const at::Tensor& v, const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero) {
  TORCH_CHECK(v.dim() == 4 && v.size(3) == HEAD_DIM, "v must be [L, T, H, 128]");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16 && v.is_contiguous(), "v must be contiguous bf16");
  const int64_t L = v.size(0), T = v.size(1), H = v.size(2);
  TORCH_CHECK(payload.scalar_type() == at::kByte && payload.is_contiguous() && payload.sizes() == at::IntArrayRef({L, T, H, HEAD_DIM / 2}), "payload must be u8 [L,T,H,64]");
  TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.is_contiguous() && scale.sizes() == at::IntArrayRef({L, T, H, V_GROUPS}), "scale must be f16 [L,T,H,4]");
  TORCH_CHECK(zero.scalar_type() == at::kHalf && zero.is_contiguous() && zero.sizes() == scale.sizes(), "zero must match scale");
}

// ---- K: per (layer, token-group, head, channel) over up to 32 tokens
void quantize_k4_cpu(const at::Tensor& k, at::Tensor& payload, at::Tensor& scale, at::Tensor& zero) {
  check_k_shapes(k, payload, scale, zero);
  const int64_t L = k.size(0), T = k.size(1), H = k.size(2), Tg = k_groups(T);
  const c10::BFloat16* kp = k.data_ptr<c10::BFloat16>();
  uint8_t* pp = payload.data_ptr<uint8_t>();
  c10::Half* sp = scale.data_ptr<c10::Half>(); c10::Half* zp = zero.data_ptr<c10::Half>();
  auto idx = [&](int64_t l, int64_t t, int64_t h, int64_t d) { return ((l * T + t) * H + h) * HEAD_DIM + d; };
  for (int64_t l = 0; l < L; ++l)
    for (int64_t g = 0; g < Tg; ++g) {
      const int64_t t0 = g * GROUP, t1 = std::min<int64_t>(T, t0 + GROUP);
      for (int64_t h = 0; h < H; ++h)
        for (int64_t d = 0; d < HEAD_DIM; ++d) {
          float mn = INFINITY, mx = -INFINITY;
          for (int64_t t = t0; t < t1; ++t) { float x = static_cast<float>(kp[idx(l, t, h, d)]); mn = std::min(mn, x); mx = std::max(mx, x); }
          c10::Half z16(mn);                                   // f16 RN
          float zf = static_cast<float>(z16);
          c10::Half s16(scale_from_range(mx, zf));
          float inv = inv_scale(static_cast<float>(s16));
          const int64_t m = ((l * Tg + g) * H + h) * HEAD_DIM + d;
          sp[m] = s16; zp[m] = z16;
          for (int64_t t = t0; t < t1; ++t) {
            uint8_t c = quant_code(static_cast<float>(kp[idx(l, t, h, d)]), zf, inv);
            uint8_t& b = pp[((l * T + t) * H + h) * (HEAD_DIM / 2) + d / 2];
            if (d % 2 == 0) b = static_cast<uint8_t>((b & 0xF0) | c); else b = static_cast<uint8_t>((b & 0x0F) | (c << 4));
          }
        }
    }
}

// ---- V: per (layer, token, head, 32-dim group)
void quantize_v4_cpu(const at::Tensor& v, at::Tensor& payload, at::Tensor& scale, at::Tensor& zero) {
  check_v_shapes(v, payload, scale, zero);
  const int64_t L = v.size(0), T = v.size(1), H = v.size(2);
  const c10::BFloat16* vp = v.data_ptr<c10::BFloat16>();
  uint8_t* pp = payload.data_ptr<uint8_t>();
  c10::Half* sp = scale.data_ptr<c10::Half>(); c10::Half* zp = zero.data_ptr<c10::Half>();
  const int64_t rows = L * T * H;
  for (int64_t r = 0; r < rows; ++r) {
    const c10::BFloat16* row = vp + r * HEAD_DIM;
    uint8_t* out = pp + r * (HEAD_DIM / 2);
    for (int g = 0; g < V_GROUPS; ++g) {
      float mn = INFINITY, mx = -INFINITY;
      for (int d = g * GROUP; d < (g + 1) * GROUP; ++d) { float x = static_cast<float>(row[d]); mn = std::min(mn, x); mx = std::max(mx, x); }
      c10::Half z16(mn); float zf = static_cast<float>(z16);
      c10::Half s16(scale_from_range(mx, zf)); float inv = inv_scale(static_cast<float>(s16));
      sp[r * V_GROUPS + g] = s16; zp[r * V_GROUPS + g] = z16;
      for (int d = g * GROUP; d < (g + 1) * GROUP; d += 2)
        out[d / 2] = pack2(quant_code(static_cast<float>(row[d]), zf, inv), quant_code(static_cast<float>(row[d + 1]), zf, inv));
    }
  }
}

static inline c10::BFloat16 dq(uint8_t code, float sf, float zf) {
  volatile float prod = static_cast<float>(code) * sf;   // volatile blocks FMA contraction: mul rounds, then add rounds
  float y = prod + zf;
  return c10::BFloat16(y);                                 // bf16 RN
}

void dequantize_k4_cpu(const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero, at::Tensor& k) {
  check_k_shapes(k, payload, scale, zero);
  const int64_t L = k.size(0), T = k.size(1), H = k.size(2), Tg = k_groups(T);
  c10::BFloat16* kp = k.data_ptr<c10::BFloat16>();
  const uint8_t* pp = payload.data_ptr<uint8_t>();
  const c10::Half* sp = scale.data_ptr<c10::Half>(); const c10::Half* zp = zero.data_ptr<c10::Half>();
  for (int64_t l = 0; l < L; ++l)
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < H; ++h)
        for (int64_t d = 0; d < HEAD_DIM; ++d) {
          const int64_t m = ((l * Tg + t / GROUP) * H + h) * HEAD_DIM + d;
          uint8_t b = pp[((l * T + t) * H + h) * (HEAD_DIM / 2) + d / 2];
          uint8_t c = (d % 2 == 0) ? (b & 0xF) : (b >> 4);
          kp[((l * T + t) * H + h) * HEAD_DIM + d] = dq(c, static_cast<float>(sp[m]), static_cast<float>(zp[m]));
        }
}

void dequantize_v4_cpu(const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero, at::Tensor& v) {
  check_v_shapes(v, payload, scale, zero);
  const int64_t rows = v.size(0) * v.size(1) * v.size(2);
  c10::BFloat16* vp = v.data_ptr<c10::BFloat16>();
  const uint8_t* pp = payload.data_ptr<uint8_t>();
  const c10::Half* sp = scale.data_ptr<c10::Half>(); const c10::Half* zp = zero.data_ptr<c10::Half>();
  for (int64_t r = 0; r < rows; ++r)
    for (int d = 0; d < HEAD_DIM; ++d) {
      const int g = d / GROUP; uint8_t b = pp[r * (HEAD_DIM / 2) + d / 2];
      uint8_t c = (d % 2 == 0) ? (b & 0xF) : (b >> 4);
      vp[r * HEAD_DIM + d] = dq(c, static_cast<float>(sp[r * V_GROUPS + g]), static_cast<float>(zp[r * V_GROUPS + g]));
    }
}

}  // namespace kvq
