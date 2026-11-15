// kvq.h -- KV4 wire format: constants, shapes, and the exact quantization arithmetic shared by the
// C++ CPU reference (kvq_ref.cpp) and the CUDA kernels (kvq_kernels.cu). Both must stay bit-identical
// to the Python reference in kvpick/ref/quant_wire.py.
//
// Layout (one layer batch of L layers, T tokens, H kv heads, D = 128 dims):
//   K  input   bf16 [L, T, H, D]
//      payload u8   [L, T, H, D/2]      two 4-bit codes per byte, low nibble = even d, high nibble = odd d
//      scale   f16  [L, Tg, H, D]       Tg = ceil(T/32): one (scale, zero) per (32-token group, head, channel)
//      zero    f16  [L, Tg, H, D]
//   V  input   bf16 [L, T, H, D]
//      payload u8   [L, T, H, D/2]
//      scale   f16  [L, T, H, D/32]     one (scale, zero) per (token, head, 32-dim group)
//      zero    f16  [L, T, H, D/32]
//
// Arithmetic (all in fp32 unless stated, no FMA contraction):
//   zero  = f16_rn(min)                      scale = f16_rn( max( (max - float(zero)) * fp32(1/15) , 2^-24 ) )
//   inv   = 1 / float(scale)   (IEEE, once per group; __frcp_rn on device)
//   code  = clamp( rint( (x - float(zero)) * inv ), 0, 15 )                 rint = round half to even
//   dequant = bf16_rn( float(code) * float(scale) + float(zero) )           two roundings: mul then add
#pragma once
#include <cstdint>

namespace kvq {

constexpr int HEAD_DIM = 128;
constexpr int GROUP = 32;                   // K: tokens per group; V: dims per group
constexpr int V_GROUPS = HEAD_DIM / GROUP;  // 4 groups per row
constexpr int QMAX = 15;                    // 4-bit asymmetric codes 0..15
constexpr float EPS = 5.9604644775390625e-08f;  // 2^-24, smallest fp16 subnormal (D21)
constexpr float INV_QMAX = 1.0f / 15.0f;         // scale = range * INV_QMAX (a multiply, as torch does for tensor / python-scalar) (D29)

#if defined(__CUDACC__)
#define KVQ_HD __host__ __device__ __forceinline__
#else
#define KVQ_HD inline
#endif

KVQ_HD int64_t k_groups(int64_t T) { return (T + GROUP - 1) / GROUP; }

// Byte sizes for one layer batch (for the transfer buffer accounting).
KVQ_HD int64_t payload_bytes(int64_t L, int64_t T, int64_t H) { return L * T * H * (HEAD_DIM / 2); }
KVQ_HD int64_t k_meta_elems(int64_t L, int64_t T, int64_t H) { return L * k_groups(T) * H * HEAD_DIM; }   // per tensor (scale or zero)
KVQ_HD int64_t v_meta_elems(int64_t L, int64_t T, int64_t H) { return L * T * H * V_GROUPS; }
KVQ_HD int64_t kv4_bytes_per_layer(int64_t T, int64_t H) {
  return 2 * payload_bytes(1, T, H) + 2 * 2 * k_meta_elems(1, T, H) + 2 * 2 * v_meta_elems(1, T, H);
}

// The reductions and quantization are expressed on floats so that host and device share them.
// f16 round-to-nearest-even is supplied by the caller's half type (at::Half on host, __half on device).
KVQ_HD float scale_from_range(float mx, float zero_f) {
  float s = (mx - zero_f) * INV_QMAX;
  return s < EPS ? EPS : s;
}

KVQ_HD float inv_scale(float scale_f) {
#if defined(__CUDA_ARCH__)
  return __frcp_rn(scale_f);                 // correctly rounded 1/x == IEEE 1.0f/x on the host
#else
  return 1.0f / scale_f;
#endif
}

KVQ_HD uint8_t quant_code(float x, float zero_f, float inv_f) {
#if defined(__CUDA_ARCH__)
  float q = rintf(__fmul_rn(x - zero_f, inv_f));
#else
  float q = __builtin_rintf((x - zero_f) * inv_f);
#endif
  q = q < 0.f ? 0.f : (q > static_cast<float>(QMAX) ? static_cast<float>(QMAX) : q);
  return static_cast<uint8_t>(q);
}

KVQ_HD uint8_t pack2(uint8_t lo, uint8_t hi) { return static_cast<uint8_t>(lo | (hi << 4)); }

}  // namespace kvq
