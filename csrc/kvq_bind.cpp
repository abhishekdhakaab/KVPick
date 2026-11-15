// kvq_bind.cpp -- PyTorch extension entry points. CPU tensors go to the C++ reference; CUDA tensors to the kernels.
#include <torch/extension.h>
#include "kvq.h"

namespace kvq {
void quantize_k4_cpu(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void quantize_v4_cpu(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void dequantize_k4_cpu(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
void dequantize_v4_cpu(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
#ifdef KVQ_WITH_CUDA
void quantize_k4_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int version);
void quantize_v4_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int version);
void quantize_kv4_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int version);
void dequantize_k4_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int version);
void dequantize_v4_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int version);
#endif
}  // namespace kvq

static void quantize_k4(const at::Tensor& k, at::Tensor payload, at::Tensor scale, at::Tensor zero, int version) {
  if (k.is_cuda()) {
#ifdef KVQ_WITH_CUDA
    kvq::quantize_k4_cuda(k, payload, scale, zero, version); return;
#else
    TORCH_CHECK(false, "kvq was built without CUDA");
#endif
  }
  kvq::quantize_k4_cpu(k, payload, scale, zero);
}
static void quantize_v4(const at::Tensor& v, at::Tensor payload, at::Tensor scale, at::Tensor zero, int version) {
  if (v.is_cuda()) {
#ifdef KVQ_WITH_CUDA
    kvq::quantize_v4_cuda(v, payload, scale, zero, version); return;
#else
    TORCH_CHECK(false, "kvq was built without CUDA");
#endif
  }
  kvq::quantize_v4_cpu(v, payload, scale, zero);
}
static void quantize_kv4(const at::Tensor& k, const at::Tensor& v, at::Tensor kp, at::Tensor ks, at::Tensor kz, at::Tensor vp, at::Tensor vs, at::Tensor vz, int version) {
  if (k.is_cuda()) {
#ifdef KVQ_WITH_CUDA
    kvq::quantize_kv4_cuda(k, v, kp, ks, kz, vp, vs, vz, version); return;
#else
    TORCH_CHECK(false, "kvq was built without CUDA");
#endif
  }
  kvq::quantize_k4_cpu(k, kp, ks, kz); kvq::quantize_v4_cpu(v, vp, vs, vz);
}
static void dequantize_k4(const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero, at::Tensor k, int version) {
  if (k.is_cuda()) {
#ifdef KVQ_WITH_CUDA
    kvq::dequantize_k4_cuda(payload, scale, zero, k, version); return;
#else
    TORCH_CHECK(false, "kvq was built without CUDA");
#endif
  }
  kvq::dequantize_k4_cpu(payload, scale, zero, k);
}
static void dequantize_v4(const at::Tensor& payload, const at::Tensor& scale, const at::Tensor& zero, at::Tensor v, int version) {
  if (v.is_cuda()) {
#ifdef KVQ_WITH_CUDA
    kvq::dequantize_v4_cuda(payload, scale, zero, v, version); return;
#else
    TORCH_CHECK(false, "kvq was built without CUDA");
#endif
  }
  kvq::dequantize_v4_cpu(payload, scale, zero, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "KV4: 4-bit KV-cache pack/unpack (per-channel keys, per-token values)";
  m.def("quantize_k4", &quantize_k4, "K -> payload, scale, zero", py::arg("k"), py::arg("payload"), py::arg("scale"), py::arg("zero"), py::arg("version") = 2);
  m.def("quantize_v4", &quantize_v4, "V -> payload, scale, zero", py::arg("v"), py::arg("payload"), py::arg("scale"), py::arg("zero"), py::arg("version") = 2);
  m.def("quantize_kv4", &quantize_kv4, "fused K and V", py::arg("k"), py::arg("v"), py::arg("kp"), py::arg("ks"), py::arg("kz"), py::arg("vp"), py::arg("vs"), py::arg("vz"), py::arg("version") = 2);
  m.def("dequantize_k4", &dequantize_k4, "", py::arg("payload"), py::arg("scale"), py::arg("zero"), py::arg("k"), py::arg("version") = 2);
  m.def("dequantize_v4", &dequantize_v4, "", py::arg("payload"), py::arg("scale"), py::arg("zero"), py::arg("v"), py::arg("version") = 2);
  m.attr("HEAD_DIM") = kvq::HEAD_DIM; m.attr("GROUP") = kvq::GROUP; m.attr("QMAX") = kvq::QMAX;
#ifdef KVQ_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}
