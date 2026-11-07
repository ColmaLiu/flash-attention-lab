#include "../s40_attention/sm100_attention.cuh"
using namespace sm100_attention;
void *prepare_attention(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o,
                        uintptr_t lse, int b, int n, int h, int d, bool half,
                        float scale, bool causal, uintptr_t stream) {
  if (d != 64 && d != 128)
    throw std::runtime_error("attention requires D=64 or 128");
  auto *s = new Prepared{};
  try {
    encode(&s->q, q, b, n, h, d, BM, half);
    encode(&s->k, k, b, n, h, d, BN, half);
    encode(&s->v, v, b, n, h, d, BN, half);
    int device;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&s->sms, cudaDevAttrMultiProcessorCount,
                                 device));
  } catch (...) {
    delete s;
    throw;
  }
  s->batch = b;
  s->n = n;
  s->heads = h;
  s->dim = d;
  s->half = half;
  s->causal = causal;
  s->scale = scale;
  s->output = reinterpret_cast<void *>(o);
  s->lse = reinterpret_cast<float *>(lse);
  s->stream = reinterpret_cast<cudaStream_t>(stream);
  return s;
}
void run_attention(void *state) {
  const auto &s = *static_cast<Prepared *>(state);
  if (s.half == false && s.dim == 64)
    launch<__nv_bfloat16, 64, true, false, false>(s);
  else if (s.half == false && s.dim == 128)
    launch<__nv_bfloat16, 128, true, false, false>(s);
  else if (s.half == true && s.dim == 64)
    launch<__half, 64, true, false, false>(s);
  else if (s.half == true && s.dim == 128)
    launch<__half, 128, true, false, false>(s);
}
void delete_attention(void *state) {
  delete static_cast<Prepared *>(state);
}
