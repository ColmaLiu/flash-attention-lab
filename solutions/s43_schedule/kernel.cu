#include "attention_dense.cuh"
#include "attention_single.cuh"
using namespace sm100_attention;
void *prepare_attention(uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o,
                        uintptr_t lse, int b, int n, int h, int d, bool half,
                        float scale, bool causal, uintptr_t stream,
                        bool polynomial, bool delayed_rescale,
                        bool persistent) {
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
  s->polynomial = polynomial;
  s->delayed_rescale = delayed_rescale;
  s->persistent = persistent;
  return s;
}
// Query task count controls the tradeoff between parallelism and KV reuse.
// Triangular work needs finer query granularity when one paired wave underfills
// the device. Dense work retains paired KV reuse unless less than half is busy.
template <class T, int D, bool Persistent>
void run_selected(const Prepared &s) {
  const int paired_tasks = s.batch * s.heads * ((s.n + 255) / 256);
  const bool single =
      s.causal ? paired_tasks < s.sms : paired_tasks * 2 < s.sms;
  if (single)
    sm100_attention_single::launch_numerics<T, D, Persistent>(s);
  else
    sm100_attention_dense::launch_numerics<T, D, Persistent>(s);
}
void run_attention(void *state) {
  const auto &s = *static_cast<Prepared *>(state);
  if (s.persistent) {
    if (s.half == false && s.dim == 64)
      run_selected<__nv_bfloat16, 64, true>(s);
    else if (s.half == false && s.dim == 128)
      run_selected<__nv_bfloat16, 128, true>(s);
    else if (s.half == true && s.dim == 64)
      run_selected<__half, 64, true>(s);
    else if (s.half == true && s.dim == 128)
      run_selected<__half, 128, true>(s);
  } else {
    if (s.half == false && s.dim == 64)
      run_selected<__nv_bfloat16, 64, false>(s);
    else if (s.half == false && s.dim == 128)
      run_selected<__nv_bfloat16, 128, false>(s);
    else if (s.half == true && s.dim == 64)
      run_selected<__half, 64, false>(s);
    else if (s.half == true && s.dim == 128)
      run_selected<__half, 128, false>(s);
  }
}
void delete_attention(void *state) {
  delete static_cast<Prepared *>(state);
}
