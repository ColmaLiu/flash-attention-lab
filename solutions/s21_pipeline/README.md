# s21: shared reuse and an explicit asynchronous pipeline

`kernel.cu` implements a 64×64 CTA output tile with a 32-element reduction step,
four warps, and two shared-memory buffers. `cp.async` loads the next A/B tile
while WMMA consumes the current tile. Partial CTA tiles are zero-filled, and
only complete valid 16×16 fragments are stored. The public domain remains
multiples of 16; it is not restricted to multiples of the CTA tile.

The synchronization has two separate obligations: all consumers must finish
before a buffer can be reused, and every producer's copies must complete before
the CTA consumes that buffer. Follow these obligations across prologue, steady
state, and final iteration. `wait_group` alone does not publish other threads'
copies to the CTA.

WMMA is used here to keep the experiment about buffering and reuse. s20 exposes
the underlying instruction/register mapping directly. This is a complete local
kernel; cuBLAS supplies the independent speed comparison.

```bash
./lab check s21 --solution
./lab solution s21
./lab sanitize s21 --solution --tool racecheck --case 4
./lab sanitize s21 --solution --tool synccheck --case 3
```
