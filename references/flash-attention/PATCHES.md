# Lab patch to the fixed FA4 source

Upstream commit: `1fc0221183e799363001394a12dd9e78e15f5dda`.

`flash_attn/cute/flash_fwd_sm100.py` exposes the existing 16-bit softmax rescaling threshold as an optional constructor argument:

- Add `rescale_threshold: float = 8.0` to `FlashAttentionForwardSm100.__init__`.
- Store it as `self.lab_rescale_threshold`.
- Use that constant where the 16-bit path previously selected a literal `8.0`; the non-16-bit path remains `0.0`.

This permits exact-rescale versus conditional-rescale experiments from separate lab solution configurations. The default preserves the selected upstream strategy. The complete original kernel structure and license remain. The source manifest records hashes of the distributed files; `docs/sources.lock.json` also records the original upstream file hash.
