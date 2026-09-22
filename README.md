# Mixed sm_70 + sm_89 layer split: works, cheap, no leak

A Tesla V100 (sm_70, Volta) and an RTX 4070 (sm_89, Ada) in one llama.cpp job.
Multi-GPU on Volta has open bug reports; mixing architectures is stranger still.
It works.

Model: `DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M` (9.65 GiB), upstream llama.cpp
`c550d2f60`, `llama-bench -ngl 99 -fa 0 -sm layer -p 512 -n 128 -r 3`.

## Result

| config | pp512 | tg128 |
| :--- | ---: | ---: |
| V100 alone (sm_70) | 1836.3 ± 16.5 | 164.6 ± 0.4 |
| **Both, `-sm layer`** | **2080.5 ± 8.1** | **173.7 ± 0.4** |
| RTX 4070 alone (sm_89) | 2789.3 ± 329.3 | **208.0 ± 0.9** |

The split distributed by VRAM ratio: 2,840 MiB on the 4070 against 7,642 MiB on
the V100, about 27/73, with GPU utilisation tracking it at 24% and 75%.

**Splitting beats the V100 alone by 5.5% at decode.** It loses to the 4070 alone
by 17%, because this model fits entirely on the 4070 — so for models that fit on
one card, use the fastest single card. Splitting only pays when the model does
not fit otherwise.

**No VRAM leak.** Both cards returned to exactly their baseline (796 MiB desktop
on the 4070, 10 MiB on the V100) with only the pre-existing process remaining.
Checked because [#27366](https://github.com/ggml-org/llama.cpp/issues/27366)
reports `-sm tensor` hanging and leaking VRAM on multi-GPU Volta.

## A prediction that was wrong, and why

Before running this, the expectation was that splitting would scale badly,
because the V100 sits on a **PCIe x4** link (`LnkSta: Speed 8GT/s, Width x4
(downgraded)`) — roughly 3.9 GB/s, the narrowest part of the machine.

That reasoning confused two split modes:

- **`-sm layer`** (tested here) passes only the hidden state across the boundary
  — a couple of thousand floats, order 8 KB per token, once per crossing.
  Negligible at any link width. Hence the good result.
- **`-sm tensor`** divides individual matrices and exchanges partial results
  continuously. *That* is the bandwidth-bound mode the x4 link would punish —
  and it is also the mode with the hang-and-leak report.

So the bandwidth concern was real but aimed at the wrong mode.

## Untested

- **`-sm tensor`** on this pair. It is the mode #27366 reports hanging and
  leaking VRAM on multi-GPU Volta, and reproducing that on a *mixed*
  architecture pair would be a genuine datapoint — but the leak would land on
  the card that runs the Etsy business. Not attempted without a specific
  decision to accept that.
- **A model too large for either card alone.** Now justified by this result
  rather than assumed: combined 44 GB is real and the split overhead is low. A
  70B at Q4_K_M (~40 GB) would fit.
- Flash attention under split. Only `-fa 0` was measured, since FA is a
  regression for this model on both cards (see `volta-deepseek-mla`).

## Environment

See `volta-bonsai/ENVIRONMENT.md`. ComfyUI was stopped throughout; the 4070
otherwise belongs to it.
