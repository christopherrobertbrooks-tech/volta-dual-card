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

---

# Part 2: a dense 27B, and why splitting made it *slower*

`Qwen3.8-27B-Q8_0` (27.04 GiB, 27.32 B params — the base model Bonsai is
quantised from), same card pair, `-fa 0 -p 512 -n 128 -r 2`.

| config | pp512 | tg128 |
| :--- | ---: | ---: |
| **V100 alone** | **720.0 ± 39.5** | **23.07 ± 0.02** |
| Both, `-sm layer` | 713.5 ± 0.3 | 20.86 ± 0.00 |
| Both, `-sm row -mg 0` | — | **fails to load** |

**Splitting cost 9.6% of decode here**, the opposite of the DeepSeek result
above where it gained 5.5%.

## Why the two models disagree

Decode on a *dense* model is bound by weight bandwidth, and the two cards have
very different memory:

| card | memory | bandwidth |
| :--- | :--- | ---: |
| Tesla V100-PCIE | HBM2 | ~900 GB/s |
| RTX 4070 | GDDR6X | ~504 GB/s |

Layer split hands roughly 27% of the weights to the card with **half the
bandwidth**, so a dense model decodes slower. DeepSeek-V2-Lite is a MoE with
only ~2.4 B active parameters per token, so its weight traffic is small and the
4070's share costs little while its compute helps.

**Rule of thumb: splitting a dense model onto a lower-bandwidth card loses.**
Layer split pays when the model does not fit otherwise, or when the second card
is at least as fast — not as a general speed-up.

## `-sm row` is unavailable on this pair

It fails to load, and **not from memory pressure**: it fails identically with
DeepSeek-V2-Lite at 9.65 GiB, which would need only ~4.8 GiB per card under a
row split. Tested specifically to separate "row split broken" from "out of
memory" — the small model rules out OOM.

This matches [#27366](https://github.com/ggml-org/llama.cpp/issues/27366)
("`-sm row` unavailable on CUDA") and extends it to a mixed sm_70/sm_89 pair.

**Consequence: KV cannot be placed on a chosen card here.** `-mg` only
redirects KV and intermediate results under `-sm row`; with `-sm layer` the KV
for each layer lives on whichever card holds that layer. So "weights on the
V100, KV cache on the 4070" is not achievable with current llama.cpp on this
hardware.

`llama-bench` reports only `failed to load model` with no cause, at any
verbosity tried.

## Incidental: what the ternary quantisation is worth

Qwen3.8-27B is the base model for Ternary-Bonsai-2-27B, so these are directly
comparable on the same card:

| build | size | tg128 on V100 |
| :--- | ---: | ---: |
| Q8_0 (conventional) | 27.04 GiB | 23.07 |
| Bonsai PQ2_0 (ternary) | 6.70 GiB | **51.0** |

**4x smaller and 2.2x faster decode**, same weights underneath. Quality is not
compared here — that needs a benchmark suite, not a stopwatch — but the model
card's claim of 98.2% retention is at least being paid for with a real speed and
footprint win on this hardware.

No VRAM leak in any configuration; both cards returned to baseline each time.
