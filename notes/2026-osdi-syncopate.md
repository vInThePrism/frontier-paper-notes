# Syncopate: compiling fine-grained compute–communication overlap

**Question:** Kernel-level overlap already hides some communication, so why move overlap *inside* a kernel, and how can a compiler do that automatically instead of an expert hand-fusing every operator?

- Paper: Qiang, Guan, et al., *Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap*, OSDI '26 ([PDF](https://www.usenix.org/system/files/osdi26-qiang.pdf), [code](https://github.com/tie-pilot-qxw/syncopate))
- Setting: tensor/sequence-parallel operators inside one NVLink node (AllGather–GEMM, GEMM–ReduceScatter, GEMM–AllReduce, attention, Ring Attention)
- Core idea: a **chunk** is a tunable communication unit between "one tile" and "one whole kernel". It lets a compiler rewrite a single-GPU Triton kernel into a fused multi-GPU operator.

Related: [distributed training communication](https://github.com/vInThePrism/ai-infra-notebook/blob/main/notes/05-distributed-training-communication.md) covers stream-level hiding (DDP buckets, Megatron overlap); [TP, SP, CP, and ring attention](https://github.com/vInThePrism/ai-infra-notebook/blob/main/notes/04-tp-sp-cp-and-ring-attention.md) covers where these collectives come from.

---

## 1. The problem with kernel-level overlap

Take AllGather → GEMM under 4-way TP. The classic approach splits the GEMM into one sub-GEMM per incoming shard and staggers them against the communication on separate streams:

```text
comm stream:     [AG chunk1][AG chunk2][AG chunk3]
compute stream:  [GEMM 0   ][GEMM 1   ][GEMM 2   ][GEMM 3   ]
```

This leaves three costs on the table:

1. **Launch and sync at every boundary.** Each phase pays a kernel launch and a device-wide synchronization.
2. **Wave quantization.** A smaller GEMM has fewer tiles, so its last wave fills fewer SMs. The paper measures that a GEMM split into many kernels is clearly slower than one kernel that pipelines the same tiles internally.
3. **A communication tail.** At coarse granularity, the last transfer has little or no compute left to hide behind.

> **My understanding**
>
> My first reading was "they put the communication inside the kernel." That is only half right. What never changes is that **computation stays one complete kernel**; it is never split. Where the communication runs is a tunable choice: the copy engine, a separate communication kernel on dedicated SMs, or the same SMs as the compute.

---

## 2. Why the unit of communication matters

Communication efficiency depends on three things at once (the paper's Fig. 2):

- **Tile cadence.** Data becomes available (or is needed) at the rate tiles are produced (or consumed).
- **Transfer size.** Small transfers overlap more finely but pay a fixed cost per signal/wait; large transfers use bandwidth well but leave fewer overlap opportunities.
- **Backend sweet spots.** They differ sharply:

| Backend | Runs on | Strength | Weakness |
|---|---|---|---|
| Copy engine | Dedicated DMA unit, no SMs | Highest bandwidth; free compute | Host-launched (~2–3 µs each); contiguous copies only |
| TMA | SM-issued async instructions | High bandwidth with few SMs (paper: 300+ GB/s from ~16 SMs) | Intra-node point-to-point only; no in-switch reduction |
| Load/store | Ordinary SM instructions | Most flexible; works with NVSHARP reductions | Lower peak; occupies and blocks the issuing warps |

So neither fixed choice is right: per-tile communication (as in Flux) pays too many synchronizations, and per-kernel communication (as in Alpa or Mercury) has the three costs from §1. The paper concludes that the unit of communication needs two properties:

1. **Tunable granularity**, to match tile cadence and backend sweet spots.
2. **A stable interface across backends**, so a schedule is written once and can be realized by any backend.

---

## 3. The chunk abstraction

A **chunk** is a logical block of a tensor that is communicated as a unit, described as `Chunk(sizes, layout, tensor)`. It covers one or more compute tiles.

```text
global tensor (logical)
 └─ chunk   unit of communication; size is tunable; defined by a user, template, or higher-level compiler
     └─ tile    unit of computation; set by the kernel's blocking; aligned to chunks via annotations
physical: each chunk → several copy-engine copies, or TMA instructions, or loads/stores
```

Three points that took me a while:

- **A chunk does not exist at run time.** It is compile-time metadata: a few integers describing a tensor region. At run time there are only ordinary buffers and one flag per chunk.
- **A tile is a unit of work, tied to a data region.** Tile `(pid_m, pid_n)` in a GEMM writes one block of C and reads one row band of A and one column band of B. Syncopate needs exactly this link: which chunk does this tile read or write?
- **A chunk is defined on the tensor, not built out of tiles.** Chunk boundaries are aligned to tile boundaries, but a chunk is a tensor region first. The direction of the dependency depends on the operator:
  - In AG-GEMM, tiles **consume** chunks: a chunk must arrive before its tiles can start.
  - In GEMM-RS, tiles **produce** chunks: all of a chunk's tiles must finish before it can be sent.

Schedules are written with two operator types:

```text
P2P(src_rank, dst_rank, src_chunk, dst_chunk, dependency)        # defined on the source side = push, on the destination side = pull
Collective(type, src_chunk, dst_chunk, ranks, dependency)
```

Here `dependency` is a `(rank, index)` pair meaning "wait for that rank's index-th operation". Chaining these per-rank lists gives a cross-rank dependency DAG. That DAG is enough to express ring AllGather, partitioned AllReduce, and hierarchical swizzled patterns.

> **My understanding**
>
> A chunk does not "mix" the tuning choices together. It is the **shared axis** they are all defined on:
>
> | Knob | What it says about a chunk |
> |---|---|
> | Split factor | how big each chunk is |
> | Backend | how a chunk is moved |
> | Communication SMs | how many resources move it |
> | Chunk order | when a chunk arrives and is computed |
> | Intra-chunk tile order | how its tiles are computed once it arrives |
>
> Because every knob hangs off the same unit, changing one (say, copy engine → TMA) swaps a code template and **reuses the same dependency graph**. Without chunks, communication granularity is welded to compute granularity, and changing it means rewriting the kernel.

---

## 4. What "compiler" means here

Not nvcc and not Triton's own compiler. Syncopate is a **source-to-source** compiler that sits upstream of them:

```text
single-GPU Triton source + chunk-level communication plan
        │  Syncopate: parse → analyze → rewrite → generate → autotune
        ▼
distributed Triton source + communication code + host wrapper
        │  Triton JIT (unchanged)
        ▼
PTX → SASS
```

It is called a compiler because it does what compilers do: it parses, analyzes dependencies, transforms code, generates code, and optimizes. It does not convert data into chunks; it **rewrites code** so the code knows which rows to move and which flag to wait on. A small **runtime** handles what cannot happen at compile time: resetting flags, launching copy-engine transfers, and integrating with PyTorch distributed.

### Inputs

1. **A local Triton kernel** with comment annotations that describe **only its blocking**. The communication plan is not in the comments. From the paper's GEMM example:

   ```python
   # @sy.axis_count M block=BLOCK_SIZE_M        ← tile size along M
   num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
   # @sy.tile_id persistent                     ← which variable is the tile index
   tile_id = start_pid - NUM_SMS
   for _ in range(...):
       tile_id += NUM_SMS                       # persistent kernel: one program per SM
       # @sy.dispatch begin                     ← the tile scheduler Syncopate may replace
       # @sy.pid_map M=pid_m N=pid_n
       pid_m, pid_n = get_pid_mn(tile_id, ...)
       # @sy.dispatch end
       a = a_desc.load([pid_m * BLOCK_M, k])
   ```

   Comments keep the kernel's semantics intact: it still runs on one GPU without Syncopate.

2. **A separate chunk-level communication plan.** It can be written with a Python API, instantiated from a template (1D/2D AllGather or ReduceScatter swizzles), or lowered from a distributed compiler's IR (Alpa, Domino, Mercury). Syncopate **does not choose the parallelization strategy**; it realizes overlap after the strategy is fixed.

### Pipeline, on a toy AG-GEMM

This walk-through is my illustration of the mechanism, not the repository's literal code. Setting: 4 ranks, `M = 4096`, 1024 local rows per rank, `BLOCK_M = 128`, viewed from rank 0.

1. **Parse** the `# @sy` directives to get the tile size, the tile index variable, and the replaceable dispatch region.
2. **Expand the plan** for the concrete world size:
   ```text
   chunk 0: rows [0, 1024)     local
   chunk 1: rows [1024, 2048)  pull from rank 1
   chunk 2: rows [2048, 3072)  pull from rank 2
   chunk 3: rows [3072, 4096)  pull from rank 3
   ```
3. **Derive dependencies** with interval arithmetic: `chunk_of(pid_m) = (pid_m * 128) // 1024 = pid_m // 8`. Nobody writes the dependencies by hand.
4. **Allocate one flag per transferred chunk** in global memory (the implementation uses NVSHMEM).
5. **Rewrite the kernel** in two places:
   - insert `wait(signal[chunk_of(pid_m)])` before the load, which guarantees **correctness**;
   - replace the dispatch region with a chunk-ordered tile schedule, which provides the **performance** (§5).
6. **Generate communication code** from per-backend templates: host copy-engine calls, a separate TMA or load/store kernel on dedicated SMs that writes flags, or transfers fused into the compute kernel with shared-memory barriers.
7. **Generate a host wrapper** that resets flags, starts communication, launches the rewritten kernel, and finishes with a sync. It is called like the original operator, plus rank and world-size arguments.

At run time, the communication side moves chunk *k* and sets its flag. Each SM takes a tile, finds its chunk, spins on the flag, loads, and computes. Because the tile order already follows arrival order, the flag is almost always set already.

> **My understanding**
>
> Each step is simple: parse comments, divide integers, splice text, fill templates. Flux and TritonDistributed ask an expert to do steps 3–7 by hand for every operator and pattern. The contribution is the abstraction that makes these steps **mechanical enough to automate**. Systems papers often look like this: the novelty is the interface that makes the problem simple, not a complicated algorithm.

---

## 5. Reordering tiles to follow chunk arrival

Inserting waits alone gives a correct but slow kernel. The original single-GPU tile order (for example, grouped for L2 reuse) knows nothing about communication. Its first wave is spread over all of A, so many SMs land on chunks that are still in flight and stall, while chunk 0 still has unclaimed work.

Syncopate rewrites the dispatch so tiles of chunk 0 run first, then chunk 1, and so on. Within each chunk, it applies a swizzle to keep locality:

```text
comm:     [chunk1 ][chunk2 ][chunk3 ]
compute:  [chunk0 tiles][chunk1 tiles][chunk2 tiles][chunk3 tiles]
```

The order is **rank-dependent**: rank 1's local chunk is chunk 1, so it starts there.

Prior work fixes the same layout mismatch by **moving data**: an extra reorder pass through global memory. Syncopate **leaves data in place and changes only the visiting order**, which costs a few lines of index arithmetic.

| Mechanism | Guarantees |
|---|---|
| `wait` before a load | correctness: never compute on missing data |
| chunk-ordered tile schedule | performance: the wait almost never blocks |

---

## 6. Autotuning: enumerate and measure, once

The search space covers split factor, backend (copy engine, plus TMA or load/store on dedicated or co-located SMs: five realizations), number of communication SMs, tile sizes, and intra-chunk tile order. Configurations that break hardware limits (minimum efficient copy-engine size, TMA alignment) are pruned. Each remaining candidate is generated, compiled by Triton, run, and timed; the fastest is kept.

Is brute force too slow? Not in the setting that matters:

- **It is an offline, one-time cost per shape and hardware configuration**, the same idea as `@triton.autotune`. A training job reuses the same FFN shapes for every step, so the cost is amortized.
- **Each candidate is cheap**: a light source rewrite plus Triton JIT.
- **The space is small**: a handful of options per axis, nothing like the tens of thousands of candidates in TVM- or Ansor-style search. The paper does not report total tuning time.

**The real limitation is dynamic shapes.** A tuned configuration is specific to one shape, so variable sequence lengths need shape buckets. The paper leaves a full dynamic-shape runtime, and multi-node support, to future work. The design fits fixed-shape training most naturally.

The ablations show every knob matters (all paper-reported, H100):

- **Backend:** for the same schedule, the wrong backend can lose more than half the throughput.
- **Split factor:** non-monotonic; GEMM-AR peaks at 2–3 splits. The splits that are easiest to write by hand (one chunk, or one per rank) are far from the optimum.
- **Communication SMs:** too few leave the links idle, too many starve the compute kernel, and the best count shifts with model size.
- **Tile order alone:** more than a 2× spread.

---

## 7. Results and positioning

All numbers below are **paper-reported**, not reproduced here. Setup: 8× H100 with NVLink (900 GB/s aggregate), CUDA 12.9, NVSHMEM 3.3.9, PyTorch 2.7, operator shapes from Llama-3 and Qwen.

- **Overall:** about 1.3× average end-to-end speedup on multi-GPU workloads, up to 4.7×.
- **AG-GEMM and GEMM-RS vs. hand-written kernels** (ThunderKittens, TritonDistributed, AsyncTP, Flux): about 99.8% of the best baseline on 4 GPUs and about 104% on 8 GPUs.
- **GEMM-AR:** slightly behind TritonDistributed on 7B/8B shapes, ahead on larger models.
- **Attention:** tracks the best baselines on head-parallel attention; the gap widens in Syncopate's favor on longer sequences and 8 GPUs, and it is the best on the most communication-heavy Ring Attention settings.
- **As a backend for distributed compilers:** taking the searched plans of Domino, Alpa, and Mercury unchanged and letting Syncopate generate the kernels lowers latency in every reported case.

| | Overlap granularity | Automatic? |
|---|---|---|
| Alpa, Mercury, Domino | kernel | yes |
| Flux, AsyncTP | tile | no, hand-written per operator |
| FlashOverlap | chunk | no |
| ThunderKittens, TritonDistributed | tile / chunk | no, DSL with manual signal/wait |
| **Syncopate** | **chunk** | **yes**, from a local kernel + plan or template |

---

## 8. Connections

- **Hardware:** the copy engine is the DMA unit you can count with `asyncEngineCount`; TMA, persistent kernels, and wave quantization are the same concepts that show up in GEMM tuning.
- **Measuring overlap:** the [communication note](https://github.com/vInThePrism/ai-infra-notebook/blob/main/notes/05-distributed-training-communication.md#hiding-communication) measures *exposed* communication in Nsight Systems. Syncopate attacks the part of that exposure that comes from kernel boundaries and the trailing transfer.
- **A layered answer to "how do you hide communication?":**
  1. stream-level overlap of whole kernels (DDP buckets, Megatron);
  2. hand-fused tile-level kernels (Flux, AsyncTP);
  3. compiler-generated chunk-level overlap (Syncopate).
- **Open questions I would check in the code:** how the generated dispatch encodes chunk order (arithmetic or lookup table); how co-located SMs split warps between TMA issue and MMA; and how much tuning time a full GEMM-RS sweep actually takes.
