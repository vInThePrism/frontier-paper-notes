# Frontier Paper Notes

**Study notes on recent AI infrastructure and systems papers: training, communication, kernels, and compilers.**

Each note starts from the question the paper answers, explains the mechanism in my own words, and marks where my first reading went wrong. Results quoted from a paper are labeled as paper-reported; nothing here is a reproduction unless it says so.

## Notes

| Paper | Venue | Question | Note |
|---|---|---|---|
| Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap | OSDI '26 | Why move compute–communication overlap inside a kernel, and how can a compiler generate it automatically? | [notes/2026-osdi-syncopate.md](notes/2026-osdi-syncopate.md) |

## How each note is organized

1. **Question** — the problem in one sentence.
2. **Mechanism** — what the system does, with a small worked example.
3. **My understanding** — boxed corrections to my first reading.
4. **Results** — paper-reported numbers with their hardware and software setup.
5. **Connections** — links to related concepts and open questions to check in the code.

Notes are named `YYYY-venue-system.md`, using the venue year.

## Related repositories

- [AI Infra Notebook](https://github.com/vInThePrism/ai-infra-notebook): first-principles notes on attention, distributed training, and GPU performance.
- [GPU Kernel Lab](https://github.com/vInThePrism/gpu-kernel-lab): operator implementations and kernel measurements.

## License

Notes are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper content belongs to its authors; notes summarize it in original prose and link to the source.
