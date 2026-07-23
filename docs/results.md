# Amalfi Acceptance Results

- Date:
- Fleet: <n> laptops (list OS / GPU / RAM / measured mem-bandwidth per node)
- Model: <key> (<size> GB), quantization Q4_K_M
- Split: <tensor-split values> across <hosts>

## Success criteria (spec §2)
- [ ] Model loaded split across >= 2 nodes (does not fit on one) — evidence:
- [ ] Coherent chat output — sample:
- [ ] Batch aggregate tok/s > single-stream tok/s — numbers below

## Benchmark
| mode | concurrency | aggregate tok/s | p50 latency s | p95 latency s | ok/total |
|------|-------------|-----------------|---------------|---------------|----------|
| single | 1 | | | | |
| batch  | 8 | | | | |

## Notes / observations
