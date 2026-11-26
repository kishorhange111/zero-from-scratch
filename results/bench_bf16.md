GPT with 25.5M parameters (d = 512, 8 layers), bf16 compute, fp32 master, gloo on one machine, one thread per rank, micro-batch 2 × 64 tokens per rank.

| p | stage | step (ms) | in collectives | in Adam | MB sent per step | formula | engine MB per rank | formula | peak gathered MB | RSS before steps (MB) | peak RSS in steps (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 6857 | 727 | 1409 | 76.5 | 76.5 | 408.2 | 408.2 | 0.0 | 650 | 722 |
| 4 | 1 | 5131 | 1143 | 257 | 76.5 | 76.5 | 178.6 | 178.6 | 0.0 | 421 | 468 |
| 4 | 2 | 2440 | 361 | 151 | 76.5 | 76.5 | 140.3 | 140.3 | 0.0 | 382 | 433 |
| 4 | 3 | 2532 | 471 | 154 | 114.8 | 114.8 | 102.1 | 102.1 | 6.3 | 344 | 402 |
| 4 | 3 + prefetch | 2368 | 268 | 146 | 114.8 | 114.8 | 102.1 | 102.1 | 12.6 | 344 | 408 |
