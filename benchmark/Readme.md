# Performance benchmarking

# Run benchmark for single configuration:
`python benchmark.py --backend torch --device cpu --compile True --batch_sizes 1 16 256 --prediction_lengths 32 64 128 --context_lengths 2048 --hardware test`

# Run benchmark on all possible configurations of a system:
`python benchmark.py --all --hardware test`

# Hardware

| Name    | Device                  | Accelerator           | OS           |
| ------- | ----------------------- | --------------------- | ------------ |
| H100    | Cloud                   | NVIDIA H100 80GB HBM3 | Linux Server |
| macbook | MacBook Pro 16 MX303D/A | Apple M4 Max          | macOS 15.6.1 |
