# Benchmark Results: Matrix Multiplication 🚀

This report presents benchmark results for matrix multiplication using Criterion.rs.
## Benchmark System  
The benchmark was run on:  
- 💻 **MacBook M2 Pro**  
- 🧠 **8GB RAM**  
- ⚙️ **macOS 15.2 (Build 24C101)**  
- 🛠️ **Rust Version: 1.84.1 (e71f9a9a9 2025-01-27)**  

## Performance Graph
![Benchmark Graph](images/violin.svg)

## Histogram
![Histogram](images/lines.svg)

## 🏆 Benchmark Results  

### **Small Matrices (64x64)**
| Implementation                     | Time (µs) |
|------------------------------------|-----------|
| Naive                              | 569.71 µs |
| Naive In-Register                  | 155.85 µs |
| Good Loop Order                    | 55.75 µs  |
| Tiling (32)                        | 56.04 µs  |
| Multithreaded Tiling (32)          | 127.26 µs |
| Tiling (64)                        | 54.82 µs  |
| Multithreaded Tiling (64)          | 125.46 µs |
| Tiling (128)                       | 54.79 µs  |
| Multithreaded Tiling (128)         | 125.01 µs |

### **Large Matrices (512x512)**
| Implementation                     | Time (ms) |
|------------------------------------|-----------|
| Naive                              | 471.45 ms |
| Naive In-Register                  | 117.81 ms |
| Good Loop Order                    | 13.82 ms  |
| Tiling (32)                        | 10.87 ms  |
| Multithreaded Tiling (32)          | 7.11 ms   |
| Tiling (64)                        | 11.35 ms  |
| Multithreaded Tiling (64)          | 7.65 ms   |
| Tiling (128)                       | 12.14 ms  |
| Multithreaded Tiling (128)         | 9.32 ms   |

### **Very Large Matrices (1024x1024)**
| Implementation                     | Time (ms) |
|------------------------------------|-----------|
| Naive                              | 3.87 s    |
| Naive In-Register                  | 1.05 s    |
| Good Loop Order                    | 81.83 ms  |
| Tiling (32)                        | 80.53 ms  |
| Multithreaded Tiling (32)          | 32.84 ms  |
| Tiling (64)                        | 83.85 ms  |
| Multithreaded Tiling (64)          | 43.17 ms  |
| Tiling (128)                       | 83.14 ms  |
| Multithreaded Tiling (128)         | 43.33 ms  |

## 📊 Observations  
- **Multithreaded tiling significantly improves performance**, especially for **512x512 and 1024x1024** matrices.  
- **Tiling helps optimize cache usage**, with **64x64** and **128x128** tiling giving the best results.  
- **The naive approach becomes unusable for large matrices**, taking nearly 4 seconds for a 1024x1024 multiplication.  

Generated using [Criterion.rs](https://bheisler.github.io/criterion.rs/book/).
