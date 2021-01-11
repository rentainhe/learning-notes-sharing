### Computation Analysis
#### 1. 常规操作的计算量
- 全连接层: `B × M × N`, B=batch_size, M=input_size, N=output_size
- Conv2d: `B × H × W × Cout × Cin × K × K`
  - `B` 表示 batch size
  - `H × W × Cout` 表示输出feature map的大小
  - `Cin × K × K` 表示计算出feature map中每个点需要的计算量
- Pooling: `B × H × W × C × K × K`
- ReLU: `B × H × W × C`