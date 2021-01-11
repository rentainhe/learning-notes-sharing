### Memory Analysis
#### 观察显存占用以及GPU利用率情况
```bash
$ nvidia-smi
$ watch -n 1 nvidia-smi
```
使用`gpustat`工具
```bash
$ pip install gpustat
$ watch --color -n1 gpustat -cpu
```
可以有效观察`温度-利用率-显存占用`情况

#### 存储指标分析
- `1MB` = `1024×1024 B`
- `int8, int16, int32, int64` 分别对应 `1,2,4,8` 字节
- `float16, float32` 分别对应 `2,4` 字节
- 最常用的数值类型`float32`，一个`float32`占用`4Byte`显存
- 一个四维数组`32×3×256×256(B×C×H×W)`, 占用`24M`显存

#### 神经网络显存占用
- 模型 __自身参数__
- 模型的 __输出__

有参数的层会占用显存，这部分显存占用与 __输入无关__, 模型加载后就会占用
- __有参数的层__: `卷积, 全连接, BatchNorm, Embedding...`
- __无参数的层__: `多数的activation(Sigmoid/ReLU), Pooling, Dropout...`

模型参数（不考虑偏置项b）为:
- Linear ( __M->N__ ): `M×N`
- Conv2d ( __Cin, Cout, K__ ): `Cin×Cout×K×K`
- BatchNorm ( __N__ ): `2N`
- Embedding ( __N,W__ ): `N×W`

参数占用显存 = `参数数目 × n`
- n=`4`: `float32`
- n=`2`: `float16`
- n=`8`: `double64`
