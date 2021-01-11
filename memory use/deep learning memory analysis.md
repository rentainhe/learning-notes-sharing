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

#### 梯度与动量的显存占用
优化器的参数取决于放置在CPU还是GPU上
- SGD: 保存`参数`及其`对应的梯度`, 总共显存占用为`参数占用×2`
- Momentum-SGD: 额外保存`动量`, 显存占用×3
- Adam: `动量`占用显存更多, 显存占用×4

#### Summary
与输入无关的显存占用:
- 参数 `W`
- 梯度 `dW`
- 优化器`动量` (SGD没有动量，momentum-SGD动量数量与梯度相等，Adam优化器动量数量是梯度的两倍)

#### 输入输出的显存占用
- 每一层feature map的形状
- 保存输出对应的梯度用以反向传播
- 显存占用与batch size成正比
- 模型输出不需要存储相应的动量信息
```
显存占用 = 模型显存占用 + batch_size × 每个样本显存占用
```
显存占用并不是和batch_size简单地成正比，尤其是在自身模型比较复杂的情况下

#### Addition
- 输入一般不需要计算梯度
- 神经网络每一层输入输出的结果都需要保存下来，用以反向传播
- nn.ReLU(inplace=True)的情况下, 可以将激活函数的输出直接覆盖保存于模型的输入之中，节省大量显存

### Implemented Article
- [深度学习中的GPU和显存分析](https://zhuanlan.zhihu.com/p/31558973)