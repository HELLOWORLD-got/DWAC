# 固定单组拟合实验

## 直接运行

Git同步后，在服务器项目根目录执行（单卡即可，不使用torchrun）：

```bash
CUDA_VISIBLE_DEVICES=1 bash tools/run_single_group_fit.sh \
  runs/mdmt_car_q25_author/last.pt 500
```

脚本顺序执行两个独立实验，都从同一个第15轮checkpoint重新加载权重，并各自新建AdamW优化器；不是恢复原优化器，也不是让第二组接着第一组训练。

| 调试输入 | manifest索引 | 序列/帧 | GT实例 | 可见框 |
|---|---:|---|---:|---:|
| train | 0 | 23 / 0 | 13 | 18 |
| val | 0 | 22 / 289 | 25 | 37 |

这些值来自当前本地manifest；脚本会记录实际加载样本的序列、帧及ID。验证图在这里被当作拟合输入，因此其拟合成绩不能作为验证泛化结果。原训练权重和诊断结果不会被覆盖。

## 保持与改变的设置

- 作者12个模型文件启动时校验哈希，不修改模型、Hungarian matcher或原始损失。
- checkpoint内部配置为准，不受当前默认配置文件变化影响。
- 关闭裁剪、翻转、颜色增强；图像组只加载一次、固定batch=1，重复优化500步。
- 保留dropout=0.1、原有BN策略、学习率1e-4、weight decay、梯度裁剪等checkpoint配置。全网络中原来可训练的参数继续训练。
- 每25步及第0、1、500步切到eval测量，之后恢复train。比较定位趋势应使用eval指标，训练模式匹配切换还包含dropout噪声。
- 回归梯度在更新前测量；对应的eval指标在此次更新后测量。探针只求L_reg对参数的梯度，不进行额外更新，不替换正常总损失backward。

## 结果文件

自动保存到`runs/single_group_fit_时间/`，下有`train_index0/`与`val_index0/`：

- `summary.json`：初始、最终指标及最佳记录点的匹配平均IoU。
- `history.jsonl`：每个测量点的原始/过滤覆盖、严格匹配框命中、中心和尺寸误差、回归梯度与eval匹配切换。
- `assignments.jsonl`：每个训练步真实GT到query的映射及相邻步切换比例；忽略填充背景槽的交换。
- `step_*.jpg`：GT、全部query、过滤后query对照图。
- `predictions_*.json`：原始预测框/分数/可见性、GT与匹配，便于后续数值核查。
- `experiment.json`：实际配置、输入样本、数据哈希和初始化来源。
- `best.pt`、`last.pt`：调试权重；best按记录点的联合匹配平均IoU选择，绝非验证集最佳权重。

中心误差为Hungarian联合匹配后的`|cx_pred-cx_gt|, |cy_pred-cy_gt|`乘以输入宽、高；尺寸误差同理，为输入像素单位。只统计真实且可见的GT框。附带中心误差相对GT宽高的比例，方便比较小目标。`matched_box_hits`用固定IoU>=0.5且每个GT使用联合匹配query，原始覆盖则允许非排他匹配，二者不是同一指标。

`regression_gradients`按模块汇总L_reg梯度的连通数、非零数、L2范数和有限性。分类头、置信度头不直接参与框回归损失，没有该项梯度是预期行为；不能要求所有参数都有L_reg梯度。

## 如何判断下一步

- 中心/尺寸误差持续下降、匹配IoU及命中明显提高：至少证明该样本存在可学习路径，但不能证明整个数据管线和泛化都正确。
- loss下降但框命中不提高：结合中心、尺寸、预测数组和匹配切换定位具体退化。
- 主干/MVFF/GIQ/MVOQ框分支回归梯度缺失、非有限或长期接近零：优先检查对应路径。非零梯度本身不等于有效学习。
- train单组能拟合、val单组难拟合：优先分析尺度/密度/可见性差异，不能直接解释为泛化，因为两者都正在被拟合。
- 两组单独都能拟合，但原验证仍差：再考虑跨图泛化、场景依赖和训练配方。
- 500步不是预设合格线。若仍在改善，可以增加预算，但需要从独立目录重跑；脚本没有恢复调试优化器的功能。

若要从ImageNet初始化而不是已有15轮权重做独立对照：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/fit_single_group.py \
  --config configs/mdmt_resnet50.json --split train --index 0 \
  --steps 500 --log-every 25 --device cuda \
  --output runs/single_group_imagenet_train0
```

默认先跑前面的checkpoint初始化实验，避免同时增加实验变量。
