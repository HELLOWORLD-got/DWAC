# MVOQ 完整位置通路版本

本版本将位置处理统一在一个配置中，不再要求用户逐个位置系数做试验。它是明确的架构改进实现，尚未被证明能恢复论文指标，也不是论文式(11)的逐式实现。

## 前向

每个视图：

```text
visual = LayerNorm(flatten(F_k))
position = sine_cosine_2d(H, W)
position = position - mean_over_spatial_tokens(position)
position = position / max(global_RMS(position), epsilon)
alpha = sigmoid(learnable_logit)  # 初始0.1，范围(0,1)
K = V = visual + alpha * position
Q_i = object_query_i + W_ins * instance_feature_i
x_i = LayerNorm(Q_i + Attention(Q_i, K, V))
object_feature_i = LayerNorm(x_i + FFN(x_i))
```

处理了以下问题：视觉幅度与PE幅度没有统一标度；PE中不随空间变化的分量；实例信息只有注意力权重这条间接通路。视觉LayerNorm初始RMS约1，空间编码归一化后RMS为1，实际位置项初始RMS为0.1。后续LayerNorm参数和位置系数均学习，不宣称二者训练后比例永远固定。

位置编码按每个视图实际H/W生成，横纵轴顺序与flatten一致；不把不同相机的相同像素坐标当作同一个物理位置。所有MVOQ层都使用该视图的完整位置输入，不重复累加PE。1×1网格的位置项安全为0，不产生除零。

这提供明确的身份和空间输入，不保证注意力必然聚焦正确目标，也不保证验证性能改善。未更改stride、query容量、匹配、损失或阈值。

## 配置与同步

唯一新配置：configs/mdmt_resnet50_spatial.json。

```json
"mvoq_position_encoding": "normalized_keys_values",
"mvoq_instance_residual": true
```

保留已有none/keys_values及残差开关的行为，旧检查点仍严格按原配置加载；四种旧组合输出逐位一致。新模式新增LayerNorm参数与位置系数，需从ImageNet初始化新训练，不沿用旧优化器。

使用Git同步这些本次修改（以及所需现有文件），不需要ZIP：

- model/mvoq.py
- train.py（记录每轮mvoq_position_scale）
- tools/diagnose.py（报告实际归一化视觉与实际缩放位置项的RMS）
- tools/prepare_spatial_config.py
- tools/run_spatial_training.sh
- configs/mdmt_resnet50_spatial.json（脚本也会自动生成）
- tests/test_mvoq_position.py、本说明

服务器需已有当前model/config.py/association.py、training/engine.py、tools/prepare_localization_controls.py，它们包含前面已同步的模式/实例残差支持。

## 一条运行命令

服务器项目根目录：

```bash
bash tools/run_spatial_training.sh
```

默认卡1、2、global batch4、FP32、15轮，沿用已验证的NCCL设置与端口。脚本缺清单时自动用原有稀疏实验配置生成，随后自动训练、诊断和评估，不要求人工中途调整位置参数。

沿用当前931组训练、253组同序列留出验证、416×720、25query、无增强、dropout0、background_weight0.1，方便与旧结果比较；额外评估原始稀疏未见序列64共240组。这仍是定位开发划分，不是完整MDMT论文协议。此配置的test=序列64，不是官方测试集。

输出runs/mdmt_resnet50_spatial，含last.pt、best.pt、history.jsonl、diagnosis.json、unseen_sequence64.json。history额外记录mvoq_position_scale。已有同名last.pt时拒绝覆盖。

## 验证范围

- 18项回归测试通过。
- 检查新增位置项确实改变空间重排后的输出，实例条件残差保留身份输入。
- 位置系数、视觉归一化和实例特征都有有限非零梯度。
- 真实4训练/4验证样例在64×96完成训练、验证、保存。
- 真实416×720图像完整模型回归、反向检查通过；原始记录位于runs/spatial_implementation_checks/verification.json。
- 旧none/keys_values与残差开/关共四组合，严格载入state_dict并逐位输出对比通过。
- CPU验证不等于双卡CUDA验证，更不等于证明收敛。后续主要检验单个新版本的实际训练成绩，不再把运行检查当作性能修复。
