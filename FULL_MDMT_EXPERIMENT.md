# 完整MDMT中心监督实验

## 本轮目的与设置

将已能学习定位和关联的中心监督版本扩展到完整MDMT，检查更丰富的训练场景能否改善泛化。本轮主要改变数据覆盖与必要的query容量，不继续堆叠架构改动。

| 项目 | 设置 |
|---|---|
| 训练 | data/processed/MDMT/train.jsonl，12026组，25个序列 |
| 验证 | val.jsonl，1930组，5个独立序列 |
| 官方测试 | test_official.jsonl，5883组，14个独立序列 |
| queries | 256；三份清单最多191/221/201实例，无GT截断 |
| 模型 | ResNet50，归一化PE、实例残差、空间参考点+中心偏移 |
| 损失 | 原分类/框/可见性损失，背景权重.1，中心热图权重.25 |
| 输入 | 高416×宽720 |
| 优化器 | AdamW，lr1e-4，weight_decay1e-5 |
| batch/设备 | 全局4，双卡每卡2；FP32 |
| 其它 | dropout0，BN统计冻结，backbone更新，增强保持关闭 |
| 初始训练阶段 | 15轮，从ImageNet重新初始化 |

这是改进模型的完整数据实验，不是作者原始设置的严格复现。原931/253稀疏划分指标与新1930组验证指标不在同一评估协议，不能直接横比或把差异单独归因于某模块。

## 数据与指标检查

prepare_full_center.py扫描全部清单，检查256容量、类别、ID唯一性、框/可见性维度、归一化坐标、可见实例，以及每个图像路径存在；确认train/val/test序列完全不重叠，保存清单SHA256。没有重新划分或截断完整GT。

**序列64属于完整训练集。**本轮之后不能再用它当作未见序列，验证改用原始1930组清单。启动脚本只读取官方测试的元数据进行审计，不用其训练或自动评估。

AP继续使用现有标准COCO maxDets=100。完整验证有1110/3860张视图图像超过100个GT，这会限制密集图上的AP/召回上限；MDA/UCR使用全部256 query，不受maxDets=100限制。本轮保持评分规则不变，需在报告中写清楚。MDA还保留现有序列宏平均和空关联空GT约定；不要将旧legacy官方脚本结果当作完全相同的指标。

## Git同步与一条启动命令

提交推送本次新增/修改文件，服务器git pull：

```text
tools/prepare_full_center.py
tools/run_full_center_training.sh
configs/mdmt_resnet50_center_full.json
configs/mdmt_resnet50_center_full_smoke.json
train.py
FULL_MDMT_EXPERIMENT.md
README.md
```

模型及training目录继续使用此前已经同步的中心监督版本。服务器保留现有完整数据、三个完整清单和ImageNet权重；不必从本地上传新的数据清单子集或审计runs，生成器会在服务器自动准备。

从服务器项目根目录执行：

```bash
bash tools/run_full_center_training.sh
```

脚本默认使用卡1、2、NCCL_P2P_DISABLE=1、端口29643：

1. 审计全量数据并生成配置。
2. 首次自动执行密集GPU检查：256query、最密集的16组训练/8组验证、原尺寸、1轮、workers0。检查通过的last.pt保存在独立目录；再次运行已有此检查点时跳过。这个小检查不用于评估模型效果。
3. 从ImageNet初始化完整模型，训练完整12026组，每轮验证完整1930组，共15轮；没有limit-train或limit-val。
4. 自动用最后检查点抽查train/val各64组，生成诊断；不自动评估官方测试。

密集检查的权重不会用于后续完整训练。旧25query的GIQ/object query形状与256不同，不能用--resume直接续上。完整训练输出新目录，不覆盖此前稀疏结果。

如只想执行GPU密集检查：`SMOKE=1 bash tools/run_full_center_training.sh`。默认完整命令已经包含它，无需另跑一次。

## 输出和续训

```text
runs/mdmt_resnet50_center_full/
  config.json
  invocation.json
  history.jsonl
  last.pt
  best.pt
  data_audit.json
  diagnosis.json
```

另有runs/mdmt_resnet50_center_full_gpu_check用于启动前检查。训练配置与审计报告保存在输出目录，方便检查query容量及数据来源。

保持完整实验设置续训到总计30轮：

```bash
RESUME=1 EPOCHS=30 bash tools/run_full_center_training.sh
```

这里30是总轮数，不是追加30轮。必须从该完整实验last.pt续训；不能把旧稀疏实验混入。同名last.pt已存在但没有指定RESUME时，脚本会提前拒绝覆盖。

每轮完整验证会比此前253组验证明显更久，256query的匹配/解码也更贵。用第一轮实际train_seconds和elapsed_seconds估计余下时间，不沿用旧稀疏实验每轮40秒的估算。

## 本阶段观察与最终测试

关注train_losses的分类/回归/中心/可见性分项，完整验证的AP50、MDA、true_associations，以及诊断中的原始框覆盖、双视角共同覆盖和分序列表现。总损失包括中心项，不能直接与原版总loss比较。

这阶段用于检验数据覆盖是否改善泛化，不承诺15轮收敛，也不因第1轮指标低就宣布失败。若趋势支持续训，可使用上述RESUME命令；不要同时更换骨干、stride、阈值和学习率。

最终训练方案确定后，再执行官方测试：

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
  --checkpoint runs/mdmt_resnet50_center_full/best.pt \
  --split test --device cuda --batch-size 4 --workers 0 \
  --output runs/mdmt_resnet50_center_full/test_metrics.json
```

本轮test实际指向完整官方测试清单，不是此前的序列64。

## 本地验证证据

- 全量清单与图像路径审计通过，容量和序列独立性检查通过。
- 真实最密集训练样本191实例、验证样本221实例，在416×720、256query下完整前后向通过，所有需训练参数梯度有限。记录在runs/full_center_local_checks/dense_forward_backward.json。
- 最密集子集上的4训练/2验证小流程在64×96完成训练、验证和保存，验证报告包含中心loss；用于检查流程，不是性能实验。
- 23项既有回归测试通过，脚本bash语法检查通过。
- 本地没有A6000，实际GPU显存和通信由脚本首段密集检查验证。
