# 快速定位对照

## 第5轮与第15轮预测对比

同一组、均匀抽取的16训练图像对和16验证图像对，原尺寸405×720，本地CPU重新推理。忽略分数的框命中允许一个预测覆盖多个GT，不是一对一召回。

| 项目 | epoch5 | epoch15 |
|---|---:|---:|
| 训练框命中 / 465 | 37 | 91 |
| 训练最佳IoU中位数 | 0.1020 | 0.2251 |
| 验证框命中 / 445 | 3 | 1 |
| 验证最佳IoU中位数 | 0 | 0 |

训练原来命中的37个GT，15轮时保留17、丢失20，同时新增74个命中。验证原来的3个全部丢失，仅新增1个。验证预测框坐标平均绝对变化0.0225（归一化坐标），所以不是完全不更新，而是更新后仍未对准目标。第一批验证样本MVOQ逐头平均归一化注意力熵在第15轮仍为0.9958–0.9997，接近均匀；此现象本身不足以证明唯一架构根因。

报告：runs/epoch5_vs15/comparison.json；叠图：train.jpg与val.jpg，上排epoch5，下排epoch15，绿色GT、红色过滤后预测。仍然支持定位不足及跨序列泛化失败，不支持全背景塌缩或仅靠降阈值解决。

## 两组短对照

只运行两组，从相同seed=42的ImageNet初始化重新开始，都训练5轮，25query、dropout0、background_weight0.1、无增强、原尺寸、全局batch4。不能续用epoch15：新验证帧曾被旧模型训练过，续训会污染留出验证。

| 组 | ResNet预训练部分 | 2048→256投影 | 训练数据/超参数 |
|---|---|---|---|
| trainable | 更新 | 更新 | 两组相同 |
| frozen | 冻结 | 更新 | 两组相同 |

冻结随机初始化的投影会引入额外因素，因此增加了training.freeze_backbone_stem选项，只冻结预训练stem/layer1–4。默认未启用，原实验行为不变；续训时禁止更改该设置。新配置的model.freeze_backbone保持false。

共同训练清单931组；同序列留出验证253组，覆盖23/39/58/63；从每个序列末尾留出20%，与训练之间排除至少20个原始帧索引。原始序列64的240组作为额外未见序列评估。

**这些专用配置的test指向序列64，不是官方MDMT测试集。**同序列评估仍可能共享静态目标或相似背景，时间间隔不保证统计独立；它用来诊断同布局学习，不能充当论文基准。

## 上传与运行

服务器同步这些文件/目录并保持相对结构：

- train.py
- training/engine.py
- tools/run_localization_controls.sh
- tools/prepare_localization_controls.py（启动时缺清单会自动调用）
- configs/localization_controls/（两份已生成配置）
- data/processed/MDMT/localization_controls/（可选；缺失时在服务器自动生成）

服务器原有数据、ImageNet权重、tools/diagnose.py、evaluate.py继续使用；无需上传本地审计runs。原实验检查点保留。

在服务器项目根目录运行：

```bash
bash tools/run_localization_controls.sh
```

脚本缺少清单或配置时，会使用服务器原有的 `runs/mdmt_sparse_q25_no_dropout/config.json` 和它指向的训练清单自动生成划分。`CONTROL_SOURCE_CONFIG` 可指定其他源配置，必须指向原始稀疏训练清单，不能指向本次生成的划分。

脚本默认用卡1、2，沿用已验证通信设置与端口29643，依次运行trainable/frozen各5轮。若改卡：

```bash
CUDA_VISIBLE_DEVICES=1,2 bash tools/run_localization_controls.sh
```

每轮自动验证253组同序列留出图像；每组结束后抽查64组train/val，并对序列64全240组另行评分。输出在runs/localization_trainable和runs/localization_frozen。已有同名last.pt时训练会拒绝覆盖，应换新输出目录，不要删除旧结果来强行重跑。

相较完整100轮，这两组很短。参考此前服务器每轮约一分钟的稀疏实验，预计合计数分钟到十分钟，实际以进度条为准。未在本地启动长训。

## 如何判断

1. 正常更新组：若同序列留出明显好于序列64，支持对未见相机布局泛化差。若两者都差，继续检查定位优化，不能只归因于跨序列差异。
2. 冻结组：若相同训练预算下，同序列与序列64定位都明显优于正常更新组，说明预训练特征更新方式值得调整；接下来再对照较小backbone学习率，而不是直接认定永久冻结最佳。
3. 若两组都差，5轮短预算本身仍可能不足。查看训练诊断是否在学习；不要据此直接宣布某个模块损坏或开始100轮长训。
4. 比较框覆盖、AP50、回归损失、true_associations；不单看总loss。不同split的框总数不同，用比例比较。

完成后提供两组history.jsonl、diagnosis.json和unseen_sequence64.json，足够先判断方向，暂时不必传大权重。

## 本地验证

两组在同seed下所有初始化参数逐张量相同；真实两样本前后向检查通过：正常组ResNet有梯度，冻结组ResNet无梯度，两个组投影都有梯度。清单全部满足25query容量、路径可加载、训练/验证帧间隔断言。shell语法及现有回归测试通过。CUDA双卡执行仍需服务器验证。
