# MVOQ位置编码与ViT短对照

## 当前下一步：补齐实例残差两组

已完成的ResNet原版和ResNet+PE无需重跑。新选项 `model.mvoq_instance_residual`
默认false，开启后每层第一条残差从 `LN(q_obj + attn)` 改为
`LN(q_obj + W_ins*f_ins + attn)`，其余运算不变、不新增权重。
这是候选架构改动，并非已证明有效的修复。

从同样的ImageNet初始化，保持931/253划分、416×720、25query、无dropout、
背景权重0.1、global batch4、同学习率和seed，各训练5轮：

```bash
bash tools/run_architecture_controls.sh resnet_residual resnet_pe_residual
```

前者只开实例残差，后者同时开K/V位置编码与实例残差。服务器执行前，通过Git
同步model/config.py、model/mvoq.py、model/association.py、
tools/prepare_architecture_controls.py、tools/run_architecture_controls.sh。
若希望有新增中心/宽高诊断，一并同步tools/diagnose.py。配置由脚本自动生成，
不需要ZIP或重新上传数据。旧的train.py已兼容新增默认配置字段。

输出为runs/architecture_resnet_residual与runs/architecture_resnet_pe_residual，
脚本结束会自动生成history.jsonl、diagnosis.json、unseen_sequence64.json。
不从旧checkpoint续训，不更改stride、NMS、阈值或query数。

最终比较四组：原版、仅PE、仅残差、PE+残差。如果只有联合组在训练定位和验证
定位上都改善，才支持两者互补；若仅残差已足够，则不能把收益归给PE；若两组仍差，
不继续100轮或叠加其它模块。单seed、5轮仍只用于筛选。

本地16项测试通过，两组64×96真实4样本训练/4样本验证/保存通过；默认关闭残差时，
原版和PE输出均与修改前逐位一致。这些是运行与机制验证，不是性能结论。

## 实现与实验目的

新增 `model.mvoq_position_encoding` 配置：

- `none`：原行为，旧配置默认此值。
- `keys_values`：每个视图按实际特征网格生成固定二维正余弦坐标，K=F_k+PE，V=F_k+PE。Q仍为q_obj+W_ins*f_ins，残差和预测头保持原行为。

将PE放入K便于按位置选择，放入V使输出特征能直接携带位置。它是工程实验方案，不是论文式(11)已经明确指定的操作，也不同于之前临时的只给K加PE测试。不增加可学习参数，位置编码不改变state_dict键。

## 三组实验

| 名称 | backbone | MVOQ位置编码 |
|---|---|---|
| resnet_baseline | ResNet50 | none |
| resnet_pe | ResNet50 | keys_values |
| vit_baseline | ViT-B/16 | none |

第一阶段分别改变位置编码或backbone，不同时改变两者。全部使用ImageNet初始化、seed42、25query、dropout0、background_weight0.1、无增强、全局batch4、AdamW lr1e-4、weight_decay1e-5、5轮，backbone更新。PE两组ResNet在相同seed下参数初始化一致。

共同输入高416×宽720，两者均可被ViT的16像素patch完整覆盖。沿用405高度会使当前ViT卷积patch忽略最下方5行。为了输入相同，ResNet基线也重跑，不能直接复用此前405×720的成绩。数据集直接resize到416×720，归一化标签无需改动。

沿用localization_controls的931组训练、253组同序列留出验证；额外评估序列64的240组。专用配置的test是序列64，不是官方MDMT测试集。

ViT的视图特征为26×45=1170 tokens，ResNet为13×23=299 tokens。替换ViT同时改变了骨干、特征步长及计算量，不能把差异全部归因为Transformer或位置编码。MVFF的注意力开销也会增加，ViT不应预期为加速方案。

## 同步到服务器

同步以下文件，保持项目相对路径：

```text
model/config.py
model/mvoq.py
model/association.py
train.py
tools/prepare_architecture_controls.py
tools/run_architecture_controls.sh
```

服务器继续使用已同步的training/engine.py、tools/prepare_localization_controls.py，以及原有数据和ResNet/ViT ImageNet权重。新配置由脚本自动生成，不必上传本地runs。为方便同步，本地提供一个仅含上述代码的architecture_controls_code.zip，解压位置是服务器项目根目录，会覆盖这六个同名代码文件。

## 先检查GPU运行与ViT显存

在服务器项目根目录：

```bash
SMOKE=1 bash tools/run_architecture_controls.sh resnet_pe vit_baseline
```

使用卡1、2，依次跑两个1轮小样例，各16组训练、8组验证；这一步仅检查运行与速度，零指标不用于判断方法有效性。输出目录以_smoke结尾，与正式短对照隔离。

## 正式短对照

优先比较位置编码，两组各5轮：

```bash
bash tools/run_architecture_controls.sh resnet_baseline resnet_pe
```

ViT小样例速度和显存合适后，再运行：

```bash
bash tools/run_architecture_controls.sh vit_baseline
```

也可一次运行全部三组：`bash tools/run_architecture_controls.sh`。每组自动生成history.jsonl、diagnosis.json、unseen_sequence64.json，目录为runs/architecture_<名称>。同名last.pt存在时会拒绝覆盖。脚本每次重新写入专用配置；若要调参，请先修改生成器或另存独立配置，不要直接编辑后被重新生成覆盖。

不续训旧权重：这些是从同条件初始化的独立对照。旧checkpoint默认none模式仍能加载；旧模型续训时会补齐新增默认字段；不允许把PE开关改掉后直接沿用旧优化器续训。

5轮只用于筛查候选是否更快学习；若两组都很差，短预算不足仍是可能原因。不能据此宣称某方法永远无效。重点比较相同步数的训练框覆盖、验证框覆盖、AP50、回归损失和正确关联；只降低总loss不足以支持改进。骨干比较还应同时报告耗时和显存。最终研究结论需要更长训练、多个seed及完整评估协议。

## 本地验证

- 12项单元测试通过，包含开启PE后对空间重排敏感、有限非零梯度、旧state_dict严格加载。
- 使用真实第15轮权重，none模式与修改前MVOQ代码输出逐位一致。
- ResNet+PE和ViT完成真实4样本训练、4样本验证、保存，输入64×96；这只是可运行性检查。
- 两条路径另做416×720真实图像前向，验证特征网格和输出形状。证据见runs/architecture_local_checks/verification.json。
- CPU检查不能替代服务器双卡CUDA/NCCL检查；因此先执行上面的SMOKE命令。
