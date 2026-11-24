# 车辆25-query：下一步诊断

目的：区分训练集定位尚未学会、未见序列泛化失败、分类/可见性过滤，以及同一query跨视角身份不一致。不修改作者模型，不重训，不使用测试集调参。

Git同步到服务器后，在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 bash tools/run_car_q25_diagnosis.sh runs/mdmt_car_q25_author
```

若本次实验目录不同，仅替换最后的路径。需要同一实验的best.pt与last.pt；使用checkpoint内部配置和权重，不读取当前默认配置，不下载预训练权重。默认CPU不可替代以上服务器CUDA速度。

产物保存在独立的`runs/car_q25_diagnosis_时间/`，不会覆盖历史结果：

- comparison.json：best与last的训练/验证指标、损失、原始框覆盖率对照。
- 每个checkpoint目录的report.json：实际配置、源码/manifest哈希、逐样本索引、逐序列与视角统计。
- JPG：每行一个视角；三列分别为GT、全部原始query框、经过分类及可见性过滤的框。GT显示原始ID，预测显示query索引和分数。

训练集关闭裁剪、翻转和颜色增强，均匀抽取整个manifest的240组，避免只取第一个序列；验证集最多240组，当前95组会全部评估。best/last必须使用相同manifest、图像尺寸和评估阈值，否则拒绝将结果合并比较。逐样本索引固定，便于后续复查。训练集抽样指标不能直接当作完整训练集指标；95组验证全部来自序列22，结论受此限制。

解读顺序：

1. 先看train的AP50、median_best_raw_iou、raw_box_coverage。如果训练集也很低，下一步核对图像与GT对齐，并做固定小样本拟合，先证明当前训练链路能学会定位。
2. 如果train明显高于val，优先检查逐序列可视化、场景差异和数据版本，而不是直接加query或调阈值。
3. raw_box_coverage明显高于kept_box_coverage，说明分类/可见性过滤也有损失；并不说明只调阈值即可恢复AP。
4. shared_gt_with_raw_hit_in_every_view表示各视角存在命中框，允许不同query；raw_same_query_hits要求同一query命中所有可见视角。两者差距大时，同槽身份对应也是问题。
5. raw_same_query_hits到kept_same_query_hits进一步下降，表示过滤影响了原本具有相同query的候选。

覆盖统计允许一个预测命中多个GT，因此是非排他的诊断上界，不是正式检测召回率；MDA/AP仍使用原有评估逻辑。图片用于直接检查GT是否对齐，不能只看汇总值。没有预设性能合格线，也不能根据这一步单独证明论文无法复现。

本地当前没有这次15轮的checkpoint，因此只能验证诊断代码；真实实验数值必须来自服务器上述命令或复制到本地的真实权重。
