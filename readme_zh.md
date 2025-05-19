# 欢迎来到全新的 nnU-Net

如果您在寻找旧版本，请点击[这里](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)。

从 V1 版本迁移过来？请查看 [TLDR 迁移指南](documentation/tldr_migration_guide_from_v1_zh.md)。仍然强烈建议您阅读其余的文档 ;-)

## **2024-04-18 更新：全新的残差编码器 UNet 预设可用！**

残差编码器 UNet 预设显著提高了分割性能。
它们适用于各种 GPU 内存目标。保证都是很棒的东西！
阅读更多 :point_right: [这里](documentation/resenc_presets_zh.md) :point_left:

另外，请查看我们关于系统性基准测试医学图像分割最新进展的[新论文](https://arxiv.org/pdf/2404.09556.pdf)。您可能会感到惊讶！

# 什么是 nnU-Net？

图像数据集非常多样化：图像维度（2D、3D）、模态/输入通道（RGB 图像、CT、MRI、显微镜图像等）、
图像大小、体素大小、类别比例、目标结构属性等等，在不同的数据集中有很大的差异。
传统上，对于一个新问题，需要手动设计和优化一个量身定制的解决方案——这个过程
容易出错，不可扩展，而且成功与否很大程度上取决于实验者的技能。即使
对于专家来说，这个过程也绝不简单：不仅有许多设计选择和数据属性需要
考虑，而且它们之间还紧密相连，使得可靠的手动流程优化几乎不可能实现！

![nnU-Net 概览](documentation/assets/nnU-Net_overview.png)

**nnU-Net 是一种能够自动适应给定数据集的语义分割方法。它会分析提供的
训练案例，并自动配置一个匹配的基于 U-Net 的分割流程。您无需具备任何专业知识！
您可以简单地训练模型，并将它们用于您的应用程序**。

发布后，nnU-Net 在 23 个属于生物医学领域的竞赛数据集中进行了评估。尽管与每个数据集的手工解决方案竞争，
nnU-Net 的全自动流程仍在多个公开排行榜上名列第一！从那时起，nnU-Net 经受住了时间的考验：
它继续被用作基线和方法开发框架（[MICCAI 2020 年 10 个挑战赛获胜者中有 9 个](https://arxiv.org/abs/2101.00232)
以及 MICCAI 2021 年 7 个中有 5 个将其方法建立在 nnU-Net 之上，
 [我们凭借 nnU-Net 赢得了 AMOS2022](https://amos22.grand-challenge.org/final-ranking/)）！

使用 nnU-Net 时，请引用[以下论文](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO)：

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

## nnU-Net 能为您做什么？

如果您是一位**领域科学家**（生物学家、放射科医生等），希望分析自己的图像，nnU-Net 提供了一个
开箱即用的解决方案，几乎可以保证在您的个人数据集上提供出色的结果。只需
将您的数据集转换为 nnU-Net 格式，即可享受人工智能的强大功能——无需专业知识！

如果您是一位**人工智能研究员**，正在开发分割方法，nnU-Net：

- 提供了一个非常棒的、开箱即用的基线算法，可供您进行比较
- 可以作为一种方法开发框架，在大量数据集上测试您的贡献，而无需
调整各个流程（例如评估新的损失函数）
- 为进一步针对特定数据集进行优化提供了一个强大的起点。这在参加分割挑战赛时尤其有用
- 为分割方法的设计提供了新的视角：也许您可以找到
数据集属性与最合适的分割流程之间更好的联系？

## nnU-Net 的适用范围是什么？

nnU-Net 专为语义分割而构建。它可以处理具有任意
输入模态/通道的 2D 和 3D 图像。它可以理解体素间距、各向异性，并且即使在类别高度
不平衡的情况下也具有鲁棒性。

nnU-Net 依赖于监督学习，这意味着您需要为您的应用程序提供训练案例。所需的
训练案例数量因分割问题的复杂性而有很大差异。这里无法提供
一个通用的数字！nnU-Net 所需的训练案例并不比其他解决方案多——由于我们广泛使用数据增强，甚至可能更少。

nnU-Net 希望在预处理和后处理过程中能够一次性处理整个图像，因此它无法
处理非常大的图像。作为参考：我们测试了从 40x40x40 像素到 3D 中高达 1500x1500x1500
以及 2D 中从 40x40 到高达约 30000x30000 的图像！如果您的 RAM 允许，总是可以处理更大的图像。

## nnU-Net 是如何工作的？

对于一个新的数据集，nnU-Net 会系统地分析提供的训练案例并创建一个“数据集指纹”。
然后，nnU-Net 会为每个数据集创建多个 U-Net 配置：

- `2d`：一个 2D U-Net（适用于 2D 和 3D 数据集）
- `3d_fullres`：一个在高图像分辨率下运行的 3D U-Net（仅适用于 3D 数据集）
- `3d_lowres` → `3d_cascade_fullres`：一个 3D U-Net 级联，其中首先一个 3D U-Net 在低分辨率图像上运行，
然后第二个高分辨率 3D U-Net 优化前者的预测（仅适用于具有大图像尺寸的 3D 数据集）

**请注意，并非所有 U-Net 配置都会为所有数据集创建。在图像尺寸较小的数据集中，
U-Net 级联（以及 3d_lowres 配置）会被省略，因为全分辨率
U-Net 的补丁大小已经覆盖了输入图像的很大一部分。**

nnU-Net 根据三步法配置其分割流程：

- **固定参数**不进行调整。在 nnU-Net 的开发过程中，我们确定了一个鲁棒的配置（即某些架构和训练属性），
可以一直使用。这包括例如 nnU-Net 的损失函数、（大部分）数据增强策略和学习率。
- **基于规则的参数**使用数据集指纹，通过遵循
硬编码的启发式规则来调整某些分割流程属性。例如，网络拓扑（池化行为和网络架构的深度）
会根据补丁大小进行调整；补丁大小、网络拓扑和批量大小会在给定的 GPU
内存约束下联合优化。
- **经验参数**基本上是反复试验的结果。例如，为给定数据集选择最佳的 U-Net 配置
（2D、3D 全分辨率、3D 低分辨率、3D 级联）以及后处理策略的优化。

## 如何开始？

阅读这些：

- [安装说明](documentation/installation_instructions_zh.md)
- [数据集转换](documentation/dataset_format_zh.md)
- [使用说明](documentation/how_to_use_nnunet_zh.md)

附加信息：

- [从稀疏注释（涂鸦、切片）中学习](documentation/ignore_label_zh.md)
- [基于区域的训练](documentation/region_based_training_zh.md)
- [手动数据拆分](documentation/manual_data_splits_zh.md)
- [预训练和微调](documentation/pretraining_and_finetuning_zh.md)
- [nnU-Net 中的强度归一化](documentation/explanation_normalization_zh.md)
- [手动编辑 nnU-Net 配置](documentation/explanation_plans_files_zh.md)
- [扩展 nnU-Net](documentation/extending_nnunet_zh.md)
- [V2 有哪些不同之处？](documentation/changelog_zh.md)

竞赛：

- [AutoPET II](documentation/competitions/AutoPETII_zh.md)

[//]: # (- [忽略标签]&#40;documentation/ignore_label.md&#41;)

## nnU-Net 在哪些方面表现良好，哪些方面表现不佳？

nnU-Net 在需要从头开始训练的分割问题中表现出色，
例如：具有非标准图像模态和输入通道的研究应用、
生物医学领域的挑战数据集、大多数 3D 分割问题等。我们尚未发现
nnU-Net 的工作原理会失败的数据集！

注意：在标准分割
问题上，例如 ADE20k 和 Cityscapes 中的 2D RGB 图像，微调一个基础模型（在大量
相似图像语料库上预训练，例如 Imagenet 22k、JFT-300M）将提供比 nnU-Net 更好的性能！这仅仅是因为这些
模型允许更好的初始化。nnU-Net 不支持基础模型，
因为它们 1) 对于偏离标准设置的分割问题（参见上述
数据集）没有用处，2) 通常只支持 2D 架构，3) 与我们为每个数据集仔细调整
网络拓扑的核心设计原则相冲突（如果拓扑发生改变，就无法再迁移预训练权重！）

## 旧的 nnU-Net 怎么样了？

旧 nnU-Net 的核心是在 2018 年参加医学分割
十项全能挑战赛期间在短时间内拼凑而成的。因此，代码结构和质量并非最佳。许多功能
是后来添加的，并且不太符合 nnU-Net 的设计原则。总的来说，非常混乱。而且使用起来很烦人。

nnU-Net V2 是一次彻底的改革。“删除所有内容并重新开始”的那种。所以一切都变得更好了
（作者认为哈哈）。虽然分割性能[保持不变](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing)，但添加了许多很酷的东西。
现在，将其用作开发框架以及手动微调其配置以适应新
数据集也变得更加容易。重新实现的一个重要驱动因素也是 [Helmholtz Imaging](http://helmholtz-imaging.de) 的出现，
促使我们将 nnU-Net 扩展到更多的图像格式和领域。请[在此处](documentation/changelog.md)查看一些亮点。

# 致谢

<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net 由 [Helmholtz Imaging](http://helmholtz-imaging.de) 的应用计算机视觉实验室 (ACVL)
和[德国癌症研究中心 (DKFZ)](https://www.dkfz.de/en/index.html) 的[医学图像计算部](https://www.dkfz.de/en/mic/index.php) 开发和维护。
