# nnU-Net 中的残差编码器预设

使用这些预设时，请引用我们最近关于 3D 医学图像分割中严格验证必要性的论文：

> Isensee, F.<sup>*</sup>, Wald, T.<sup>* </sup>, Ulrich, C.<sup>*</sup>, Baumgartner, M.<sup>* </sup>, Roy, S., Maier-Hein, K.<sup>†</sup>, Jaeger, P.<sup>†</sup> (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv preprint arXiv:2404.09556.

*: 共同第一作者\
<sup>†</sup>: 共同通讯作者

[论文链接](https://arxiv.org/pdf/2404.09556.pdf)

自我们参加 KiTS2019 以来，nnU-Net 就已支持残差编码器 UNet，但一直未受到广泛关注。
随着我们新的 nnUNetResEncUNet 预设的推出，这种情况必将改变 :raised_hands:！特别是在像 KiTS2023 和 AMOS2022 这样的大型数据集上，它们能提供更好的分割性能！

|                        | BTCV  | ACDC  | LiTS  | BraTS | KiTS  | AMOS  |  VRAM |  RT | Arch. | nnU |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-----|-------|-----|
|                        | n=30  | n=200 | n=131 | n=1251| n=489 | n=360 |       |     |       |     |
| nnU-Net (原始) [1]     | 83.08 | 91.54 | 80.09 | 91.24 | 86.04 | 88.64 |  7.70 |  9  |  CNN  | 是 |
| nnU-Net ResEnc M       | 83.31 | 91.99 | 80.75 | 91.26 | 86.79 | 88.77 |  9.10 |  12 |  CNN  | 是 |
| nnU-Net ResEnc L       | 83.35 | 91.69 | 81.60 | 91.13 | 88.17 | 89.41 | 22.70 |  35 |  CNN  | 是 |
| nnU-Net ResEnc XL      | 83.28 | 91.48 | 81.19 | 91.18 | 88.67 | 89.68 | 36.60 |  66 |  CNN  | 是 |
| MedNeXt L k3 [2]       | 84.70 | 92.65 | 82.14 | 91.35 | 88.25 | 89.62 | 17.30 |  68 |  CNN  | 是 |
| MedNeXt L k5 [2]       | 85.04 | 92.62 | 82.34 | 91.50 | 87.74 | 89.73 | 18.00 | 233 |  CNN  | 是 |
| STU-Net S [3]          | 82.92 | 91.04 | 78.50 | 90.55 | 84.93 | 88.08 |  5.20 |  10 |  CNN  | 是 |
| STU-Net B [3]          | 83.05 | 91.30 | 79.19 | 90.85 | 86.32 | 88.46 |  8.80 |  15 |  CNN  | 是 |
| STU-Net L [3]          | 83.36 | 91.31 | 80.31 | 91.26 | 85.84 | 89.34 | 26.50 |  51 |  CNN  | 是 |
| SwinUNETR [4]          | 78.89 | 91.29 | 76.50 | 90.68 | 81.27 | 83.81 | 13.10 |  15 |   TF  | 是 |
| SwinUNETRV2 [5]        | 80.85 | 92.01 | 77.85 | 90.74 | 84.14 | 86.24 | 13.40 |  15 |   TF  | 是 |
| nnFormer [6]           | 80.86 | 92.40 | 77.40 | 90.22 | 75.85 | 81.55 |  5.70 |  8  |   TF  | 是 |
| CoTr [7]               | 81.95 | 90.56 | 79.10 | 90.73 | 84.59 | 88.02 |  8.20 |  18 |   TF  | 是 |
| No-Mamba Base          | 83.69 | 91.89 | 80.57 | 91.26 | 85.98 | 89.04 |  12.0 |  24 |  CNN  | 是 |
| U-Mamba Bot [8]        | 83.51 | 91.79 | 80.40 | 91.26 | 86.22 | 89.13 | 12.40 |  24 |  Mam  | 是 |
| U-Mamba Enc [8]        | 82.41 | 91.22 | 80.27 | 90.91 | 86.34 | 88.38 | 24.90 |  47 |  Mam  | 是 |
| A3DS SegResNet [9,11]  | 80.69 | 90.69 | 79.28 | 90.79 | 81.11 | 87.27 | 20.00 |  22 |  CNN  |  否 |
| A3DS DiNTS [10, 11]    | 78.18 | 82.97 | 69.05 | 87.75 | 65.28 | 82.35 | 29.20 |  16 |  CNN  |  否 |
| A3DS SwinUNETR [4, 11] | 76.54 | 82.68 | 68.59 | 89.90 | 52.82 | 85.05 | 34.50 |  9  |   TF  |  否 |

结果取自我们的论文（见上文），报告值为在每个数据集上进行 5 折交叉验证计算得出的 Dice 分数。所有模型均从头开始训练。

RT：训练运行时间（在 1x Nvidia A100 PCIe 40GB 上测量）\
VRAM：训练期间使用的 GPU VRAM，由 nvidia-smi 报告\
Arch.：CNN = 卷积神经网络；TF = Transformer；Mam = Mamba\
nnU：该架构是否已与 nnU-Net 框架集成并测试（由我们或原始作者完成）

## 如何使用新的预设

我们提供三种新的预设，每种预设针对不同的 GPU VRAM 和计算预算：

- **nnU-Net ResEnc M**：与标准 UNet 配置类似的 GPU 预算。最适合具有 9-11GB VRAM 的 GPU。训练时间：在 A100 上约 12 小时
- **nnU-Net ResEnc L**：需要具有 24GB VRAM 的 GPU。训练时间：在 A100 上约 35 小时
- **nnU-Net ResEnc XL**：需要具有 40GB VRAM 的 GPU。训练时间：在 A100 上约 66 小时

### **:point_right: 我们推荐将 **nnU-Net ResEnc L** 作为新的默认 nnU-Net 配置！ :point_left:**

新的预设如下可用（（M/L/XL）= 任选其一！）：

1. 在运行实验规划和预处理时指定所需的配置：
`nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)`。这些规划器与标准的 2d 和 3d_fullres 配置使用相同的预处理数据文件夹，因为预处理数据是相同的。只有 3d_lowres 不同，并将保存在不同的文件夹中，以允许所有配置共存！如果您只打算运行 3d_fullres/2d 并且已经预处理了这些数据，您可以只运行
`nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)` 以避免再次预处理！
2. 现在，在运行 `nnUNetv2_train`、`nnUNetv2_predict` 等命令时，只需指定正确的计划。所有 nnU-Net 命令的接口都是一致的：`-p nnUNetResEncUNet(M/L/XL)Plans`

新预设的训练结果将存储在一个专用文件夹中，不会覆盖标准的 nnU-Net 结果！所以请放心尝试！

## 超越预设扩展 ResEnc nnU-Net

预设与 `ResEncUNetPlanner` 有两点不同：

- 它们为 `gpu_memory_target_in_gb` 设置了新的默认值，以针对相应的 VRAM 消耗
- 它们取消了 0.05 的批处理大小上限（= 以前一个批处理不能覆盖超过整个数据集 5% 的像素，现在可以任意大）

预设的存在仅仅是为了简化操作，并提供人们可以用来进行基准测试的标准化配置。
您可以轻松调整 GPU 内存目标以匹配您的 GPU，并扩展到超过 40GB 的 GPU 内存。

以下是如何在 Dataset003_Liver 上扩展到 80GB VRAM 的示例：

`nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncM -gpu_memory_target 80 -overwrite_plans_name nnUNetResEncUNetPlans_80G`

如上所述，接下来只需使用 `-p nnUNetResEncUNetPlans_80G`！运行上面的示例会产生一个警告（“您正在使用非标准的 gpu_memory_target_in_gb 运行 nnUNetPlannerM”）。此警告在此处可以忽略。
**在调整 VRAM 目标时，请务必使用 `-overwrite_plans_name NEW_PLANS_NAME` 更改计划标识符，以免覆盖预设计划！**

为什么不使用 `ResEncUNetPlanner` -> 因为那个仍然有 5% 的上限！

### 扩展到多个 GPU

扩展到多个 GPU 时，不要仅仅将 VRAM 的总和指定给 `nnUNetv2_plan_experiment`，因为这可能导致补丁大小过大而无法由单个 GPU 处理。最好让此命令针对单个 GPU 的 VRAM 预算运行，然后手动编辑计划文件以增加批处理大小。您可以使用[配置继承](explanation_plans_files_zh.md)。
在生成的计划 JSON 文件的配置字典中，添加以下条目：

```json
        "3d_fullres_bsXX": {
            "inherits_from": "3d_fullres",
            "batch_size": XX
        },
```

其中 XX 是新的批处理大小。如果 3d_fullres 对于一个 GPU 的批处理大小为 2，并且您计划扩展到 8 个 GPU，则将新的批处理大小设为 2x8=16！
然后，您可以使用 nnU-Net 的多 GPU 设置来训练新的配置：

```bash
nnUNetv2_train DATASETID 3d_fullres_bsXX FOLD -p nnUNetResEncUNetPlans_80G -num_gpus 8
```

## 提出新的分割方法？以正确的方式进行基准测试

在针对 nnU-Net 对新的分割方法进行基准测试时，我们鼓励针对残差编码器变体进行基准测试。为了进行公平比较，请选择与您的方法的 GPU 内存和计算要求最匹配的变体！

## 参考文献

 [1] Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods 18.2 (2021): 203-211.\
 [2] Roy, Saikat, et al. "Mednext: transformer-driven scaling of convnets for medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [3] Huang, Ziyan, et al. "Stu-net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training." arXiv preprint arXiv:2304.06716 (2023).\
 [4] Hatamizadeh, Ali, et al. "Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images." International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021.\
 [5] He, Yufan, et al. "Swinunetr-v2: Stronger swin transformers with stagewise convolutions for 3d medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [6] Zhou, Hong-Yu, et al. "nnformer: Interleaved transformer for volumetric segmentation." arXiv preprint arXiv:2109.03201 (2021).\
 [7] Xie, Yutong, et al. "Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24. Springer International Publishing, 2021.\
 [8] Ma, Jun, Feifei Li, and Bo Wang. "U-mamba: Enhancing long-range dependency for biomedical image segmentation." arXiv preprint arXiv:2401.04722 (2024).\
 [9] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4. Springer International Publishing, 2019.\
 [10] He, Yufan, et al. "Dints: Differentiable neural network topology search for 3d medical image segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.\
 [11] Auto3DSeg, MONAI 1.3.0, [LINK](https://github.com/Project-MONAI/tutorials/tree/ed8854fa19faa49083f48abf25a2c30ab9ac1c6b/auto3dseg)
