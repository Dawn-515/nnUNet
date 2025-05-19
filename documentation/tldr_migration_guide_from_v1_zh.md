# nnU-Net V1 迁移指南 TLDR

- nnU-Net V2 可以与 V1 同时安装。它们不会相互干扰。
- V2 所需的环境变量名称略有不同。请阅读[此文](setting_up_paths_zh.md)。
- nnU-Net V2 数据集命名为 DatasetXXX_NAME，而不是 Task。
- 数据集具有相同的结构（imagesTr、labelsTr、dataset.json），但我们现在支持更多的[文件类型](dataset_format_zh.md#supported-file-formats)。dataset.json 文件已简化。请使用 nnunetv2.dataset_conversion.generate_dataset_json.py 中的 `generate_dataset_json`。
- 注意：标签现在不再声明为 value:name，而是 name:value。这与[分层标签](region_based_training_zh.md)有关。
- nnU-Net v2 命令以 `nnUNetv2...` 开头。它们的工作方式基本相同（但不完全相同）。只需使用 `-h` 选项即可。
- 您可以使用 `nnUNetv2_convert_old_nnUNet_dataset` 将 V1 原始数据集传输到 V2。您无法传输已训练的模型。请继续使用旧版 nnU-Net 对这些模型进行推理。
- 以下是您最可能使用的命令（按此顺序）：
  - `nnUNetv2_plan_and_preprocess`。示例：`nnUNetv2_plan_and_preprocess -d 2`
  - `nnUNetv2_train`。示例：`nnUNetv2_train 2 3d_fullres 0`
  - `nnUNetv2_find_best_configuration`。示例：`nnUNetv2_find_best_configuration 2 -c 2d 3d_fullres`。此命令现在会在您的 `nnUNet_preprocessed/DatasetXXX_NAME/` 文件夹中创建一个 `inference_instructions.txt` 文件，其中详细说明了如何进行推理。
  - `nnUNetv2_predict`。示例：`nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -c 3d_fullres -d 2`
  - `nnUNetv2_apply_postprocessing`（请参阅 inference_instructions.txt）
