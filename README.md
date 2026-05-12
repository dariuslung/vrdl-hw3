# NYCU Visual Recognition using Deep Learning - Spring 2026 - Homework 2

- Student ID: 314561005
- Name: 龍偉亮

## Introduction

This repository contains a PyTorch-based instance segmentation pipeline designed for detecting and segmenting micro-scale medical cells. The system utilizes a Mask R-CNN architecture (ResNet50 FPN V2) configured specifically for dense, high-resolution imagery.

Key features of the pipeline include:

- **Custom Micro-Anchors:** The Region Proposal Network (RPN) is adjusted to generate smaller anchor boxes (`16, 32, 64, 128, 256`) suited for cellular structures rather than large natural objects.
- **Class Imbalance Handling:** Implements a `WeightedRandomSampler` during training to ensure rarer cell classes are sampled appropriately.
- **Large Image Processing:** Employs a sliding window approach (800x800 patches with 200px overlap) during inference to process large `.tif` files without running out of GPU memory.
- **Test-Time Augmentation (TTA):** Applies horizontal flip augmentations during inference and merges predictions using Batched Non-Maximum Suppression (NMS) to improve detection robustness.
- **COCO Format Output:** Converts prediction masks into Run-Length Encoding (RLE) and outputs a standard COCO-compatible JSON results file.

## Environment Setup

The environment is managed via Conda. An `environment.yml` file is provided to install Python 3.10, PyTorch (with CUDA 12.1 support), Torchvision, and all necessary dependencies.

To set up the environment, run the following commands in your terminal:

```bash
# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate ml
```

## Usage

### Training

The `train.py` script handles data loading, augmentation (using `torchvision.transforms.v2`), model initialization, and training loops with mixed-precision (`torch.amp`).

By default, the script expects training data to be located in a `data/train` directory, with subdirectories for each sample containing an `image.tif` and corresponding class masks (`class1.tif`, `class2.tif`, etc.).

To start a training run, execute:

```bash
python train.py --run_name my_first_run
```

**Training Arguments:**

- `--run_name` (Required): The name of the run. This will dictate where the TensorBoard logs and model weights (`best_model.pth`, `latest_model.pth`) are saved under the `outputs/` directory.
    

### Inference

The `inference.py` script applies the trained model to unseen test images, reconstructs the sliding window predictions, and generates a JSON file ready for evaluation or submission.

It expects test images in `data/test_release` and a metadata JSON file mapping image filenames to COCO IDs.

To run inference, execute:

```bash
python inference.py --model_path outputs/my_first_run/best_model.pth
```

**Inference Arguments:**

- `--model_path` (Required): The file path to your trained `.pth` model weights.
- `--test_dir` (Optional): The directory containing the test images. Defaults to `data/test_release`.
- `--meta_json` (Optional): The file path to the JSON mapping test image names to IDs. Defaults to `data/test_image_name_to_ids.json`.
- `--output` (Optional): The filename for the output predictions. Defaults to `test-results.json`.
- `--score_threshold` (Optional): The minimum confidence score required to keep a detection. Defaults to `0.15`.

## Performance Snapshot

<img width="968" height="512" alt="image" src="https://github.com/user-attachments/assets/2c556d92-f7e6-4a74-8f63-0665015cfef7" />
