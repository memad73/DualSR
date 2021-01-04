# DualSR: Zero-Shot Dual Learning for Real-World Super-Resolution

This repository is the official implementation of "DualSR: Zero-Shot Dual Learning for Real-World Super-Resolution".

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate dualsr_env
```

## Datasets

You can download datasets mentioned in the paper from the following links.
- [DIV2KRK](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip)
- [NTIRE2017 track 2](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip)
- [RealSR](https://github.com/csjcai/RealSR)

## Evaluation

To super-resolve an image using DualSR, put the image in 'test/LR' folder and run:
```eval
python main.py
```

If you want to get PSNR values, you need to provide ground-truth image and/or ground-truth blur kernel directories:
```eval-gt
python main.py --gt_dir 'path to the ground-truth image' --kernel_dir 'path to the ground-truth blur kernel'
```
You can use argument ```--debug``` to see PSNR and loss values online during the training

To evaluate DualSR on a dataset, specify the directory that contains LR images:
```eval-dataset
python main.py --input_dir 'path to the LR input images' --output_dir 'path to save results'
```

## Results

Our model achieves the following performance values (PSNR / SSIM) on DIV2KRK, Urban100 and NTIRE2017 datasets:

| Model name         | DIV2KRK         | Urban100        | NTIRE2017        |
| ------------------ |---------------- |---------------- | ---------------- |
| DualSR             |  30.92 / 0.8728 |  25.04 / 0.7803 |  28.82 / 0.8045  |

All PSNR and SSIM values are calculated using 'Evaluate_PSNR_SSIM.m' script provided by [RCAN](https://github.com/yulunzhang/RCAN).
## Acknowledgement

The code is built on [KernelGAN](https://github.com/sefibk/KernelGAN). We thank the authors  for sharing the codes.
