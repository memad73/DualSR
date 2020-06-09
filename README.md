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
python main.py --gt_dir 'path to ground-truth image' --kernel_dir 'path to ground-truth blur kernel'
```
You can use argument ```--debug``` to see PSNR and loss vlaues online during the training

To evaluate DualSR on a dataset, specify the directory that contains LR images:
```eval-dataset
python main.py --input_dir 'path to LR input image' --output_dir 'path to save results'
```

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 
