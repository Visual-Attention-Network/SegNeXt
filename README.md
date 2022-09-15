# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation (NeurIPS 2022)

The repository contains official Pytorch implementations of training and evaluation codes and pre-trained models for [**SegNext**](https://arxiv.org/abs/).

The code is based on [MMSegmentaion 0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1).


## Citation

```bib

```

## Results

**Notes**: Pre-trained models can be found in [TsingHua Cloud](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/).

### ADE20K

|   Method  |    Backbone     |  Pretrained | Iters | mIoU(ss/ms) | Params | FLOPs  | Config | Download  |
| :-------: | :-------------: | :-----: | :---: | :--: | :----: | :----: | :----: | :-------: |
|  SegNeXt  |     MSCAN-T  | IN-1K | 160K | 41.4/42.2 | 4M | 7G | [config](local_configs/segnext/tiny/segnext.tiny.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/5da98841b8384ba0988a/?dl=1) |
|  SegNeXt  |     MSCAN-S | IN-1K  | 160K |  44.3/45.8  | 14M | 16G | [config](local_configs/segnext/small/segnext.small.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/b2d1eb94f5944d60b3d2/?dl=1) |
|  SegNeXt  |     MSCAN-B  | IN-1K  | 160K |  48.5/49.9 | 28M | 35G | [config](local_configs/segnext/base/segnext.base.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/1ea8000916284493810b/?dl=1) |
|  SegNeXt  |     MSCAN-L  | IN-1K  | 160K |  51.0/52.1 | 49M | 70G | [config](local_configs/segnext/large/segnext.large.512x512.ade.160k.py)  | [TsingHua Cloud](https://cloud.tsinghua.edu.cn/f/d4f8e1020643414fbf7f/?dl=1) |


**Notes**: In this scheme, The number of FLOPs (G) is calculated on the input size of 512 $\times$ 512 for ADE20K by [torchprofile](https://github.com/zhijian-liu/torchprofile) (recommended, highly accurate and automatic MACs/FLOPs statistics).

## Installation
Install MMSegmentation and download ADE20K according to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/v0.24.1/docs/en/get_started.md#installation).


```
pip install timm
cd SegNeXt && python -m pip setup.py develop
```

## Training

We use 8 GPUs for training by default. Run:

```bash
./tools/dist_train.sh /path/to/config 8
```

## Evaluation

To evaluate the model, run:

```bash
./tools/dist_test.sh /path/to/config /path/to/checkpoint_file 8 --eval mIoU
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/get_flops.py /path/to/checkpoint_file --shape 512 512
```


## Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [Segformer](https://github.com/NVlabs/SegFormer) and [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger). Thanks for their authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
