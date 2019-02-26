# Simplified CycleGAN Implementation in PyTorch

Great thanks to [Jun-Yan Zhu](https://github.com/junyanz) et al. for their contribution of the CycleGAN paper. Original project and paper - 

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="600"/>

The code is adopted from the authors' implementation but simplified into just a few files. If you use this code for your research, please cite Jun-Yan Et al.:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)

Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros). In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Install [PyTorch](http://pytorch.org) 0.4+ (1.0 tested) with GPU support.
- Clone this repo:
```bash
    git clone https://github.com/cy-xu/simple_CycleGAN
    cd simple_CycleGAN
```
- The command `pip install -r requirements.txt` will install all required dependencies.

### CycleGAN train/test
- Download a CycleGAN dataset from the authors (e.g. horse2zebra):
```bash
    bash ./util/download_cyclegan_dataset.sh horse2zebra
```

- Train a model (different from original implementation):
```bash
    python simple_cygan.py train
```
  - Change training options in `simple_cygan.py`, all options will be saved to a txt file
  - A new directory by name of `opt.name` will be created inside the checkpoints directory
  - Inside `checkpoints\project_name\` you will find 
    - `checkpoints` for training processing results
    - `models` for saved models
    - `test_results` for running `python simple_cygan.py test` on testing dataset

- Test the model:
```bash
    python simple_cygan.py test
```
## Use your own Dataset
Follow the naming pattern of `trainA`, `trainB`, `testA`, and place them in `datasets\your_dataset\`. You can also change directories inside `simple_cygan.py`.

## Citation
If you use this code for your research, please cite Jun-Yan et al's papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Related Projects
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) |
[iGAN](https://github.com/junyanz/iGAN) |
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/)**
