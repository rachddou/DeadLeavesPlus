# **VibrantLeaves : A principled parametric image generator for training deep restoration models**

|                                          ![teaser](readme_images/teaser.png)                                          |
| :------------------------------------------------------------------------------------------------------------------: |
| Image denoising comparison of different versions of DRUNet trained on synthetic images vs real-world natural images. |

**VibrantLeaves : A principled parametric image generator for training deep restoration models**

[Raphael Achddou](https://rachddou.github.io/), [Yann Gousseau](https://perso.telecom-paristech.fr/gousseau/), [Said Ladjal](https://perso.telecom-paristech.fr/ladjal/), [Sabine Susstrunk](https://www.epfl.ch/labs/ivrl/people/susstrunk/)

[![arXiv](https://img.shields.io/badge/arXiv-2504.10201-red)](https://arxiv.org/pdf/2504.10201)

## Abstract

Even though Deep Neural Networks (NN) are extremely powerful for image restoration tasks, they have several limitations. They are poorly understood and suffer from strong biases inherited from the training sets. One way to address these shortcomings is to have a better control over the training sets, in particular by using synthetic sets. In this paper, we propose a synthetic image generator relying on a few simple principles. In particular, we focus on geometric modeling, textures, and a simple modeling of image acquisition. These properties, integrated in a classical Dead Leaves model, enable the creation of efficient training sets. Standard image denoising and super-resolution networks can be trained on such datasets, reaching performance almost on par with training on natural image datasets. As a first step towards explainability, we provide a careful analysis of the considered principles, identifying which image properties are necessary to obtain good performances. Besides, such training also yields better robustness to various geometric and radiometric perturbations of the test sets.

| ![dl++](readme_images/im_first_page_2.png)  ![dl++](readme_images/im_first_page.png) | ![dl](readme_images/im_69809436.png)  ![dl](readme_images/im_69810650.png) |
| :------------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|                        **Vibrant Leaves examples**                        |                          Dead Leaves examples                          |

## Usage of this repository

The main contribution here corresponds to the data generation code. We also include training and testing code, which was taken from the original repositories of DRUNet and FFDNet. To generate dead leaves images, we first need to install a few python libraries and create a dictionnary of shapes:

```
sh setup.sh
```

### Data Generation

In order to generate  VibrantLeaves images, run the following command:

```
python image_generation.py
```

This function will generate and store images in the `dataset/vibrantLeaves/` folder. In order to change the parameters of generation feel free to modify the `config/default.yaml` file.

This file is organized as follows:

```yaml
defaults:
  - override hydra/launcher: joblib

shape: ## geometry parameters
  radius_min: 10           # minimum shape radius in pixels
  radius_max: 500          # maximum shape radius in pixels
  power_law_exponent: 3.0  # controls size distribution; higher = more small shapes
  shape_type: "poly"       # "poly" | "disk" | "rectangle" | "mix"
  multiple_shapes: True    # randomly switch between "poly" and "mix" each image

task: 1

texture: ## texture parameters
  enabled: True
  types: ["sin", "freq_noise", "texture_mixes"]
  type_weights: [0.16, 0.67, 0.17]  # sampling probabilities, must sum to 1
  slope_range: [[0.5, 2.5]]         # 1/f^s frequency slope; supports disjoint
                                    # intervals e.g. [[0.5, 1.1], [1.75, 2.4]]
  online_generation: True           # True = generate per shape; False = precompute dict
  apply_warp: True
  random_phase: False
  texture_path: ""
  apply_perspective: True

color: ## color parameters
  use_natural_images: True
  image_dir: "path/to/waterlooDB/"  # path to source images
  grey: False
  partial_images: False

io: ## saving parameters
  path_origin: "datasets/"
  path: "vibrant_leaves/"

post_process: ## depth-of-field and other post-processing
  depth_of_field: True
  blur_type: "gaussian"             # "gaussian" | "lens"
  extra_blur: False
  downscale: False

n_images: 10
image_size: 512
image_type: "dead_leaves"          # "dead_leaves" | "textures"
test: False
```

Hydra allows us to run this code in parralel to save time.

### Datasets

If you don't want to bother with launching data generation, here's a link to dowload the dataset: [UNRELEASED(WIP)]()

### Training

Once the images are generated, you can run the command `.jobs/train.sh`

This will create a directory where the weights are stored inside `TRAINING_LOGS/`

### Testing

#### Denoising

If you just want to test the models you can download the weights on the following [link](https://drive.switch.ch/index.php/s/Bmdq0lOHylwgb9d).

You should also download the [testsets](https://drive.switch.ch/index.php/s/jfh3N5ZNv1KVPpP) and place them in the `datasets/test_sets/` folder.

To test the models, you can run the command `.jobs/test.sh`

This calls the launcher_test.py function with a set of arguments such as the testing dataset and the model to test.

#### Super-resolution

To test the SWIN-IR lightweight super-resolution model, we provide the weights on the following [link](https://drive.switch.ch/index.php/s/uCdAIpnKEfE09xJ).

Please refer to the official implementation for testing/training the models: [swin-IR](https://github.com/JingyunLiang/SwinIR?tab=readme-ov-file).
