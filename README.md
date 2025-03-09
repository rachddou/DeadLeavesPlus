# Dead Leaves++ : Bridging the gap between synthetic and natural images for deep image restoration

|                                          ![teaser](readme_images/teaser.png)                                          |
| :------------------------------------------------------------------------------------------------------------------: |
| Image denoising comparison of different versions of DRUNet trained on synthetic images vs real-world natural images. |

**Dead Leaves++ : Bridging the gap between synthetic and natural images for deep image restoration**

[Raphael Achddou](https://rachddou.github.io/), [Yann Gousseau](https://perso.telecom-paristech.fr/gousseau/), [Said Ladjal](https://perso.telecom-paristech.fr/ladjal/), [Sabine Susstrunk](https://www.epfl.ch/labs/ivrl/people/susstrunk/)

## Abstract

Even though Deep Neural Networks (NN) are extremely powerful for image restoration tasks, they have several defects. **(1)** they generalize poorly to unseen image modalities, **(2)** they are strongly biased, **(3)** they are hard to interpret. Achddou et al. proposed to address these problems by replacing standard training datasets with synthetic Dead Leaves images, achieving reasonable performance but not nearly as good as the original methods. We identified that this stochastic image model lacked three key properties: geometry, textures, and depth. In this paper, we present the **Dead Leaves++** model which incorporates this properties. Experimental results show that for both Image Denoising and Single-Image Super-Resolution, training state-of-the-art architectures on such images closes the gap from $1.4 \sim 2$dB to $0.5$dB in terms of PSNR, and greatly improves visual quality. We further show that denoisers trained on our synthetic images are more robust to slight distortions. Finally, we carefully analyze which image properties are necessary to achieve good image restoration results.

| ![dl++](readme_images/im_first_page_2.png)  ![dl++](readme_images/im_first_page.png) | ![dl](readme_images/im_69809436.png)  ![dl](readme_images/im_69810650.png) |
| :------------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|                         **Dead Leaves++ examples**                         |                          Dead Leaves examples                          |

## Usage of this repository

The main contribution here corresponds to the data generation code. We also include training and testing code, which was taken from the original repositories of DRUNet and FFDNet. To generate dead leaves images, we first need to install a few python libraries:

```
pip install -r requirements.txt

```

### Data Generation

In order to generate the final Dead Leaves++, run the following command:

```
python image_generation.py
```

This function will generate and store images in the `dataset/dead_leaves_++` folder. In order to change the parameters of generation feel free to modify the `config/default.yaml` file.

This file is organized as follows:

```
defaults:
  - override hydra/launcher: joblib
shape: ## geometry parameters
  rmin : 10
  rmax : 1000
  alpha: 3.0
  shape_type : "poly"
  multiple_shapes : True

task: 1

texture: ## texture parameters
  texture: True
  texture_types: ["gradient","freq_noise","texture_mixes"]
  texture_gen: True
  warp: True
  rdm_phase: False
  texture_path: ""
  perspective: True

color: ## color parameters
  natural: True
  color_path: "path/to/waterlooDB/"
  grey: False
  partial_images: False

io: ## saving parameters
  path_origin: "datasets/"
  path: "dead_leaves_++/"

post_process: ## depth-of-field and other postprocessing functions
  downscaling: True
  dof: True
  blur_type: "lens"
  blur: False

number : 10
size : 1000
image_type : "dead_leaves" 
test: False
```

Hydra allows us to run this code in parralel to save time.

### Datasets

If you don't want to bother with launching data generation, here's a link to dowload the dataset: [UNRELEASED(WIP)]()

### Training

Once the images are generated you can run the command `.jobs/train.sh`

This will create a directory where the weights are stored inside TRAINING_LOGS/

### Testing

If you just want to test the models you can download the weights on the following [link](todo).

To test the models, you can also run the command `.jobs/test.sh` 

This calls the launcher_test.py function with a set of arguments such as the testing dataset and the model to test.



---
