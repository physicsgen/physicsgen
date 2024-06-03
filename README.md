## Abstract

The image-to-image translation abilities of generative learning models have recently made significant progress in the estimation of complex (steered) mappings between image distributions. While appearance based tasks like image in-painting or style transfer have been studied at length, we propose to investigate the potential of generative models in the context of physical simulations. Providing a dataset of 300k image-pairs and baseline evaluations for three different physical simulation tasks, we propose the following basic research questions: 
- Are generative models able to learn complex physical relations from input-output image pairs? 
- What speedups can be achieved by replacing differential equation based simulations?
- How can the physical correctness of model outputs be enforced?


![teaserfig](figures/TeaserFig.svg "Teaser Figure")



## Download Datasets

The dataset used for evaluation is publicly available and published via Zenodo, ensuring easy access and reproducibility of our research findings [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11401239.svg)](https://doi.org/10.5281/zenodo.11401239).


## Code for baseline experiments

The code is located in the [GitHub repository](https://github.com/physicsgen/physicsgen).

**Project Structure:**

```
project_root/
│
├── data/                         # Dedicated data folder
│   └── urban_sound_25k_baseline/ # Download this via provided DOI
│       ├── test/
│       │   ├── test.csv
│       │   ├── soundmaps/
│       │   └── buildings/
│       │
│       └── pred/                 # Your predictions
│           ├── y_0.png
│           └── ...
│
└── eval_scripts/
    ├── lens_metrics.py
    └── sound_metrics.py
```

The indexing system for predicted sound propagation images in the `pred` folder aligns directly with the `test.csv` dataframe rows. Each predicted image file, named as `y_{index}.png`, corresponds to the test data's row at the same index, with index 0 referring to the dataframe's first row.

### Sound Propagation Evaluation Script

**Description:**
Evaluates sound propagation predictions by comparing them to ground truth noise maps, including Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) errors.

**Usage:**
```sh
python sound_metrics.py --data_dir data/true --pred_dir data/pred --output evaluation.csv
```

**Arguments:**
- `--data_dir`: Directory containing true sound maps and `test.csv`.
- `--pred_dir`: Directory containing predicted sound maps.
- `--output`: Path to save the evaluation results.


### Lens Evaluation Script

**Description:**
Evaluates the accuracy of facial landmark predictions by comparing them to ground truth images.

**Usage:**
```sh
python lens_metrics.py --data_dir data/true --pred_dir data/pred --output results/
```

**Arguments:**
- `--data_dir`: Directory containing true label images and `test.csv`.
- `--pred_dir`: Directory containing predicted landmark images.
- `--output`: Directory to save the results.

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)