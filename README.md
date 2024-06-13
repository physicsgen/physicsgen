## Abstract

The image-to-image translation abilities of generative learning models have recently made significant progress in the estimation of complex (steered) mappings between image distributions. While appearance based tasks like image in-painting or style transfer have been studied at length, we propose to investigate the potential of generative models in the context of physical simulations. Providing a dataset of 300k image-pairs and baseline evaluations for three different physical simulation tasks, we propose the following basic research questions: 
- Are generative models able to learn complex physical relations from input-output image pairs? 
- What speedups can be achieved by replacing differential equation based simulations?
- How can the physical correctness of model outputs be enforced?


![teaserfig](figures/TeaserFig.svg "Teaser Figure")



## Download Datasets

The dataset used for evaluation is publicly available and published via Zenodo, ensuring easy access and reproducibility of our research findings [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11448582.svg)](https://doi.org/10.5281/zenodo.11448582).


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




# Results
## Urban Sound Propagation

The table below presents baseline performance metrics for various architectural approaches, encompassing combined mean absolute error (MAE) and weighted mean absolute percentage error (wMAPE), alongside specific line-of-sight (LoS) and non-line-of-sight (NLoS) metrics. 

| Condition   | Architecture | MAE | wMAPE | LoS MAE | NLoS MAE | LoS wMAPE | NLoS wMAPE |
|-------------|--------------|--------------|----------------|---------|----------|-----------|------------|
| Baseline    | UNet         | 2.08         | 19.45          | 2.29    | 1.73     | 12.91     | 37.57      |
| Baseline    | GAN          | **1.52**         | **8.21**           | **1.73**    | **1.19**     | **9.36**      | **6.75**       |
| Baseline    | Diffusion    | 2.57         | 25.21          | 2.42    | 3.26     | 15.57     | 51.08      |
|             |              |                 |                 |                |                |                |                |
| Diffraction | UNet         | **1.65**         | 9.75           | 0.94    | **3.27**     | 4.22      | 22.36      |
| Diffraction | GAN          | 1.66         | **8.03**           | **0.91**    | 3.36     | **3.51**      | **18.06**      |
| Diffraction | Diffusion    | 2.12         | 11.85          | 1.59    | **3.27**     | 8.25      | 20.30      |
|             |              |                 |                 |                |                |                |                |
| Reflection  | UNet         | 3.22         | 31.87          | 2.29    | 5.72     | 12.75     | 80.46      |
| Reflection  | GAN          | **2.88**         | **16.57**          | **2.14**    | **4.79**     | **11.30**     | **30.67**      |
| Reflection  | Diffusion    | 4.14         | 35.20          | 2.74    | 7.93     | 17.85     | 80.38      |
|             |              |                 |                 |                |                |                |                |
| Combined    | UNet         | 1.77         | 20.59          | 1.39    | 2.63     | 10.10      | 45.15      |
| Combined    | GAN          | 1.76         | **19.12**          | 1.37    | 2.67     | **9.80**      | 40.68      |
| Combined    | Diffusion    | **1.57**         | 21.45          | **1.26**    | **2.21**     | 13.07     | **40.38**      |

## Lens Distortion

TODO

## Dynamics of rolling and bouncing movements

TODO

### Contribute to the Leaderboard

We welcome contributions from the research community! If you have conducted research on sound propagation prediction and have results that outperform those listed in our leaderboard, or if you've developed a new architectural approach that shows promise, we invite you to share your findings with us.

Please submit your results, along with a link to your publication, to martin.spitznagel@hs-offenburg.de. Submissions should include detailed performance metrics and a description of the methodology used. Accepted contributions will be updated on the leaderboard.

## License
This dataset is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/)