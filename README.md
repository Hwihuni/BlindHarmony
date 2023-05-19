# BlindHarmony
Official implementation of BlindHarmony: Blind harmonization for MR image
> H. Jeong, H. Byun, D. Kang, and J. Lee, _BlindHarmony: “Blind” Harmonization for MR Images via Flow model_, Arxiv 2023,
> [[arXiv]]()

## Dependencies

Neural spline flow is used for flow model training. See https://github.com/bayesiains/nsf.git

Use `environment.yml` for required packages, or create a Conda environment with all dependencies:
```bash
conda env create -f environment.yml
```

## Dataset

The whole data of OASIS-3 can be accessed by https://www.oasis-brains.org/.

The whole data of BRATS can be accessed by https://www.synapse.org/#!Synapse:syn27046444/wiki/616571

## Pretrained models

The checkpoints can be downloaded from google drive link in https://drive.google.com/drive/folders/1AuCYGiNOZ8hWrqiV_npsjmcodNVfRb6z?usp=share_link

## Usage

`DATAROOT` environment variable needs to be set before running experiments.

### Flow model training

Use `train_flow.py`.

### Harmonization using simulation data

Use `BlindHarmony_simulated_data.py`.


### Harmonization using simulation data

Use `BlindHarmony_real_data.py`.
