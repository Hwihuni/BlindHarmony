# BlindHarmony
Official implementation of BlindHarmony: Blind harmonization for MR image
> H. Jeong, H. Byun, D. Kang, and J. Lee, _BlindHarmony: “Blind” Harmonization for MR Images via Flow model_, Arxiv 2023,
> [[arXiv]]()

##Dependencies

See `environment.yml` for required Conda/pip packages, or use this to create a Conda environment with all dependencies:
```bash
conda env create -f environment.yml
```

## Dataset

The whole data of OASIS-3 can be accessed by https://www.oasis-brains.org/.

The whole data of BRATS can be accessed by https://www.synapse.org/#!Synapse:syn27046444/wiki/616571

## Pretrained models

The checkpoints can be download form google drive link in /python/r2_r2star_mapping/checkpoints/download.txt

## Requirements
```
MATLAB: R2021b
PYTHON: use env.yml file in python/r2_r2star_mapping
FSL built with wsl
```

## x-separation reconstruction
* change 'path' in Biobank_x_sep_step0_Dicomprocessing.m, Biobank_x_sep_step1_GREprocessing.m, Biobank_x_sep_step2_DLpreprocessing.m, and Biobank_x_sep_step3_reconstrction.m
* run Biobank_x_sep_step0_Dicomprocessing.m, Biobank_x_sep_step1_GREprocessing.m and Biobank_x_sep_step2_DLpreprocessing.m
* move Data_to_gpu_t2map.mat and Data_to_gpu_t2starmap.mat to python\r2_r2star_mapping\data
* run predict_r2.py and predict_r2star.py
* move inference file to 'inf_from_gpu' which is made in your directory
* run Biobank_x_sep_step3_reconstrction.m
