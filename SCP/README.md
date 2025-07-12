# Self-generated Cross-modal Prompt Tuning (SCP)

We are pleased to announce that our paper, **Self-generated Cross-modal Prompt Tuning**, has been accepted at **ECML-PKDD 2025**.

This repository is built upon **MaPLe**. Most of the infrastructure remains unchanged from MaPLe.  
The following files have been modified for SCP:
- `Trainers/maple.py`
- `Trainers/clip_text.py`
- `Dassl.pytorch`
- `Trainers/clip.py`

## Installation

For installation and package requirements, please refer to [INSTALL.md](docs/INSTALL.md).
Please use the provided Dassl during the installation.

## Data Preparation

Please follow the instructions in [DATASETS.md](docs/DATASETS.md) to prepare the datasets.

## Training and Evaluation

For detailed guidance on training, evaluation, and reproducing our results with the pre-trained models, please refer to [RUN.md](docs/RUN.md).

Additionally, we provide a convenient script, `sleep_all.bash`, to automate the training and evaluation process, allowing researchers to run experiments overnight.

## Contact

If you have any questions, please feel free to open an issue on this repository or contact us at:  
**guiming.cao@student.uts.edu.au**

## Acknowledgements

Our code builds upon [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning).  
We thank the authors for making their code publicly available.  
If you use our model or code, please consider citing their work as well.