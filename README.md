# Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation
## Installation
Our experimental environment is follow [ULIP](https://github.com/salesforce/ULIP) to establish, the code is tested with python = 3.8, CUDA==11.3 and pytorch==1.10.1.
You can install GSP following (noted that you need to download the requirements.txt from [ULIP](https://github.com/salesforce/ULIP) and put it into the folder):
```bash
conda create -n GSP
conda activate GSP
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt

## Pretrained model
