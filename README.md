# Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection via Graph Score Propagation
## Installation
Our experimental environment is follow [ULIP2](https://github.com/salesforce/ULIP) to establish, the code is tested with python = 3.8, CUDA==11.3 and pytorch==1.10.1.
You can install GSP following (noted that you need to download the requirements.txt from [ULIP2](https://github.com/salesforce/ULIP) and put it into the folder):
```bash
conda create -n GSP
conda activate GSP
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Pre-trained model
You need to download the pre-trained model from [ULIP2](https://github.com/salesforce/ULIP). In our work, we use the pre-trained pointbert as 3D encoder. So you need download the weight as follow:
```bash
./Dwonload (the path is decision by you)
 --pretrained_models_ckpt_zero-sho_classification_pointbert_ULIP-2.pt
```
## Running
Test pre-trained model on ScanObjectNN:
```bash
## test on SR1
python main.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr path/to/your/pre-trained/model --dataset_name ScanObjectNN15 --dataset_split SR1

## test on SR2
python main.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr path/to/your/pre-trained/model --dataset_name ScanObjectNN15 --dataset_split SR2

## test on SR3
python main.py --model ULIP_PointBERT --evaluate_3d --npoints 2048 --test_ckpt_addr path/to/your/pre-trained/model --dataset_name ScanObjectNN15 --dataset_split SR3
```

# Citation


# Acknowledge
We sincerely appreciate these highly valuable repositories [ULIP2](https://github.com/salesforce/ULIP) and [3DOS](https://github.com/antoalli/3D_OS)









