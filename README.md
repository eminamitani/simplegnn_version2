# Simple GNN models for machine learning interatomic potential

## install 
- required modules: `torch`

### sample for creating virtual environment

- conda 
```
conda create -n gnn_mlp python=3.11
#for old Linux system, gcc update is required
conda install gcc
#install cuda-toolkit makes glibc in conda also up to date
conda install nvidia::cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pyg -c pyg
conda install scikit-learn tensorboardX matscipy
```

### example for training

#### data preparation
In `data` directory, there are sample data for training and testing (DFT calculation results of a-SiO2). 
By running the following script, ASE dataset convert to PyG Data.
```
python data_convert.py
```

#### training
- SchNet model
```
python train_schnet.py
```

- PaiNN model
```
python train_painn.py
```
