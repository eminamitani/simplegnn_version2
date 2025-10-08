from simplegnn.preprocess import *
from ase.io import iread
import torch

cutoff=5.0
atoms=iread('dataset_with_stress_energy_forces.extxyz',format='extxyz')

samples=[]
for atom in atoms:
    samples.append(atom)

datalist=ConvertAtomsListToDataList(samples, cutoff, include_virial=True)
torch.save(datalist, 'test.pt')