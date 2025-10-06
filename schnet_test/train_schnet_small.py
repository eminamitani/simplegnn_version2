import torch
import numpy as np
import random

# シードを設定する関数
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 使用するシード値を指定
set_seed(42)



from sklearn.model_selection import train_test_split
import torch
datalist=torch.load('./data/test.pt', weights_only=False)
train_data, test_data = train_test_split(datalist, test_size=0.2)

from torch_geometric.loader import DataLoader
batch_size = 20
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

device=torch.device('cuda')
from simplegnn.schnet import SchNetModel

cutoff=5.0
num_gaussians=60
hidden_dim=100
num_interactions=3
num_filters=100

model = SchNetModel(hidden_dim=hidden_dim, num_gaussians=num_gaussians, 
                    num_filters=num_filters, num_interactions=num_interactions, cutoff=cutoff)
model=model.to(device)


from tensorboardX import SummaryWriter
writer = SummaryWriter()

from torch.optim import AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
loss= MSELoss()

force_weight=torch.tensor(0.99).to(device)
ene_weight=torch.tensor(0.01).to(device)

epochs=50 
for epoch in range(epochs):
    loss_total=0
    loss_e_total=0
    loss_f_total=0

    model.train()
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        energies,forces,sigma = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                
        y = batch.y
        loss_e = loss(energies, y)
        loss_f = loss(forces, batch.forces)
        
        l = loss_e*ene_weight + loss_f*force_weight
        l.backward()
        optimizer.step()
        
        loss_total += l.item()/len(train_dataloader)
        loss_e_total += loss_e.item()/len(train_dataloader)
        loss_f_total += loss_f.item()/len(train_dataloader)

    scheduler.step()
    print('epoch: train', epoch, 'loss_total', loss_total, 'loss_e', loss_e_total, 'loss_f', loss_f_total)
    print('sigma', sigma)
    writer.add_scalar('loss_total', loss_total, epoch)
    writer.add_scalar('loss_e', loss_e_total, epoch)
    writer.add_scalar('loss_f', loss_f_total, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    model.eval()
    loss_total=0
    loss_e_total=0
    loss_f_total=0
    for batch in test_dataloader:
        batch = batch.to(device)
        
        energies,forces,sigma = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        y = batch.y
        loss_e = loss(energies, y)
        loss_f = loss(forces, batch.forces)
        l = loss_e*ene_weight + loss_f*force_weight
        loss_total += l.item()/len(test_dataloader)
        loss_e_total += loss_e.item()/len(test_dataloader)
        loss_f_total += loss_f.item()/len(test_dataloader)
    print('epoch: test', epoch, 'loss_total', loss_total, 'loss_e', loss_e_total, 'loss_f', loss_f_total)
    print('sigma', sigma)
    writer.add_scalar('loss_total_test', loss_total, epoch)
    writer.add_scalar('loss_e_test', loss_e_total, epoch)
    writer.add_scalar('loss_f_test', loss_f_total, epoch)


writer.close()

#save model
torch.save({'model_state_dict': model.state_dict(),
           'setups': model.setups}, 'model_schnet.pth')
torch.save(model, 'model_schnet_full.pth')

#plotting 
results_energy=[]
results_forces=[]
ref_energy=[]
ref_forces=[]
for batch in test_dataloader:
    batch = batch.to(device)
    energies,forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    results_energy.extend(energies.detach().to('cpu').numpy().flatten())
    results_forces.extend(forces.detach().to('cpu').numpy().flatten())
    ref_energy.extend(batch.y.detach().to('cpu').numpy().flatten())
    ref_forces.extend(batch.forces.detach().to('cpu').numpy().flatten())

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(ref_energy,results_energy)
plt.xlabel('reference energy')
plt.ylabel('predicted energy')
plt.title('SchNetModel energy')
plt.savefig('energy_schnet_torch_full_stepLR.png')
plt.close()

plt.figure(figsize=(5,5))
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(ref_forces,results_forces)
plt.xlabel('reference forces')
plt.ylabel('predicted forces')
plt.title('SchNetModel forces')
plt.savefig('forces_schnet_torch_full_stepLR.png')
plt.close()
