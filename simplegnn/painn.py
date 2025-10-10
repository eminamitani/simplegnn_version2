
'''
radial basis function  
'''

import torch
import torch.nn as nn


def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """
    condition = (edge_dist < cutoff).to(edge_dist.device)  
    return torch.where(
        condition,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype), 
    )


class TypeEmbedding(nn.Module):
    """
    embedding atom type to node scalar representation
    """
    def __init__(self, type_num, type_dim):
        super().__init__()
        self.embedding = nn.Embedding(type_num, type_dim)

    def forward(self, x):
        return self.embedding(x)

class MessageLayer(nn.Module):
    #atomwise message passing
    def __init__(self, natom_basis, n_radial, cutoff):
        super(MessageLayer, self).__init__()
        self.natom_basis=natom_basis
        self.n_radial=n_radial
        self.cutoff=cutoff
        self.interaction_context_network=nn.Sequential(
            nn.Linear(self.natom_basis, natom_basis),
            nn.SiLU(),
            nn.Linear(self.natom_basis, natom_basis*3),
        )

        
        self.filter_network=nn.Sequential(
            nn.Linear(self.n_radial,natom_basis*3),
        )

    def forward(self, q, mu, edge_index, edge_weight):
        #q: scalar representation
        #mu: vector representation

        #message passing
        x = self.interaction_context_network(q)
        distances=torch.norm(edge_weight, dim=-1)
        directions=edge_weight/distances.unsqueeze(-1)
        
        basis_fn=sinc_expansion(distances, self.n_radial, self.cutoff)
        cutoff=cosine_cutoff(distances, self.cutoff).unsqueeze(-1)
        filter_Wij=self.filter_network(basis_fn)*cutoff
        
        idx_i=edge_index[0]
        idx_j=edge_index[1]

        xj=x[idx_j]
        muj=mu[idx_j]
        x=filter_Wij*xj

        #gated 
        dq, dmuR, dmumu=torch.split(x, self.natom_basis, dim=-1)

        #aggregation
        index=idx_i.unsqueeze(1).expand_as(dq)
        
        q_update=torch.zeros_like(q,device=q.device)
        q_update=torch.scatter_add(q_update, 0, index, dq)

        dmuR = dmuR.unsqueeze(1)
        dmumu = dmumu.unsqueeze(1)
        dmu = dmuR * directions[..., None] + dmumu * muj

        index=idx_i.unsqueeze(-1).unsqueeze(-1).expand_as(dmu)
        
        mu_update=torch.zeros_like(mu,device=mu.device)
        mu_update = torch.scatter_add(mu_update, 0, index, dmu)
        
        q=q+q_update
        mu=mu+mu_update

        return q, mu

class UpdateLayer(nn.Module):
    def __init__(self, natom_basis, epsilon):
        """
        updating scaler representation using vector representation
        
        natom_basis: embedding atom type dimension, dimension of scalar representation
        epsilon: small value to avoid zero division
        """
        super(UpdateLayer, self).__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.natom_basis=natom_basis
        self.intraatomic_context_net=nn.Sequential(
            nn.Linear(self.natom_basis*2, self.natom_basis),
            nn.SiLU(),
            nn.Linear(self.natom_basis, self.natom_basis*3),
        )
        self.mu_chammel_mix=nn.Sequential(
            nn.Linear(self.natom_basis, self.natom_basis*2),
        )

    def forward(self, q, mu):
        """
        updating scaler representation using vector representation
        
        q: scalar representation
        mu: vector representation
        """
        mu_mix=self.mu_chammel_mix(mu)
        mu_V,mu_W=torch.split(mu_mix, self.natom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=False) + self.epsilon)
        
        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)
        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.natom_basis, dim=-1)

        dmu_intra = dmu_intra.unsqueeze(1) * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=False)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        
        return q,mu

class Painn(nn.Module):
    def __init__(self, natom_basis, n_radial, cutoff, epsilon, num_interactions):
        super(Painn, self).__init__()
        self.natom_basis=natom_basis
        self.n_radial=n_radial
        self.cutoff=cutoff
        self.epsilon=epsilon
        self.embedding=TypeEmbedding(100, self.natom_basis)
        self.num_interactions=num_interactions

        self.message_layers = nn.ModuleList()
        for _ in range(self.num_interactions):
            block = MessageLayer(self.natom_basis, self.n_radial, self.cutoff)
            self.message_layers.append(block)
        
        self.mixing_layers=nn.ModuleList()
        for _ in range(self.num_interactions):
            block = UpdateLayer(self.natom_basis, self.epsilon)
            self.mixing_layers.append(block)

        self.output = nn.Sequential(
            nn.Linear(self.natom_basis, self.natom_basis // 2),
            nn.SiLU(),
            nn.Linear(self.natom_basis // 2, 1)
        )
    
    def forward(self, Z, edge_index, edge_weight, batch):
        edge_weight.requires_grad_(True)
        node_scalar=self.embedding(Z)
        node_vector=torch.zeros((node_scalar.shape[0],3,node_scalar.shape[1]),device=node_scalar.device)

        for message, mixing in zip(self.message_layers, self.mixing_layers):
            node_scalar, node_vector = message(node_scalar, node_vector, edge_index, edge_weight)
            node_scalar, node_vector = mixing(node_scalar, node_vector)
        
        energy = self.output(node_scalar)

        # derivative with respect to edge_weight=rj-ri
        diff_E = torch.autograd.grad(energy.sum(), edge_weight, create_graph=True)[0]
        #p: pair, k,l: x,y,z
        # virial tensor for ij pair = r_ij (dE/d r_ij)
        sigma_ij = torch.einsum('pk,pl->pkl', edge_weight, diff_E)

        force_i = torch.zeros((len(node_scalar), 3), device=energy.device)
        force_j = torch.zeros((len(node_scalar), 3), device=energy.device)

        index_i=edge_index[0].unsqueeze(1) if edge_index[0].ndim == 1 else edge_index[0]
        index_j=edge_index[1].unsqueeze(1) if edge_index[1].ndim == 1 else edge_index[1]
        force_i=torch.scatter_add(force_i, 0, index_i.expand_as(diff_E), diff_E)
        force_j=torch.scatter_add(force_j, 0, index_j.expand_as(diff_E), -diff_E)

        forces=force_i+force_j

        if batch is not None:
            batch_max = batch.max().item()
            total_energy = torch.zeros(batch_max + 1, device=energy.device)
            total_energy = total_energy.index_add_(0, batch, energy.squeeze())
            sigma = torch.zeros((batch_max + 1, 3, 3), device=edge_weight.device)
            batch_edge = batch[edge_index[0]]  # edge_indexのi側の原子のバッチ情報
            sigma = sigma.index_add_(0, batch_edge, sigma_ij)
        else:
            total_energy = energy.sum()
            sigma = sigma_ij.sum(dim=0)

        if sigma.dim() == 3:
            sigma = 0.5 * (sigma + sigma.transpose(1, 2))
        else:
            sigma = 0.5 * (sigma + sigma.T)
            
        return total_energy,forces, sigma
