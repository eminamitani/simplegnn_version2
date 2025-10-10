from ase.calculators.calculator import Calculator, all_changes
import torch

from simplegnn.preprocess import RadiusInteractionGraph
from torch_geometric.data import Data


def converter(atoms, cutoff):
    x = torch.tensor(atoms.numbers, dtype=torch.long)

    # 近傍情報の取得
    edge_index, edge_weight = RadiusInteractionGraph(atoms, cutoff)

    # ノードの3次元座標
    pos = torch.tensor(atoms.positions, dtype=torch.float32)

    # PyG Dataオブジェクトを構築
    data = Data(x=x, edge_index=edge_index,edge_weight=edge_weight, pos=pos)

    return data

class SchNetCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']  # stress も返すなら追加

    def __init__(self, model, cutoff, device='cpu', return_stress=False):
        Calculator.__init__(self)
        self.model = model.to(device).eval()  # 推論モード
        self.cutoff = cutoff
        self.device = device

    def calculate(self, atoms=None, properties=('energy', 'forces', 'stress'),
                  system_changes=all_changes):
        
        if self.calculation_required(atoms, properties):
            self.results = {}
             # ASE AtomsオブジェクトをPyGのData形式に変換
            data = converter(atoms, self.cutoff).to(self.device)
            total_energy, forces, sigma = self.model(data.x, data.edge_index, data.edge_weight, batch=None)
            self.results['energy'] = total_energy.to('cpu').item()
            self.results['forces'] = forces.to('cpu').detach().numpy()

            # stress --> virial/volume : eV/A^3 
            volume=atoms.get_volume()
            stress_eV_per_A3 = -sigma.to('cpu').detach().numpy() / volume
            self.results['stress'] = stress_eV_per_A3
