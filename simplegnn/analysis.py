from __future__ import annotations

from typing import Callable, Iterable, Literal

import ase
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import torch
from cycler import cycler
from matplotlib.ticker import MaxNLocator
from torch import Tensor

from simplegnn.preprocess import ConvertAtomsListToDataList
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from ase.neighborlist import neighbor_list

def parity_plot(
    model,
    structures,
    batch_size: int = 5,
    device: Literal["cpu", "cuda"] | torch.device = "cpu",
    **scatter_kwargs,
):

    graphs = ConvertAtomsListToDataList(structures, cutoff=model.cutoff)
    loader = DataLoader(graphs, batch_size=batch_size,shuffle=False)
    model.eval()
    model.to(device)
    results_energy=[]
    results_forces=[]
    ref_energy=[]
    ref_forces=[]


    for batch in loader:
        batch = batch.to(device)
        energies,forces, sigma = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        results_energy.extend(energies.detach().to('cpu').numpy().flatten())
        results_forces.extend(forces.detach().to('cpu').numpy().flatten())
        ref_energy.extend(batch.y.detach().to('cpu').numpy().flatten())
        ref_forces.extend(batch.forces.detach().to('cpu').numpy().flatten())

    
    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_energy,results_energy,c='blue',s=2)
    plt.xlabel('reference energy (eV)')
    plt.ylabel('predicted energy (eV)')
    plt.title('PaiNN energy')
    plt.savefig('energy_painn_parity_plot.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_forces,results_forces,c='blue',s=2)
    plt.xlabel('reference forces (eV/A)')
    plt.ylabel('predicted forces (eV/A)')
    plt.title('PaiNN forces')
    plt.savefig('forces_painn_parity_plot.png')
    plt.close()

from torch_geometric.data import Data
def AtomsToPyGData(atoms, cutoff):
    """
    ASEのAtomsオブジェクトをtorch_geometricのData形式に変換

    Args:
        atoms: ASEのAtomsオブジェクト
        cutoff: カットオフ距離

    Returns:
        data: PyGのDataオブジェクト
    """
    # 原子特徴量 (例: 原子番号)
    x = torch.tensor(atoms.numbers, dtype=torch.long)
    
    # 近傍ペアと距離情報を取得
    i, j, D = neighbor_list('ijD', atoms, cutoff=cutoff)

    i = np.array(i, dtype=np.int64)
    j = np.array(j, dtype=np.int64)
    D = np.array(D, dtype=np.float32)

    # 原子間の方向ベクトル
    edge_weight = torch.tensor(D, dtype=torch.float32)  # 方向ベクトル

    # エッジインデックスをテンソルに変換
    edge_index = torch.tensor(np.stack([i, j]), dtype=torch.long)

    # ノードの3次元座標
    pos = torch.tensor(atoms.positions, dtype=torch.float32)

    # PyG Dataオブジェクトを構築
    data = Data(x=x, y=None, forces=None, edge_index=edge_index,edge_weight=edge_weight, pos=pos)

    return data

def dimer_curve(
    model,
    system: str,
    units: str | None = None,
    device: Literal["cpu", "cuda"] | torch.device = "cpu",
    rmin: float = 0.9,
    rmax: float | None = None,
    **plot_kwargs,
):
    r"""
    A nicely formatted dimer curve plot for the given :code:`system`.

    Parameters
    ----------
    model
        The model for generating predictions.
    system
        The dimer system. Should be one of: a single element, e.g. :code:`"Cu"`,
        or a pair of elements, e.g. :code:`"CuO"`.
    units
        The units of the energy, for labelling the axes. If not provided, no
        units are used.
    set_to_zero
        Whether to set the energy of the dimer at :code:`rmax` to be zero.
    rmin
        The minimum seperation to consider.
    rmax
        The maximum seperation to consider.
    ax
        The axes to plot on. If not provided, the current axes are used.
    plot_kwargs
        Keyword arguments to pass to :code:`plt.plot`.

    Examples
    --------

    .. code-block:: python

        from graph_pes.utils.analysis import dimer_curve
        from graph_pes.models import LennardJones

        dimer_curve(LennardJones(sigma=1.3, epsilon=0.5), system="OH", units="eV")

    .. image:: dimer-curve.svg
        :align: center
    """  # noqa: E501

    trial_atoms = ase.Atoms(system)
    if len(trial_atoms) != 2:
        system = system + "2"

    if rmax is None:
        rmax = model.cutoff - 0.5
    rs = np.linspace(rmin, rmax, 200)
    dimers = [ase.Atoms(system, positions=[[0, 0, 0], [r, 0, 0]],pbc=False) for r in rs]
    graphs = [AtomsToPyGData(dimer, cutoff=model.cutoff) for dimer in dimers]

    loader=DataLoader(graphs, batch_size=1,shuffle=False)
    
    energies_predict=[]
    for batch in loader:
        batch = batch.to(device)
        energies,forces, sigma = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        energies_predict.extend(energies.detach().to('cpu').numpy().flatten())

    energies_predict = np.array(energies_predict)
    plt.figure(figsize=(5,5))
    plt.plot(rs, energies_predict)
    plt.xlabel("Interatomic distance / Å")
    plt.ylabel(f"Energy / {'eV' if units is None else units}")
    plt.title(f"Dimer curve for {system}")
    plt.savefig(f'dimer_curve_{system}.png')