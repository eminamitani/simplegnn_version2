import torch 
import numpy as np
from matscipy.neighbours import neighbour_list

def RadiusInteractionGraph(atoms, cutoff):
    """
    ASE Atomsオブジェクトから近傍情報を計算し、
    torch_geometric用のデータ形式に変換する関数。

    Args:
        atoms: ASEのAtomsオブジェクト
        cutoff: カットオフ距離

    Returns:
        edge_index: Tensor, サイズは (2, num_edges)
            近傍ペアのインデックス
        edge_weight: Tensor, サイズは (num_edges, 3)
            原子間距離ベクトル (方向情報)
    """
    # 近傍ペアと距離情報を取得
    i, j, D = neighbour_list('ijD', atoms, cutoff=cutoff)

    i = np.array(i, dtype=np.int64)
    j = np.array(j, dtype=np.int64)
    D = np.array(D, dtype=np.float32)

    # 原子間の方向ベクトル
    edge_weight = torch.tensor(D, dtype=torch.float32)  # 方向ベクトル

    # エッジインデックスをテンソルに変換
    edge_index = torch.tensor(np.stack([i, j]), dtype=torch.long)

    return edge_index, edge_weight

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
    y= torch.tensor(atoms.get_potential_energy(),dtype=torch.float32)
    forces = torch.tensor(atoms.get_forces(),dtype=torch.float32)

    # 近傍情報の取得
    edge_index, edge_weight = RadiusInteractionGraph(atoms, cutoff)

    # ノードの3次元座標
    pos = torch.tensor(atoms.positions, dtype=torch.float32)

    # PyG Dataオブジェクトを構築
    data = Data(x=x, y=y, forces=forces, edge_index=edge_index,edge_weight=edge_weight, pos=pos)

    return data

# ASE Atomsのリストを変換
def ConvertAtomsListToDataList(atoms_list, cutoff, include_virial=False):
    """
    ASEのAtomsオブジェクトのリストをtorch_geometricのData形式のリストに変換
    Args:
        atoms_list: ASEのAtomsオブジェクトのリスト
        cutoff: カットオフ距離
        include_virial: virial応力を含めるかどうか
    Returns:
        data_list: PyGのDataオブジェクトのリスト
    virialとしてdatasetに入れられているのは、VASPのFORCE on cellブロックのTotal部分。
    VASPでのvirialはeV単位、セル体積では割られていない。さらに、通常の定義とは符号が逆
    並び順は
      FORCE on cell =-STRESS in cart. coord.  units (eV):
      Direction    XX          YY          ZZ          XY          YZ          ZX
    """
    data_list = []
    for atoms in atoms_list:
        edge_index, edge_weight = RadiusInteractionGraph(atoms, cutoff)
        if include_virial==False:
            data = Data(
                x=torch.tensor(atoms.numbers, dtype=torch.long),
                edge_index=edge_index,
                edge_weight=edge_weight,  # Enable gradient tracking
                pos=torch.tensor(atoms.positions, dtype=torch.float32),
                forces=torch.tensor(atoms.get_forces(), dtype=torch.float32),
                y=torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
            )
        else:
            data = Data(
                x=torch.tensor(atoms.numbers, dtype=torch.long),
                edge_index=edge_index,
                edge_weight=edge_weight,  # Enable gradient tracking
                pos=torch.tensor(atoms.positions, dtype=torch.float32),
                forces=torch.tensor(atoms.get_forces(), dtype=torch.float32),
                y=torch.tensor(atoms.get_potential_energy(), dtype=torch.float32),
                stress=torch.tensor(-atoms.info['virial_eV_tensor'], dtype=torch.float32)
            )
        data_list.append(data)
    return data_list
