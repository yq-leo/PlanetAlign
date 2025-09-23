from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class Italy(Dataset):
    """A pair of power grid networks from two regions in Italy. Nodes represent power stations and edges represent the
    existence of power transfer lines. Node attributes are derived from node labels. There are in total 377 common nodes
    across two networks inferred from the ground-truth cross-layer dependencies.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Italy1
          - 1050
          - 3813
          - 6
          - 0
        * - Italy2
          - 1200
          - 4389
          - 6
          - 0
    """

    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1YoyembxnyNgA_D-2FgeQShRMWC5WKQ_U',
                save_filename='Italy.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Italy dataset not found or corrupted. You can use download=True to download it')

        super(Italy, self).__init__(root=root, name='Italy', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'Italy.pt'))
