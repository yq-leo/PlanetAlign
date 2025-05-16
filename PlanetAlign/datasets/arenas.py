from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class Arenas(Dataset):
    """A pair of networks synthesized from the email communication network Arenas at the University Rovira i Virgili.
    Nodes are users and each edge represents that at least one email was sent. The two networks are noisy permutations
    of each other. The dataset is proposed by the paper: "`KONECT – The Koblenz Network Collection.
    <https://dl.acm.org/doi/pdf/10.1145/2487788.2488173>`_" in WWW 2013. There are in total 1,135 common nodes across two
    networks.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Arenas1
          - 1,135
          - 10,902
          - 50
          - 0
        * - Arenas2
          - 1,135
          - 10,800
          - 50
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
                remote_file_id='10ajbVon_Vbp4HRPEAQGBAIYdza23qvpk',
                save_filename='arenas.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Arenas email dataset not found or corrupted. You can use download=True to download it')

        super(Arenas, self).__init__(root=root, name='arenas', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'arenas.pt'))
