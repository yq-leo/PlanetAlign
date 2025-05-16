from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class ACM_DBLP(Dataset):
    """
    A pair of undirected co-authorship networks, ACM and DBLP. Nodes represent authors and edges an edge exists between
    two authors if they co-author at least one paper. Node attributes are available in both networks.
    There are 6,325 common authors across two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Phone
          - 9,872
          - 39,561
          - 17
          - 0
        * - Email
          - 9,916
          - 44,808
          - 17
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
                remote_file_id='1yr6Wi1cyHDl4V7vi5kmdhNIToJbtso7g',
                save_filename='ACM-DBLP.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('ACM-DBLP dataset not found or corrupted. You can use download=True to download it')

        super(ACM_DBLP, self).__init__(root=root, name='ACM-DBLP', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'ACM-DBLP.pt'))
