from typing import Union, Optional
from pathlib import Path
import os
import torch

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class Douban(Dataset):
    """
    A pair of online-offline social networks constructed from Douban. Nodes represent
    users and edges represent user interactions on the website. The location of a suer is treated as the node attribute,
    and the contact/friend relationship are treated as the edge attributes.
    There are 1,118 common user across the two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Douban(online)
          - 3,906
          - 8,164
          - 538
          - 2
        * - Douban(offline)
          - 1,118
          - 1,511
          - 538
          - 2

    """
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1KnP4yzQIZ9J36x-or8uViYVc1eyacn95',
                save_filename='douban.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Douban dataset not found or corrupted. You can use download=True to download it')

        super(Douban, self).__init__(root=root, name='douban', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'douban.pt'))
