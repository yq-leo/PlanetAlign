from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class PhoneEmail(Dataset):
    """
    A pair of communication networks among people via phone or email.
    Nodes represent people and an edge exists between two people if they communicate via
    phone or email at least once. Phone network includes 1,000 nodes and 41,191 edges.
    Email network includes 1,003 nodes and 4,627 edges. Both networks are plain networks.
    There are 1,000 common people across two networks.

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
          - 1,000
          - 41,191
          - 0
          - 0
        * - Email
          - 1,003
          - 4,628
          - 0
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
                remote_file_id='1RTovO4olFFiXhXbDeO_X2NSDB8wkq742',
                save_filename='phone-email.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Phone-Email dataset not found or corrupted. You can use download=True to download it to the specified "root" directory.')

        super(PhoneEmail, self).__init__(root=root, name='phone-email', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'phone-email.pt'))
