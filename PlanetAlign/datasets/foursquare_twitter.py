from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class FoursquareTwitter(Dataset):
    """
    A pair of online social networks, Foursquare and Twitter. Nodes represent users and an edge exists between
    two users if they have follower/followee relationships. Both networks are plain networks.
    There are 1,609 common users across two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Foursquare
          - 5,313
          - 54,233
          - 0
          - 0
        * - Twitter
          - 5,120
          - 130,575
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
                remote_file_id='1dIfQH5chgMFmestU59qsD5EQgZVcKSoV',
                save_filename='foursquare-twitter.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Foursquare-Twitter dataset not found or corrupted. You can use download=True to download it')

        super(FoursquareTwitter, self).__init__(root=root, name='foursquare-twitter', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'foursquare-twitter.pt'))
