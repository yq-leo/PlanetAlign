from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class FlickrMySpace(Dataset):
    """A pair of social networks from Flickr and MySpace. Nodes in both networks represent users,
    and edges represent friend / following relationships. The gender of a user is treated as the node attributes
    (male, female, unknown), and the level of people a user is connected to is treated as the edge attributes
    (e.g., leader with leader). There are 267 common users across two networks.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - Flickr
          - 6,714
          - 7,333
          - 3
          - 3
        * - MySpace
          - 10,733
          - 11,081
          - 3
          - 3
    """
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1B0Mit89nVk7ZnymTfGYHFZRRckV55neO',
                save_filename='flickr-myspace.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Flickr-MySpace dataset not found or corrupted. You can use download=True to download it')

        super(FlickrMySpace, self).__init__(root=root, name='flickr-myspace', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'flickr-myspace.pt'))
