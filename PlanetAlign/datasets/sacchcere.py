from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset
from .utils import download_file_from_google_drive


class SacchCere(Dataset):
    """A pair of direct interaction layer and association layer from the SacchCere multiplex GPI network. The
    SacchCere network consider different kinds of protein and genetic interactions for Saccharomyces Cerevisiae in
    `BioGRID <https://thebiogrid.org>`_, a public database that archives and disseminates genetic and protein interaction data from humans and model
    organisms. There are in total 1,337 common nodes across two layers of networks. The original multi-layer networks can be found `here <https://manliodedomenico.com/data.php>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - SacchCere-direct
          - 5,042
          - 54,045
          - 0
          - 0
        * - SacchCere-association
          - 1,401
          - 3,918
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
                remote_file_id='1xD6ilqX6ICfi0jSkCTd1Jf17sG3xx_yN',
                save_filename='SacchCere.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('SacchCere dataset not found or corrupted. You can use download=True to download it')

        super(SacchCere, self).__init__(root=root, name='SacchCere', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'SacchCere.pt'))
