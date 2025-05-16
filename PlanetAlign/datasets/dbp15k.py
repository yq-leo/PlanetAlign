from typing import Union, Optional
from pathlib import Path
import os
import torch

from PlanetAlign.data import Dataset

from .utils import download_file_from_google_drive


class DBP15K_FR_EN(Dataset):
    """A pair of French to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (French to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - FR
          - 19,661
          - 105,997
          - 300
          - 0
        * - EN
          - 19,993
          - 115,722
          - 300
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
                remote_file_id='1xpZqBYtuLvAFrLESzXAvB7P-tPXmGWTi',
                save_filename='dbp15k_fr-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_FR-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_FR_EN, self).__init__(root=root, name='dbp15k_fr-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_fr-en.pt'))


class DBP15K_JA_EN(Dataset):
    """A pair of Japanese to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (Japanese to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - JA
          - 19,814
          - 77,214
          - 300
          - 0
        * - EN
          - 19,780
          - 93,484
          - 300
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
                remote_file_id='1RSf2tyx50zUpHnBGDKzg07CNkQqBekgL',
                save_filename='dbp15k_ja-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_JA-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_JA_EN, self).__init__(root=root, name='dbp15k_ja-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_ja-en.pt'))


class DBP15K_ZH_EN(Dataset):
    """A pair of Chinese to English version of multi-lingual DBpedia networks. The dataset is proposed by the
    `"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding" <https://arxiv.org/abs/1708.05045>`_ paper,
    and the node attributes are given by pre-trained and aligned monolingual word embeddings from the
    `"Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.
    There are 15,000 pairs of aligned entities in DBP15K (Chinese to English).

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Graph
          - #nodes
          - #edges
          - #node attrs
          - #edge attrs
        * - ZH
          - 19,388
          - 70,414
          - 300
          - 0
        * - EN
          - 19,572
          - 95,142
          - 300
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
                remote_file_id='18f5zsUBWYsSw5ACWQGFdsHQAkafuHqs5',
                save_filename='dbp15k_zh-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_ZH-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_ZH_EN, self).__init__(root=root, name='dbp15k_zh-en', train_ratio=train_ratio, dtype=dtype, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_zh-en.pt'))
