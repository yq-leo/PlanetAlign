from typing import Union, Optional, List, Tuple
import warnings
import torch

from PlanetAlign.utils import get_anchor_pairs
from PlanetAlign.data import Dataset, BaseData
from PlanetAlign.logger import TrainingLogger
from PlanetAlign.metrics import hits_ks_scores, mrr_score


class BaseModel:
    def __init__(self, dtype: torch.dtype = torch.float32):
        assert dtype in [torch.float32, torch.float64], 'Invalid floating point dtype'
        self.dtype = dtype
        self.device = 'cpu'
        self.S = None

    def train(self, 
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              *args, **kwargs):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : list[int] or tuple[int, ...]
            The indices of the graphs in the dataset to be aligned.
        """
        raise NotImplementedError

    def test(self,
             dataset: Dataset,
             gids: Union[List[int], Tuple[int, ...]],
             metrics: tuple[str] | list[str] = None):
        """
        Parameters
        ----------

        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : list[int] or tuple[int, ...]
            The indices of the graphs in the dataset to be aligned.
        metrics : tuple[str] or list[str], optional
            The metrics to be computed after alignment. Default is None, which computes Hits@K (K=1, 10, 30, 50) and MRR metrics.
        """

        assert self.S is not None, 'Model is not trained yet, call train() method first'

        results = {}

        gid1, gid2 = gids
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        if metrics is None:
            hits = hits_ks_scores(self.S, test_pairs, mode='mean')
            results.update({f'Hits@{k}': hits[k] for k in hits.keys()})
            results['MRR'] = mrr_score(self.S, test_pairs, mode='mean')
        else:
            ks = [int(hit_metric.split('@')[-1]) for hit_metric in metrics if hit_metric.startswith('Hits@')]
            hits = hits_ks_scores(self.S, test_pairs, ks=ks, mode='mean')
            results.update({f'Hits@{k}': hits[k] for k in ks if k in hits})
            if 'MRR' in metrics:
                results['MRR'] = mrr_score(self.S, test_pairs, mode='mean')

        return results

    def to(self, device: Union[str, torch.device]) -> 'BaseModel':
        try:
            self.device = torch.device(device)
        except Exception:
            raise ValueError(f"Invalid device '{device}' specified.")
        return self

    def init_training_logger(self,
                             dataset: Dataset,
                             use_attr: bool,
                             log_dir: str = 'logs',
                             log_name: Optional[str] = None,
                             save_log: bool = True,
                             **kwargs) -> TrainingLogger:
        attr_dir = 'attr' if use_attr else 'plain'
        log_dir = f"{log_dir}/{self.__class__.__name__}/{attr_dir}/{dataset.__class__.__name__}/{dataset.train_ratio}/seed_{dataset.seed}"
        return TrainingLogger(log_dir=log_dir, log_name=log_name, save=save_log, **kwargs)

    def check_inputs(self, dataset: Dataset,
                     gids: Union[list[int], tuple[int, ...]],
                     plain_method: bool,
                     use_attr: bool,
                     pairwise: bool,
                     supervised: bool):
        if plain_method:
            assert not use_attr, f'{self.__class__.__name__} does not support attributed network alignment, set "use_attr=False"'

        if pairwise:
            self.check_pairwise_input_graphs(dataset, gids)
        else:
            self.check_input_graphs(dataset, gids)

        self.check_usage_of_attributes(dataset, gids, use_attr)
        self.check_supervision(dataset, supervised)

    @staticmethod
    def check_input_graphs(dataset: Dataset, gids: Union[list[int], tuple[int, ...]]):
        assert isinstance(dataset, Dataset) or isinstance(dataset, BaseData), 'Input dataset must be a PlanetAlign Dataset or BaseData object'
        assert all([0 <= gid < len(dataset.pyg_graphs) for gid in gids]), 'Invalid graph IDs'
        assert len(set(gids)) == len(gids), 'Graph IDs must be unique'

    def check_pairwise_input_graphs(self, dataset: Dataset, gids: Union[list[int], tuple[int, ...]]):
        self.check_input_graphs(dataset, gids)
        assert len(gids) == 2, 'Exactly two graphs are required for pairwise alignment'

    def check_usage_of_attributes(self, dataset: Dataset, gids: Union[list[int], tuple[int, ...]], use_attr: bool):
        attributes_are_available = all([dataset.pyg_graphs[gid].x is not None for gid in gids])
        if not use_attr and attributes_are_available:
            warnings.warn(f'Attributes are available in {dataset.__class__.__name__} but not used by {self.__class__.__name__}')
        elif use_attr and not attributes_are_available:
            raise ValueError(f'Attributes are not available for "{dataset.__class__.__name__}" dataset, set use_attr=False')

    @staticmethod
    def check_supervision(dataset: Dataset, supervised: bool):
        if supervised and len(dataset.train_data) == 0:
            raise ValueError(f'Supervision is required but no training data is available in "{dataset.__class__.__name__}" dataset')
