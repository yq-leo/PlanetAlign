from typing import Union, Optional
import warnings
import torch

from PlanetAlign.data import Dataset, BaseData
from PlanetAlign.logger import TrainingLogger


class BaseModel:
    # TODO: implement warnings for plain and attributed alignment methods
    # TODO: implement warnings for supervised and unsupervised alignment methods

    def __init__(self, dtype: torch.dtype = torch.float32):
        assert dtype in [torch.float32, torch.float64], 'Invalid floating point dtype'
        self.dtype = dtype
        self.device = 'cpu'

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

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
