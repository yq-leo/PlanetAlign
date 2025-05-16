import copy

from .isorank import IsoRank
from .final import FINAL
from .ione import IONE
from .regal import REGAL
from .crossmna import CrossMNA
from .nettrans import NetTrans
from .bright import BRIGHT
from .nextalign import NeXtAlign
from .parrot import PARROT
from .slotalign import SLOTAlign
from .wlalign import WLAlign
from .walign import WAlign
from .hot import HOT
from .joena import JOENA
from .dualmatch import DualMatch
from .meaformer import MEAformer

__all__ = [
    'IsoRank',
    'IONE',
    'FINAL',
    'REGAL',
    'CrossMNA',
    'NetTrans',
    'BRIGHT',
    'NeXtAlign',
    'PARROT',
    'SLOTAlign',
    'WLAlign',
    'WAlign',
    'HOT',
    'JOENA',
    'DualMatch',
    'MEAformer'
]

classes = copy.copy(__all__)
