import warnings
warnings.filterwarnings('ignore')

from . import simulate as otsim
from . import preprocessing as otpp
from . import tools as ottl
from . import plotting as otpl

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['otpp','ottl','otpl']})

