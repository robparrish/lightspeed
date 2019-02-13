# => LS Extension Import <= #

from .env import *
from .tensor import *
from .solver import *
from .core import *
from .intbox import *
from .casbox import *
from .gridbox import *
from .cubic import *
from .becke import *
from .dftbox import *
from .sad import *
from .lr import *
from .util import *
from .title import *

# => LS Environment Variables <= #

import os
rootdir = os.environ.get('LIGHTSPEED_ROOT')
if rootdir is None:
    raise ValueError("Must set LIGHTSPEED_ROOT to base LS install dir")

# Provide the basis set library path
Basis.basisdir = rootdir + '/data/basis' # A bit of a hack

import atexit
@atexit.register
def _call_exit_hooks():
    exit_hooks()
    
