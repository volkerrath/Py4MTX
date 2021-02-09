# -*- coding: utf-8 -*-

"""
@author: VR Feb 2021
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from . import modules
from . import scripts

# define custEM version
with open(os.path.dirname(__file__) + '/version.txt', 'r') as v_file:
    version = v_file.readline()[:7]
    release_date = v_file.readline()[:10]

__version__ = version
# __release__ = release_date
