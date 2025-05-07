#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: VR Feb 2021
"""

import sys
import os
import inspect

sys.path.append(os.path.dirname(inspect.getfile(inspect.currentframe())))

from . import modules
from . import scripts

# define custEM version
with open(os.path.dirname(inspect.getfile(inspect.currentframe())) + '/version.txt', 'r') as v_file:
    version = v_file.readline()[:7]
    release_date = v_file.readline()[:10]

__version__ = version
# __release__ = release_date
