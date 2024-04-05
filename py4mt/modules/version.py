# -*- coding: utf-8 -*-
"""
@author: VR Feb 2021
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(__file__))


def versionstrg():
    """
    Set version string and date.
    """
    now = datetime.now()
    version = "- Py4MT2 0.99.99 -"
    release_date =now.strftime("%m/%d/%Y, %H:%M:%S")

    return version, release_date
