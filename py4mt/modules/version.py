# -*- coding: utf-8 -*-
'''
@author: VR Feb 2021
'''

import sys
import os
from datetime import datetime
import inspect

sys.path.append(os.path.dirname(inspect.getfile(inspect.currentframe())))


def versionstrg():
    '''
    Set version string and date.
    '''
    now = datetime.now()
    version = '- Py4MTX 0.99.99 -'
    release_date =now.strftime('%m/%d/%Y, %H:%M:%S')

    return version, release_date
