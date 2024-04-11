# -*- coding: utf-8 -*-
"""
@author: VR Feb 2021
"""
from setuptools import setup
from setuptools import find_packages
import os

from py4mt.modules.version import versionstr

version, release_date = versionstr()

setup(
    name='Py4MT',
    version=version+"   "+release_date,
    description='Tools for magnetotelluric modeling and inversion',
    author="Volker Rath",
    author_email="vrath@cp.dias.ie",
    license="LGPL",
    url="https://github.com/volkerrath/Py4MT2/",
    include_package_data=True,
    packages=find_packages(),
    package_data={'': ['*.txt', '*.h5', '*.npy', '*.sh', '*.zip', '*.rst']})
