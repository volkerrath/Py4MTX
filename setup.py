# -*- coding: utf-8 -*-
"""
@author: VR Feb 2021
"""
from setuptools import setup
from setuptools import find_packages
import os

with open('custEM/VERSION.rst', 'r') as v_file:
    custEM_version = v_file.readline()[:7]
    release_date = v_file.readline()[:10]

readme = open('README.rst').read()
v_str = str(custEM_version + ', ' + release_date)

setup(
    name='custEM',
    version=v_str,
    description='customizable controlled-source electromagnetic modeling',
    author="Rochlitz, Raphael",
    author_email="raphael.rochlitz@leibniz-liag.de",
    license="LGPL",
    url="https://custem.readthedocs.io/",
    include_package_data=True,
    packages=find_packages(),
    package_data={'': ['*.txt', '*.h5', '*.npy', '*.sh', '*.zip', '*.rst']})
