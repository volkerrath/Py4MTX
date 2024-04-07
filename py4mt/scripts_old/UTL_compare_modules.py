#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:57:47 2022

@author: vrath
"""

import os
import sys
import ast

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util


F1 = r"/home/vrath/Py4MT/py4mt/modules/util.py"
F2 = r"/home/vrath/AEMpyX/aempy/modules/util.py"
print("\n\n\n")
print(util.list_functions(F1))
print("\n\n")
print(util.list_functions(F2))

F1 = r"/home/vrath/Py4MT/py4mt/modules/modem.py"
F2 = r"/home/vrath/AEMpyX/aempy/modules/modem.py"
print("\n\n\n")
print(util.list_functions(F1))
print("\n\n")
print(util.list_functions(F2))


# tree = util.parse_ast(F1)
# for func in util.find_functions(tree.body):
#     print("  %s" % func.name)
