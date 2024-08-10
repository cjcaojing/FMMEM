# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:41:36 2022
python setup.py build_ext --inplace
@author: WYW
"""

import numpy as np  
from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Distutils import build_ext  

ext_modules = [Extension("fai_m", ["fai_m.pyx"], include_dirs=[np.get_include()]),]

setup(name="function app", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)

