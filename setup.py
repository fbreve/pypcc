# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:42:02 2026

@author: fbrev
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "pcc_step",
        ["pcc_step.pyx"],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="pcc_step",
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[np.get_include()],
)
