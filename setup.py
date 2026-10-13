# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:42:02 2026

@author: fbrev
"""

from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np


class PortableBuildExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        flags = {"msvc": ["/O2"], "unix": ["-O3"], "mingw32": ["-O3"]}.get(compiler, [])
        for extension in self.extensions:
            extension.extra_compile_args = flags.copy()
        super().build_extensions()

extensions = [
    Extension(
        "pcc_step",
        ["pcc_step.pyx"],
        include_dirs=[np.get_include()],
        extra_link_args=[],
    )
]

setup(
    name="pcc_step",
    version=Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip(),
    python_requires=">=3.10",
    py_modules=["pcc", "pcc_graph", "pcc_numpy", "pcc_numba", "pcc_validation", "pcc_metrics"],
    install_requires=["numpy>=2,<3", "scipy", "scikit-learn"],
    extras_require={"numba": ["numba"]},
    ext_modules=cythonize(extensions, language_level="3", force=True),
    cmdclass={"build_ext": PortableBuildExt},
)
