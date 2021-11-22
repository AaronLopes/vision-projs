#!/usr/bin/env python

"""
Ref: https://github.com/argoai/argoverse-api/blob/master/setup.py
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


setup(
    name="CSx476-Fall2021-Proj5",
    version="1.0.0",
    description="",
    long_description="Project 5",
    url="https://www.cc.gatech.edu/~hays/compvision/",
    author="Georgia Institute of Technology",
    author_email="akrishna@gatech.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision",
    packages=find_packages(),
    python_requires=">= 3.5",
    install_requires=[
        "pytest"
    ]
)