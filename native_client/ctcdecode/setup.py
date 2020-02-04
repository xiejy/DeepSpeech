#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from setuptools import setup
from setuptools.dist import Distribution

import os
import sys

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

numpy_min_ver = os.getenv('NUMPY_DEP_VERSION', '')
project_version = 'v0.7.0-alpha.1'

class BinaryDistribution(Distribution):
  def has_ext_modules(self):
    return True

setup(
    name='ds_ctcdecoder',
    version=project_version,
    description='DS CTC decoder',
    include_package_data=True,
    package_data={
        'ds_ctcdecoder': [
            '_ctcdecode_wrap_internal.so',
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    packages=['ds_ctcdecoder'],
    py_modules=[
        'ds_ctcdecoder',
        'ds_ctcdecoder.ctcdecode_wrap_internal'
    ],
    install_requires = ['numpy%s' % numpy_min_ver],
)
