import os
import sys
import imp
import numpy

from setuptools import setup, find_packages, Extension

try:
    __doc__ = open('README.md').read()
except IOError:
    pass

__file__ = './'
ROOT        = 'clouds'
LOCATION    = os.path.abspath(os.path.dirname(__file__))

NAME        = "clouds"
VERSION     = "0.1"
AUTHOR      = "Nima Vakili, Michael Habeck"
EMAIL       = "mhabeck@gwdg.de"
DESCRIPTION = __doc__
LICENSE     = 'MIT'
REQUIRES    = ['numpy', 'scipy', 'csb']

os.environ['CFLAGS'] = '-Wno-cpp'

setup(
    name=NAME,
    packages=find_packages(exclude=('tests',)),
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
    )



