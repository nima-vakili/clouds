# Clouds #

This package implements a reconstruction algorithm that takes projected point clouds (in 1d or 2d) as input and finds a Gaussian mixture model representing the object (in 2d or 3d) and the unknown projection directions along which the observed point clouds have been projected.

### How do I get set up? ###

* Call

python setup.py install --prefix=${HOME}

for local installation, and

sudo python setup.py install

for global installation.

* Dependencies: numpy, scipy, matplotlib, csb

* To run tests, go to "tests/" directory and run various test scripts

