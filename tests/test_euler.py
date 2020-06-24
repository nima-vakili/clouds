"""
Testing evaluation of rotation matrices parameterized in
Euler angles
"""
from clouds.core import take_time
from clouds.rotation import euler, random_euler

from csbplus.statistics.pdf import norm

import numpy as np

n = int(1e5)

alpha, beta, gamma = random_euler(n)

## check distribution of polar angle beta

figure()
bins = hist(beta, bins=n/1000, normed=True, histtype='stepfilled')[1]
plot(bins, np.sin(bins) / norm(bins, np.sin(bins)), color='r', lw=3)

## evaluate rotation matrices

with take_time('numpy'):
    R  = np.rollaxis(euler(alpha,beta,gamma),2)
with take_time('one by one'):
    R2 = np.array([euler(*a) for a in zip(alpha,beta,gamma)])

print 'Rotations are the same? -', np.fabs(R-R2).max() < 1e-10
