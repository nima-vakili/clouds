import os
import time
import utils
import clouds
import numpy as np
import pylab as plt

from scipy.spatial import distance
from csb.bio.utils import distance_matrix
from clouds.core import take_time

K        = 50
dataset  ='dilutedData_1aon'
params   = utils.setup(dataset, K)

i  = 0
x  = params.data[i]
mu = params.mu
R  = params.R[i][:2]

params.distance.__class__.use_csb = True

with take_time('csb (clouds)  '):

    D  = params.distance(x, mu, R)

with take_time('csb           '):

    D2 = distance_matrix(x, np.dot(mu, R.T))**2

params.distance.__class__.use_csb = False

with take_time('scipy         '):

    D4 = distance.cdist(x, np.dot(mu, R[:2].T), 'sqeuclidean')

with take_time('scipy (clouds)'):

    D3 = params.distance(x, mu, R)

print np.fabs(D-D2).max(), np.fabs(D-D3).max(), np.fabs(D-D4).max()
