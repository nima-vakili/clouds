import os
import utils2
import clouds
import numpy as np

from csb.io import load, dump
from clouds.core import take_time

## setup model

K = 50
params = utils2.setup('dilutedData_1aon', K)

if os.path.exists('/tmp/params.pkl'):
    params = load('/tmp/params.pkl')

## set parameters globally in all subclasses of Probability

clouds.Probability.set_params(params)

## create likelihood and conditional posteriors

L = clouds.Likelihood()

posterior_Z   = clouds.PosteriorAssignments(L)
posterior_tau = clouds.PosteriorPrecision(L)
posterior_mu  = clouds.PosteriorMeans(L)

posteriors = (posterior_mu, posterior_Z, posterior_tau)

## Gibbs sampling with fixed rotations

n_iter  = 5
samples = []

for i in range(n_iter):

    with take_time('Gibbs iteration {}\n'.format(i)):
        for posterior in posteriors[:]:
            posterior.sample()

    LL = L.log_prob()

    print ' with log L={0:.2e}, sigma={1:.2f}\n'.format(LL, params.sigma)

    samples.append((params.tau, LL))

posterior_R = clouds.PosteriorRotations(L)

with take_time('sampling rotations'):
    posterior_R.sample()

norms = [clouds.frobenius(rotations[i], params.R[i]) for i in range(params.n_projections)]

