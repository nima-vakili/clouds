# known Rotation

import os
import utils
import clouds
import numpy as np

from csb.io import load, dump
from copy import deepcopy

from clouds.core import take_time, threaded
from clouds.gibbs import get_state, set_state

from clouds.replica import ReplicaExchangeMC, print_rates

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')

n_iter  = 100
replicas = []
posteriors =[]
beta = np.linspace(0.95,1.,11)
K = 50

## setup replicas

for i in range(len(beta)):
    
    params = utils.setup('1aon', K)
    params.beta = beta[i]
    
    ## dilute the data

    params.data = params.data[:,::10]    
    params.Z = params.Z[:,::10]
    
    ## create Posterior
    
    priors = [clouds.PriorMeans(params=params),
              clouds.PriorPrecision(params=params),
              clouds.PriorAssignments(params=params),
              clouds.PriorRotations(params=params)]
    
    L = clouds.Likelihood(params=params)
    
    for prior in priors[:]: prior.sample() 

    posteriors.append(clouds.Posterior(params, L, *priors)) 

    ## create Gibbs sampler
    
    posterior_Z   = clouds.PosteriorAssignments(params, L)
    posterior_tau = clouds.PosteriorPrecision(params, L)
    posterior_mu  = clouds.PosteriorMeans(params, L)
    posterior_R   = clouds.PosteriorRotations(params, L)
    
    samplers = [posterior_Z, posterior_mu, posterior_tau, posterior_R][:]
    
    replicas.append(clouds.GibbsSampler(samplers))

rex = ReplicaExchangeMC(posteriors, replicas)
n_samples=1000

states = [get_state(replica.pdfs[0].params) for replica in replicas]
initial = map(deepcopy, states)

results = threaded(rex.run, states, n_samples, verbose=True)

if False:

    for state in rex.state:

        set_state(params, state)
        utils.show_projections(params, n_rows=5, n_cols=7)

    ## compute priors only

    log_lik = np.array(rex.prob_posterior) - np.array(rex.prob_prior)
    
