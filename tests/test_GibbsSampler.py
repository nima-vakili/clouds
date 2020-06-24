"""
Using the Gibbs sampler for generating the parameters with Version one and Two
In this case we set the assignments to None.
"""
import os
import time
import utils
import clouds
import numpy as np
import pylab as plt

from csb.io import load, dump
from csb.bio.utils import radius_of_gyration

from clouds.core import take_time
from clouds.gibbs import get_state, set_state
from clouds import hmc

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')

if __name__ == '__main__':

    K = 50
    n_neighbors = None
    dataset ='dilutedData_1aon'
    params = utils.setup(dataset, K, n_neighbors)
    params.Z = None    
    params.beta = 1

    #params.distance.__class__.use_csb = False
    
    sample_mu  = True
    sample_R   = not True
    sample_tau = True
    tau_0      = 1e-2
    n_hmc      = 10
    
    initialize_from_prior = not False
    
    if not True and os.path.exists('/tmp/params.pkl'):
        params = load('/tmp/params.pkl')
 
    
    L = clouds.Likelihood(params=params)

    if  False:
        
        ## Using Version one for PriorMean  and Posterior Precision

        priors = [
                   clouds.PriorMeans(params=params),
                   clouds.PriorPrecision(params=params),
                   clouds.PriorRotations(params=params)
                  ]  
       
        posterior_tau = clouds.PosteriorPrecision_NoneZ(params,L)
        posterior_mu  = hmc.MarginalPosteriorMeans_hmc(params,L)
    
    else :
        
        ## using Version Two for PriorMean and Posterior Precision

        priors = [
                  clouds.PriorMeansVersion2(params=params),
                  clouds.PriorPrecision(params=params),
                  clouds.PriorRotations(params=params)
                 ]  
   
        posterior_tau = clouds.PosteriorPrecisionVersion2_NoneZ(params,L) 
        posterior_mu  = hmc.MarginalPosteriorMeans_hmc_Version2(params,L)

    posterior = clouds.Posterior(params, L, *priors)
    
    posterior_R = hmc.MarginalPosteriorRotations_hmc(params,L,verbose=False )

    ## set hyperparameters

    for name in ('tau_0','alpha_0','beta_0'):
        setattr(params, name, tau_0)
    
    print '(hyper)parameters:'
    for name in ('beta', 'tau', 'tau_0', 'alpha_0', 'beta_0'):
        print '{0:10s}: {1:.2f}'.format(name, getattr(params, name))

    ## set number of HMC steps and initial stepsizes

    for hmc in (posterior_R, posterior_mu):
        hmc._sampler.n_steps  = n_hmc
        hmc._sampler.stepsize = 1e-2
    
    posteriors = []

    if sample_mu:
        posteriors.append(posterior_mu)
    if sample_tau:
        posteriors.append(posterior_tau)
    if sample_R:
        posteriors.append(posterior_R)

    n_iter = int(1e4)
    posterior_R._sampler.n_trials = int(1e3)

    sampler = clouds.GibbsSampler(posteriors)

    state = get_state(params)
    info  = {'posterior': [posterior.log_prob(state)],
             'likelihood': [L.log_prob()],
             'Rg': [radius_of_gyration(params.mu)],
             'mu': [state[0].copy()],
             'tau': [float(state[2])],
             'eps_mu': [posterior_mu._sampler.stepsize],
             'eps_R': [posterior_R._sampler.stepsize]}

    # sample from priors for initialization

    if initialize_from_prior:

        for prior in priors:
            name = prior.__class__.__name__ 
            if name.startswith('PriorMeans') and sample_mu:
                print 'sampling from', name
                prior.sample()
            elif name.startswith('PriorPrecision') and sample_tau:
                print 'sampling from', name
                prior.sample()
            elif name.startswith('PriorRotations') and sample_R:
                print 'sampling from', name
                prior.sample()

    state = get_state(params)

    ## Gibbs sampling
        
    for i in range(n_iter):

        t = time.clock()
        state = sampler.run(state)[-1]
        t = time.clock() - t
        
        state = get_state(params)

        ## update info
        
        info['likelihood'].append(L.log_prob())
        info['posterior'].append(posterior.log_prob(state))
        info['Rg'].append(radius_of_gyration(state[0]))
#        info['mu'].append(np.copy(state[0]))
        info['tau'].append(float(state[2]))
        info['eps_mu'].append(posterior_mu._sampler.stepsize)
        info['eps_R'].append(posterior_R._sampler.stepsize)

        print '-' * 50
        print 'iteration: {0} (time: {1:.3f}s)'.format(i, t)
        print '-' * 50
        for name in ('likelihood', 'posterior', 'tau', 'Rg','eps_mu','eps_R'):
            print '{0:15s}:{1:15.2e}'.format(name, info[name][-1])

    fig, ax = subplots(2,3,figsize=(12,8))
    ax = list(ax.flat)
    for i, name in enumerate(['tau','likelihood','posterior','Rg','eps_mu','eps_R']):
        ax[i].plot(info[name][1:])
        ax[i].axhline(info[name][0],ls='--',lw=2,color='r')
        ax[i].set_ylabel(name)
    fig.tight_layout()
