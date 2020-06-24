# known Rotation

import os, sys
import utils2, utils
import clouds
import numpy as np
import pylab as plt
import cProfile, pstats
from csb.io import load, dump
from copy import deepcopy

from clouds.core import take_time, threaded
from clouds.gibbs import get_state, set_state
from clouds.replica import ReplicaExchangeMC, print_rates
from clouds.rotation import random_euler, Poses, Pose, euler_angles, euler
from clouds import hmc, frobenius

from scipy import optimize

def create_replica(dataset, K, beta, index=None, n_steps=10):
        
    params      = utils2.setup(dataset, K)
    params.data = params.data
    params.beta = beta
    params.Z    = None
    rotations = params.R.copy()


    if index is not None:
        params.data = params.data[index:index+1]
        params.R = params.R[index:index+1]
    
    ## create Posterior
    
    priors = [clouds.PriorMeans(params=params),
              clouds.PriorPrecision(params=params),
              clouds.PriorAssignments(params=params),
              clouds.PriorRotations(params=params)]
    
    L = clouds.Likelihood(params=params)

    posterior = clouds.Posterior(params, L, *priors)
    
    ## random rotations
    
    for prior in priors[-1:]: prior.sample() 

    ## create Gibbs sampler
    
    samplers = [hmc.MarginalPosteriorRotations_hmc(params, L, verbose=False)]
    samplers[0]._sampler.n_steps = n_steps
    samplers[0].optimize = not True
    
    return posterior, clouds.GibbsSampler(samplers)

class RotationReplicaExchange(ReplicaExchangeMC):

    counter = 0
    projection_index = 0
    
    def propose_swap(self, x, y):

        x2 = deepcopy(x)
        y2 = deepcopy(y)

        if self.counter == len(self.samplers) - 2:
            self.projection_index = np.random.randint(len(x[-1]))
            self.counter = 0
            
        k  = self.projection_index
        self.counter += 1
        
        x2[-1][k,...], y2[-1][k,...] = y[-1][k,...], x[-1][k,...]

        return x2, y2
    
    
n_iter     = 100
K          = 50
dataset = ('1aon')
path1 = '/home/nvakili/Documents/clouds/tests/data/new_betas/'    
   
## true parameters

params      = utils.setup(dataset, K)
params.beta = 1.0
params.Z    = None

n_steps = 10 #100
    
for INDEX in range(35):
     
    index   = INDEX 
    beta = np.load(os.path.join(path1,'beta_index='+str(INDEX)+'.npz'))['arr_0']
    if index is not None:
        params.data = params.data[index:index+1]
        params.R = params.R[index:index+1]
    #    beta = np.append(beta[::5], 1.)
        
    rotations   = params.R.copy()
    coords      = params.mu.copy()
    
    L = clouds.Likelihood(params=params)
    L_max = L.log_prob()
    
    ## setup replicas
    
    replicas   = []
    posteriors = []
    
    for i in range(len(beta)):
    
        posterior, gibbs = create_replica(dataset, K, beta[i], index, n_steps=n_steps)
    
        posteriors.append(posterior)
        replicas.append(gibbs)
    
    ## check consistency of complete likelihood and marginal likelihoods
        
    print posterior.likelihood.log_prob()
    print np.sum([l.log_prob() for l in gibbs.pdfs[0]._likelihoods])
    
    if False:
        rex = ReplicaExchangeMC(posteriors, replicas)
    else:
        rex = RotationReplicaExchange(posteriors, replicas)
    n_samples = 1000
    
    states = [get_state(replica.pdfs[0].params) for replica in replicas]
    initial = map(deepcopy, states)
    samples, n_acc, angles = rex.run(states, n_samples, verbose = True, return_rotations=True)
    
    params = replicas[-1].pdfs[0].params 
    
    path = '/home/nvakili/Documents/clouds/tests/data/'
    marginalR = np.load(os.path.join(path,'marginalR_100.npz'))['arr_0']
    norms  =[clouds.frobenius(marginalR[index], x[-1][-1]) for x in samples]
    log_posterior = np.array(rex.prob_posterior)
    log_prior = np.array(rex.prob_prior)
        
    log_L = log_posterior - log_prior
    rates = print_rates(rex.n_acc)
    
    ################### save #######################
    
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/angles_index='+str(INDEX), angles)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/samples_index='+str(INDEX), samples)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/n_acc_index='+str(INDEX), n_acc)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/R_index='+str(INDEX), params.R)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/replicas_index='+str(INDEX),replicas)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/rotations_index='+str(INDEX), rotations)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/states_index='+str(INDEX),states)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/posteriors_index='+str(INDEX),posteriors)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/log_posterior_index='+str(INDEX),log_posterior)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/log_prior_index='+str(INDEX),log_prior)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/rates_index='+str(INDEX),rates)
    np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/index_'+str(INDEX)+'/norms_index='+str(INDEX),norms)




















