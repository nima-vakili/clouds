# known Rotation

import os, sys
import utils2
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
    
    
INDEX = 12
n_iter     = 100
K          = 50
dataset = ('dilutedData_1aon')
beta       = np.linspace(0.15,1.,51)
beta       = np.logspace(-3,0,51)


try:
    beta   = load('/tmp/beta')
    print 'reading schedule from file'

except:

    ## various versions of the optimized inverse temperature schedule
    
    beta = np.array([ 0.001     ,  0.00208235,  0.00402593,  0.00705048,  0.01049323,
                      0.01539816,  0.02148526,  0.02894992,  0.03865213,  0.0548713 ,
                      0.07740861,  0.11002925,  0.1570314 ,  0.2081295 ,  0.26740708,
                      0.34843362,  0.44923469,  0.5633238 ,  0.71027989,  0.80423235,
                      0.8843339 ,  0.94387093,  0.97956044,  1.        ])

    beta = np.array([ 0.001     ,  0.00230677,  0.00442244,  0.00759709,  0.01135677,
                      0.01651594,  0.02283097,  0.03102805,  0.04061577,  0.05715313,
                      0.07952988,  0.11961811,  0.1655176 ,  0.22440819,  0.30718875,
                      0.40215292,  0.51447057,  0.63219308,  0.73097097,  0.82630536,
                      0.94081785,  1.        ])

    beta = np.array([ 0.001     ,  0.00233424,  0.00399029,  0.00647763,  0.01027661,
                      0.01482463,  0.02100311,  0.02879456,  0.0386533 ,  0.05537244,
                      0.07566507,  0.11251927,  0.16222576,  0.22354644,  0.32179602,
                      0.42946605,  0.54436059,  0.66539307,  0.76161639,  0.85092669,
                      0.93435774,  1.        ])

    beta = np.array([ 0.001     ,  0.00208861,  0.00388253,  0.00681466,  0.01094567,
                      0.01514421,  0.02015655,  0.02826645,  0.03663346,  0.04714932,
                      0.0692529 ,  0.09371704,  0.13115818,  0.20468956,  0.27904567,
                      0.37471948,  0.51254331,  0.72239295,  0.84002057,  0.91949762,
                      1.        ])

    beta = np.array([ 0.001     ,  0.01787899,  0.09074656,  0.5186014 ,  0.83423169,  1.        ])

    beta = np.array([ 0.001     ,  0.00673676,  0.01537718,  0.02140564,  0.07041841,
                      0.10434314,  0.13540019,  0.15681377,  0.38823744,  0.57132095,
                      0.71679611,  0.82994511,  1.        ])

    beta = np.array([ 0.001     ,  0.01247368,  0.02549558,  0.04715625,  0.08743799,
                      0.13684388,  0.18548035,  0.26112722,  0.35275257,  0.43021679,
                      0.63444781,  0.83801245,  1.        ])

    beta = np.array([ 0.001     ,  0.02804036,  0.14630392,  0.31478714,  0.62308716,  1.        ])

    beta = np.array([ 0.001     ,  0.01347133,  0.17641962,  0.25493745,  0.81693063,  1.        ])

    beta = np.array([  1.00000000e-05,   7.96171523e-03,   1.72382123e-02,
                       2.72376303e-02,   5.50758219e-02,   7.52721107e-02,
                       9.65675578e-02,   1.10122746e-01,   6.11853963e-01,
                       8.84751706e-01,   1.00000000e+00])
    
    path = '/home/nvakili/Documents/clouds/tests/data/new_betas/'

    beta = np.load(os.path.join(path,'beta_index='+str(INDEX)+'.npz'))['arr_0']
    
    
## true parameters

params      = utils2.setup(dataset, K)
params.beta = 1.0
params.Z    = None

n_steps = 10 #100
index   = INDEX 

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



















