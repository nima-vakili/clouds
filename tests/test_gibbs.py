"""
Reconstruction of a projected PDB structure using the correct rotations
"""
import os
import clouds
import utils
import numpy as np
import  pylab as plt
from csb.io import load, dump
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

    K = 50#
    n_neighbors = None
    dataset ='dilutedData_1aon'#'toy_tau=1_K=5'#
    params = utils.setup(dataset, K, n_neighbors)
    params.Z = None
    
    params.beta = 1#e-5
    
    if not True and os.path.exists('/tmp/params.pkl'):
        params = load('/tmp/params.pkl')
 
      
    ## create likelihood and conditional posteriors
    
    priors = [#clouds.PriorMeans(params=params),
              clouds.PriorMeansVersion2(params=params),
              clouds.PriorPrecision(params=params),
#              clouds.PriorAssignments(params=params),
              clouds.PriorRotations(params=params)
             ]  

   #############  initialization of params
    
#    path_init = '/home/nvakili/Desktop'
#    params.mu = np.load(os.path.join(path_init, 'init_mu.npz'))['arr_0']
#    params.tau = np.load(os.path.join(path_init, 'init_tau.npz'))['arr_0']
#    params.R = np.load(os.path.join(path_init, 'init_R.npz'))['arr_0']
    
    #########################################
    
#    priors[0].sample()
#    for prior in priors[:]: prior.sample() 
    
    L = clouds.Likelihood(params=params)
    posterior = clouds.Posterior(params, L, *priors)
    
#    posterior_Z   = clouds.PosteriorAssignments(params,L)

#    posterior_tau = clouds.PosteriorPrecision_NoneZ(params,L)
    posterior_tau = clouds.PosteriorPrecisionVersion2_NoneZ(params,L)

#    posterior_mu  = hmc.MarginalPosteriorMeans_hmc(params,L)
    posterior_mu  = hmc.MarginalPosteriorMeans_hmc_Version2(params,L)
    posterior_mu._sampler.n_steps = 2

    posterior_R   = hmc.MarginalPosteriorRotations_hmc(params,L,verbose=False )
    posterior_R._sampler.n_steps = 2
    
#    posteriors = (posterior_mu, posterior_tau,posterior_Z, posterior_R )
    posteriors = [posterior_mu, posterior_tau, posterior_R][:]

#     Gibbs sampling with fixed rotations
    n_iter = 100#0#1000
    posterior_R._sampler.n_trials = int(1e3)

    sampler = clouds.GibbsSampler(posteriors[:])

    samples = [get_state(params)]

    post =[]
    
#    print 'starting from tau={0:.3e}'.format(L.params.tau)
    
    for i in range(n_iter):
            
        with take_time('Gibbs iteration {}\n'.format(i)):                        
            samples = sampler.run(samples[-1])

        LL = L.log_prob()
#        states = get_state(params)    

        post.append(posterior.log_prob(samples[-1]))
        print 'Likelihood',LL, 'post', post[-1], params.tau
        
#    np.savez('/home/nvakili/Desktop/gibbs_5toys_log', LL)    
#    np.savez('/home/nvakili/Desktop/gibbs_5toys_mu', params.mu)
#    np.savez('/home/nvakili/Desktop/gibbs_5toys_R', params.R)
#    np.savez('/home/nvakili/Desktop/gibbs_5toys_Z', params.Z)
#    np.savez('/home/nvakili/Desktop/gibbs_5toys_tau', params.tau)

#        print ' with log L={0:.2e}, sigma={1:.2f}\n'.format(LL, params.sigma)
#
#    if hasattr(pymol, 'pymol_settings'):
#        pymol.pymol_settings += ('set sphere_scale={}'.format(params.sigma),)
#
#    plt.figure()
#    plt.plot(zip(*samples)[-1],lw=3)
#    plt.ylabel('log likelhood')
#
#    norms = [clouds.frobenius(rotations[i],params.R[i]) for i in range(params.n_projections)]
#
#    utils.show_projections(params, n_rows=5, n_cols=7)
