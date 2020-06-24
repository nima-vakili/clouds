import os
import glob
import clouds
import numpy as np
import pylab as plt

import utils
from clouds.core import take_time, Parameters
from csb.bio.utils import rmsd

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')
    

if __name__ == '__main__':

    K = 3
    dataset     = 'toy_tau=1'
    params      = utils.setup(dataset, K)
    params.tau  = 1.0
    params.beta = 1.
    rotations   = params.R.copy()
    coords      = params.mu.copy()
    Z           = params.Z.copy()
    
    ## create likelihood and conditional posteriors

    L = clouds.Likelihood(params=params)

    L_max = L.log_prob()

    posterior_Z   = clouds.PosteriorAssignments(params,L)
    posterior_tau = clouds.PosteriorPrecision(params,L)
    posterior_mu  = clouds.PosteriorMeans(params,L)
    posterior_R   = clouds.PosteriorRotations(params,L)

    posteriors = (posterior_mu, posterior_Z, posterior_tau, posterior_R)
    posteriors = (posterior_mu, posterior_Z, posterior_tau)
    #posteriors = (posterior_Z, posterior_R)
    #posteriors = (posterior_R,)

    if False:
    
        prior_Z = clouds.PriorAssignments(params=params)
        prior_Z.sample()
        posterior_Z.sample()
    
    prior_R = clouds.PriorRotations(params=params)
    prior_R.sample()

    ## Gibbs sampling with fixed rotations

    n_iter = 1000
    posterior_R._sampler.n_trials = int(1e3)

    samples = []

    for i in range(n_iter):

        if True:
            for posterior in posteriors[:]:
                posterior.sample()

        else:
            with take_time('\nGibbs iteration {}\n'.format(i)):
                for posterior in posteriors[:]:
                    #with take_time(posterior):
                    posterior.sample()

        LL = L.log_prob()

        print ' {0:.2e}, {1:.2f}, {3:.2f}, ({2:.2e})'.format(LL, params.sigma, L_max, rmsd(params.mu,coords))

        samples.append((params.tau, LL))

    if hasattr(pymol, 'pymol_settings'):
        pymol.pymol_settings += ('set sphere_scale={}'.format(params.sigma),)

    plt.figure()
    plt.plot(zip(*samples)[-1],lw=3)
    plt.axhline(L_max, ls='--', lw=2, color='r')
    plt.ylabel('log likelihood')

    norms = [clouds.frobenius(rotations[i],params.R[i]) for i in range(params.n_projections)]

    utils.show_projections(params, 2, 5, thin=1)

if False:

    files = glob.glob('/tmp/*.npz')

    data = {}

    for fn in files:
        arr = np.load(fn)['arr_0']
        print arr.shape
        name = os.path.basename(fn).split('.')[0]
        data[name] = arr

    params = Parameters(data['mu'].shape[0], data['dataWithTau=1'].shape[1], data['dataWithTau=1'].shape[0])
    params.mu[...] = data['mu']
    params.R[...]  = data['R']
    params.Z[...]  = data['Z']
    params.data = data['dataWithTau=1']
    params.tau = data['Tau=1']

    utils.show_projections(params, 2, 5, thin=1)

    np.savez('./data/toy_tau=1.npz', coords=data['mu'], rotations=data['R'],
             projections=data['dataWithTau=1'], precision=1., 
             assignments=data['Z'])

    np.savez('./data/toy_tau=1_K=5.npz', projections=data['data_5'], rotations=data['R_5'],
             coords=data['mu_5'], precision=data['tau_5'], assignments=data['Z_5'])
