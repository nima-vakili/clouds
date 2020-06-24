"""
Reconstruction of a projected PDB structure using the correct rotations
"""
import os
import clouds
import utils
import numpy as np
import pylab as plt

from csb.io import load, dump
from clouds.core import take_time

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')

if __name__ == '__main__':

    K = 50
    dataset = ('1aon','bunny')[0]
    params = utils.setup(dataset, K)
    params.data = params.data[:,::10]
    params.Z = params.Z[:,::10]
    params.beta = 1.#95
    rotations = params.R.copy()
    
    if True and os.path.exists('/tmp/params.pkl'):
        params = load('/tmp/params.pkl')

    ## randomize rotations and assignments

    prior_R = clouds.PriorRotations(params=params)
#    prior_R.sample()

    prior_Z = clouds.PriorAssignments(params=params)
    prior_Z.sample()
    
    ## create likelihood and conditional posteriors

    L = clouds.Likelihood(params=params)

    posterior_Z   = clouds.PosteriorAssignments(params,L)
    posterior_tau = clouds.PosteriorPrecision(params,L)
    posterior_mu  = clouds.PosteriorMeans(params,L)
    posterior_R   = clouds.PosteriorRotations(params,L)

    posteriors = (posterior_mu, posterior_Z, posterior_tau, posterior_R)
    posteriors = (posterior_Z, posterior_tau, posterior_R)
    posteriors = (posterior_mu, posterior_Z, posterior_tau)
 
    ## Gibbs sampling with fixed rotations

    n_iter = 1000
    posterior_R._sampler.n_trials = int(1e4)

    samples = []

    for i in range(n_iter):

        with take_time('Gibbs iteration {}\n'.format(i)):
            for posterior in posteriors[:]:
                #with take_time(posterior):
                posterior.sample()

        LL = L.log_prob()

        print ' with log L={0:.2e}, sigma={1:.2f}\n'.format(LL, params.sigma)

        samples.append((params.tau, LL))

    if hasattr(pymol, 'pymol_settings'):
        pymol.pymol_settings += ('set sphere_scale={}'.format(params.sigma),)

    plt.figure()
    plt.plot(zip(*samples)[-1],lw=3)
    plt.ylabel('log likelihood')

    norms = [clouds.frobenius(rotations[i],params.R[i]) for i in range(params.n_projections)]

    fig=utils.show_projections(params, n_rows=5, n_cols=7, thin=1)

if False:

    import seaborn as sns
    
    sns.set(style='ticks', palette='Paired', context='notebook', font_scale=3.25)

    fig, ax = subplots(1, 1, figsize=(8,8))
    ax.plot(np.array(zip(*samples)[0])**(-0.5), lw=5, color='k', alpha=0.8)
    ax.set_xlabel(r'Monte Carlo iteration')
    ax.set_ylabel(r'bead size $\sigma$ [$\AA$]')
    ax.set_xlim(-10, 500)
    ax.set_ylim(9, 38)
    
    sns.despine(offset=10, trim=True, ax=ax)

    fig.savefig('/tmp/sigma.pdf', transparent=True, bbox_inches='tight')
