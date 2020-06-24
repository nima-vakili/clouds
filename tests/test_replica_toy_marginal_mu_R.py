# known Rotation

import os, sys
import utils
import clouds
import numpy as np
import pylab as plt

from csb.io import load, dump
from copy import deepcopy

from clouds.core import take_time, threaded
from clouds.gibbs import get_state, set_state
from clouds import hmc_mu_R, frobenius
from clouds.replica import ReplicaExchangeMC, print_rates

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')

def create_replica(dataset, K, beta, index=None, n_steps=10):

    params      = utils.setup(dataset, K)  
    params.beta = beta
    params.data = params.data
    params.tau  = 1.0
    params.Z    = None
    rotations = params.R.copy()

    if index is not None:
        params.data = params.data[index:index+1]
        params.R = params.R[index:index+1]
    
    ## create Posterior
    
    priors = [clouds.PriorMeans(params=params),
              clouds.PriorRotations(params=params)]
    
    L = clouds.Likelihood(params=params)

    posterior = clouds.Posterior(params, L, *priors)
    
    ## random rotations

    for prior in priors[:]: prior.sample() 

    ## create Gibbs sampler

#    posterior_tau = clouds.PosteriorPrecision(params,L)
    posterior_mu  = hmc_mu_R.MarginalPosteriorMeans_hmc(params,L)
    posterior_R   = hmc_mu_R.MarginalPosteriorRotations_hmc(params,L,verbose=False )

#    samplers = [posterior_mu, posterior_tau, posterior_R][:]
    samplers = [posterior_mu, posterior_R]
#    samplers = [posterior_R]
    samplers[-1]._sampler.n_steps = n_steps
    samplers[-1].optimize = not True

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



if __name__ == "__main__":


    n_iter     = 100
    dataset    = 'toy_tau=1_K=5'
    K          = 5    
    beta = np.array([  1.00000000e-05,   7.96171523e-03,   1.72382123e-02,
                           2.72376303e-02,   5.50758219e-02,   7.52721107e-02,
                           9.65675578e-02,   1.10122746e-01,   6.11853963e-01,
                           8.84751706e-01,   1.00000000e+00])
        
    ## true parameters
    
    params      = utils.setup(dataset, K)
    params.beta = 1.0
    params.tau  = 1.0
    params.Z    = None
    n_steps = 10 #100
    index   = None #9
    
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
    print np.sum([l.log_prob() for l in gibbs.pdfs[-1]._likelihoods])
    
    if False:
        rex = ReplicaExchangeMC(posteriors, replicas)
    else:
        rex = RotationReplicaExchange(posteriors, replicas)

    n_samples = 2000
    
    states = [get_state(replica.pdfs[0].params) for replica in replicas]
    initial = map(deepcopy, states)
    
    results =threaded(rex.run, states, n_samples, verbose=True)
#    results = rex.run(states, n_samples, verbose=True)


#    if hasattr(pymol, 'pymol_settings'):
#        pymol.pymol_settings += ('set sphere_scale={}'.format(params.sigma),)
#        pymol([params.mu,replicas[-1].pdfs[0].params.mu])

    if False:
    
        for state in rex.state[-1:]:
    
            set_state(params, state)
            utils.show_projections(params, n_rows=2, n_cols=5, thin=1)
    
        import seaborn as sns
    
        sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.25)
    
        burnin = 0 #25
        params = replicas[-1].pdfs[0].params 
        norms  = [clouds.frobenius(rotations[i],params.R[i]) for i in range(params.n_projections)]
        log_posterior = np.array(rex.prob_posterior)
        log_prior = np.array(rex.prob_prior)
            
        log_L = log_posterior - log_prior
        rates = print_rates(rex.n_acc)
    
        fig, ax = plt.subplots(2,4,figsize=(16,8))
        ax = ax.flat
        ax[0].plot(log_L[burnin:,-1])
        ax[0].axhline(L_max, ls='--', lw=2, color='r')
        ax[0].set_ylabel(r'log likelihood ($\beta$=1)')
        ax[0].set_xlabel('replica swap')
        ax[0].yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[0].set_xlim(0,len(log_L)-burnin)
            
        if False:
            ax[1].bar(np.arange(len(rates)),rates)
        else:
            ax[1].plot(rates)
        ax[1].set_ylim(0.,1.)
        ax[1].set_xlabel('replica pairs')
        ax[1].set_ylabel('swap rate')
        ax[1].set_xlim(0,len(beta))
    
        #colors = plt.cm.viridis(np.linspace(0.,1.,len(beta)))
        colors = plt.cm.jet(np.linspace(0.,1.,len(beta)))
        for k, x in enumerate(log_L[burnin:].T):
            ax[2].plot(x,color=colors[k])
        ax[2].yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[2].set_ylabel(r'log likelihood all $\beta_i$')
        ax[2].set_xlabel('replica swap')
        ax[2].set_ylim(log_L[burnin:].min(), log_L[burnin:].max())
    
        if False:
            ax[3].plot(norms)
        else:
            ax[3].bar(np.arange(params.n_projections), norms)
            
        ax[3].set_ylim(0., 1.5)
        ax[3].set_ylabel('Frobenius distance')
        ax[3].set_xlabel('projection direction')
        ax[3].set_xticks(np.arange(params.n_projections))
        
        ax[4].plot(log_posterior.sum(1))
        ax[4].yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[4].set_ylabel('log posterior')
            
        ax[5].plot(beta)
        ax[5].set_ylabel(r'$\beta_i$')
        ax[5].set_ylim(0.,1.)
        ax[5].set_xlim(0,len(beta))
    
        for k, x in enumerate(log_L[-1000:].T):
            if k < 3: continue
            ax[6].hist(x,color=colors[k],bins=30,normed=True,alpha=0.6,histtype='stepfilled')
        ax[6].semilogy()
        
        ax[7].plot([r.pdfs[-1]._sampler.stepsize for r in replicas])
        
        for a in ax:
            sns.despine(offset=10, trim=True, ax=a)
    
        sns.despine(offset=10, trim=True, ax=ax[6], left=True)
    
        fig.tight_layout()
        fig2 = utils.show_projections(params, n_rows=2, n_cols=5, thin=1)[0]
    
    
        fig.savefig('/tmp/rex.pdf')    
        fig2.savefig('/tmp/projections.pdf')
