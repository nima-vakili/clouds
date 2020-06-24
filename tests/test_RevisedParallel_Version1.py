"""
This is a Revised version of parallel computing for sampling all parameters!
In this case I initialized data with sampling from Priors!

BEFORE RUNNING THE CODE MAKE SURE THAT IN HMC.PY MarginalPosteriorMeans_hmc USES
FIRST VERSION OF PriorMeans!!!!!!!!!!!!!!!!!!
"""

import os, sys
import utils
import clouds
import numpy as np
import pylab as plt
from csb.io import dump, load
from csbplus.mcmc.rex import Replica
from csb.io import load, dump
from copy import deepcopy
from clouds.model_fasterVersion_ballTree import Posterior
from clouds.core import take_time, threaded
from clouds.gibbs import get_state, set_state
from clouds import hmc, frobenius
from clouds.replica import ReplicaExchangeMC, print_rates

if not  False:
    try:
        from csbplus.bio.structure import BeadsOnStringViewer as Viewer
        pymol = Viewer()
    except:
        from csbplus.bio.structure import Viewer
        pymol = Viewer('pymol')

import multiprocessing as mp

def get_stepsizes(gibbs_sampler):
    return [sampler._sampler.stepsize for sampler in gibbs_sampler.pdfs
            if hasattr(sampler, '_sampler')]

def set_stepsizes(gibbs_sampler, stepsizes):        
    samplers = [sampler._sampler for sampler in gibbs_sampler.pdfs
            if hasattr(sampler, '_sampler')]
    if len(samplers) != len(stepsizes):
        msg = "Inconsistency: number of samplers with stepsize doesn't match number of stepsizes"
        raise Exception(msg)

    for sampler, stepsize in zip(samplers, stepsizes):
        sampler.stepsize = stepsize
        
def sample(replica_index):

    global replicas
    global states
    global stepsizes
    
    ensemble = replicas[replica_index]
    set_stepsizes(ensemble, stepsizes[replica_index])
    states[replica_index] = ensemble.run(states[replica_index])[-1]   
    stepsizes[replica_index] = get_stepsizes(ensemble)
    
class ParallelReplicaExchange(ReplicaExchangeMC):

    def __init__(self, posteriors, samplers, n_cpus=1, dataset=None):

        super(ParallelReplicaExchange, self).__init__(posteriors, samplers)
        self.workers = mp.Pool(processes=n_cpus)
        self.dataset = dataset   
        self.stepsizes = []
        
    def move_parallel(self, x):
      
        global states
        global stepsizes
        
        jobs    = xrange(len(x))

        print 'Start parallel computation...'
        
        states[:] = x[:]
        
        for _ in self.workers.imap_unordered(sample, jobs):
            pass

        x[:] = states[:]
            
        print 'Parallel computation finished.'

        self.stepsizes.append(list(stepsizes))

def create_replica(dataset, K, beta, index=None, n_steps=1, n_neighbors=None):

    params      = utils.setup(dataset, K, n_neighbors)  
    params.beta = beta
    params.data = params.data

    if index is not None:
        params.data = params.data[index:index+1]
        params.R = params.R[index:index+1]
    
    ## create Posterior 
    
    priors = [clouds.PriorMeans(params=params),
              clouds.PriorAssignments(params=params),
              clouds.PriorPrecision(params=params),
              clouds.PriorRotations(params=params)]    
        

    L = clouds.Likelihood(params=params)
    posterior = clouds.Posterior(params, L, *priors)
    
    ## random rotations
    for prior in priors[:]: prior.sample() 

    ## create Gibbs sampler
    
    posterior_Z   = clouds.PosteriorAssignments(params,L)
    posterior_tau = clouds.PosteriorPrecision(params,L) 
    posterior_mu  = hmc.MarginalPosteriorMeans_hmc(params,L)
    posterior_R   = hmc.MarginalPosteriorRotations_hmc(params,L,verbose=False )

    samplers = [posterior_mu, posterior_Z, posterior_tau, posterior_R][:]
    
    samplers[0 ]._sampler.n_steps = 2# n_steps
    samplers[-1]._sampler.n_steps = 4#n_steps
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
        
    import time
    T1 = time.clock()

    dataset    = 'dilutedData_1aon'
    K          = 50
    

    beta_0 = np.linspace(1e-5, 1., 25)
    
    ## true parameters
    n_neighbors = None
    params      = utils.setup(dataset, K, n_neighbors)
    params.beta = 1.0
    n_samples = 500
    n_steps = 2
    index   = None 
    
    if index is not None:
        params.data = params.data[index:index+1]
        params.R = params.R[index:index+1]
        
    rotations   = params.R.copy()
    coords      = params.mu.copy()
   
    L = clouds.Likelihood(params=params)
    
    ## setup replicas
    
    manager    = mp.Manager()
    replicas   = manager.list()
    states     = manager.list()
    stepsizes  = manager.list()
    posteriors = []

    for i in range(len(beta)):
    
        posterior, gibbs = create_replica(dataset, K, beta[i], index, n_steps=n_steps, n_neighbors=n_neighbors)
    
        posteriors.append(posterior)
        replicas.append(gibbs)
        stepsizes.append(get_stepsizes(gibbs))
        
    
    ## check consistency of complete likelihood and marginal likelihoods
        
    print posterior.likelihood.log_prob()
    print np.sum([l.log_prob() for l in gibbs.pdfs[-1]._likelihoods])
    
    initial = [get_state(replica.pdfs[0].params) for replica in replicas]
    states[:] = initial[:]
    
    rex = ParallelReplicaExchange(posteriors, replicas, n_cpus=4, dataset=dataset)
 
    if not True:   
        results =threaded(rex.run, initial, n_samples, verbose=True)
    else:
      
        samples, n_acc, means, precisions, angles  = rex.run(initial, n_samples, verbose=True, \
                                                             return_mu=True, return_tau=True, \
                                                             return_rotations=True)
        T2 = time.clock()
        T3 = T2-T1
    

    """
    Data saving
    """

    if  False :
        
        path_0 = '/home/nvakili/Documents/PYTHON_CODES/weekly_experiments/21.3.2018/test_Replica_Parallel/'
 
        log_posterior = np.array(rex.prob_posterior)
        log_prior = np.array(rex.prob_prior)        
        rates = print_rates(rex.n_acc)  
        stepsizes=rex.stepsizes
    
        np.savez(path_0+'angles', angles)
        np.savez(path_0+'samples', samples)
        np.savez(path_0+'n_acc', n_acc)
        np.savez(path_0+'R', params.R)
        np.savez(path_0+'mu', means)
        np.savez(path_0+'tau', precisions)
        np.savez(path_0+'log_posterior', log_posterior)
        np.savez(path_0+'log_prior', log_prior)
        np.savez(path_0+'rates', rates)
        np.savez(path_0+'time', T3)
        np.savez(path_0+'stepsizes', stepsizes)
  
       
    """
    Data loading
    """    
    if  False:

        path_0 = '/home/nvakili/Documents/PYTHON_CODES/weekly_experiments/21.3.2018/test_RevisedParallel_2_4_500_DesktopPriors_newbeta_newPrecisions'
        beta = np.load(os.path.join(path_0, 'newbeta_RevisedParallel_2_4_500.npz'))['arr_0']
        
        path_truth =  '/home/nvakili/Documents/PYTHON_CODES/clouds/tests/data'
        
        P_max= np.load(os.path.join(path_truth, '1aon_TruthPosterior.npz'))['arr_0']
        L_max= np.load(os.path.join(path_truth, '1aon_TruthLikelihood.npz'))['arr_0']
        
        P_max_Version2= np.load(os.path.join(path_truth, '1aon_TruthPosterior_Version2.npz'))['arr_0']
        L_max_Version2= np.load(os.path.join(path_truth, '1aon_TruthLikelihood_Version2.npz'))['arr_0']
        
        P_max_NoneZ= np.load(os.path.join(path_truth, '1aon_TruthPosterior_NoneZ.npz'))['arr_0']
        L_max_NoneZ= np.load(os.path.join(path_truth, '1aon_TruthLikelihood_NoneZ.npz'))['arr_0']
        
        P_max_NoneZ_Version2= np.load(os.path.join(path_truth, '1aon_TruthPosterior_NoneZ_Version2.npz'))['arr_0']
        L_max_NoneZ_Version2= np.load(os.path.join(path_truth, '1aon_TruthLikelihood_NoneZ_Version2.npz'))['arr_0']
        
        R_max= np.load(os.path.join(path_truth, '1aon_R.npz'))['arr_0']
        mu_max= np.load(os.path.join(path_truth, '1aon_mu.npz'))['arr_0']
        Z_max= np.load(os.path.join(path_truth, '1aon_Z.npz'))['arr_0']
        tau_max= np.load(os.path.join(path_truth, '1aon_tau.npz'))['arr_0']
        
        angles = np.load(os.path.join(path_0, 'angles.npz'))['arr_0']
        samples= np.load(os.path.join(path_0, 'samples.npz'))['arr_0']
        n_acc = np.load(os.path.join(path_0, 'n_acc.npz'))['arr_0']
        mu= np.load(os.path.join(path_0, 'mu.npz'))['arr_0']
        tau= np.load(os.path.join(path_0, 'tau.npz'))['arr_0']
        log_posterior= np.load(os.path.join(path_0, 'log_posterior.npz'))['arr_0']
        log_prior= np.load(os.path.join(path_0, 'log_prior.npz'))['arr_0']
        rates= np.load(os.path.join(path_0, 'rates.npz'))['arr_0']
        time= np.load(os.path.join(path_0, 'time.npz'))['arr_0']
        stepsizes= np.load(os.path.join(path_0, 'stepsizes.npz'))['arr_0']
            
            

    if  False:
    
        for state in samples[-1:]:
    
            set_state(params, state)
            utils.show_projections(params, n_rows=5, n_cols=7, thin=1)
    
        import seaborn as sns
    
        sns.set(style='ticks', palette='Paired', context='notebook', font_scale=1.25)
    
        burnin = 0 #25
       
        log_L = log_posterior - log_prior
    
        fig, ax = plt.subplots(2,3,figsize=(16,8))
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
    
        colors = plt.cm.jet(np.linspace(0.,1.,len(beta)))
        for k, x in enumerate(log_L[burnin:].T):
            ax[2].plot(x,color=colors[k])
        ax[2].yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[2].set_ylabel(r'log likelihood all $\beta_i$')
        ax[2].set_xlabel('replica swap')
        ax[2].set_ylim(log_L[burnin:].min(), log_L[burnin:].max())
    
        ax[3].plot(log_posterior[burnin:,-1])
        ax[3].axhline(P_max, ls='--', lw=2, color='r')
        ax[3].yaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[3].set_ylabel('log posterior')
        
        ax[4].plot(log_posterior.sum(-1))
        ax[4].set_ylabel('All Replica Posteriors')
        
        ax[5].plot(beta)
        ax[5].set_ylabel(r'$\beta_i$')
        ax[5].set_ylim(0.,1.)
        ax[5].set_xlim(0,len(beta))
    

    ###########################################################################
        projections = params.data
        set_state(params, samples[-1])
        limits = projections.min(), projections.max()

        fig, axes = plt.subplots(5, 7, figsize=(2*7,2*5),
                                 subplot_kw=dict(xlim=limits, ylim=limits))
    
        for k, ax in enumerate(list(axes.flat)):
    
            ax.scatter(*projections[k][::1].T, color='k', s=5, alpha=0.6)
            ax.scatter(*np.dot(mu[-1], angles[-1][k].T)[:,:2].T,color='r',s=25,alpha=.7)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
     ##########################################################################
     
     
        if hasattr(pymol, 'pymol_settings'):
            pymol.pymol_settings += ('set sphere_scale={}'.format(params.sigma),)
            pymol(mu[-1], cleanup = False)

            
    if False:
    
        from csbplus.statmech.wham import WHAM
        from csbplus.statmech.dos import DOS
        from csbplus.statmech.ensembles import BoltzmannEnsemble
    
        pypath = os.path.expanduser('/home/nvakili/Documents/PYTHON_CODES/adarex/py')
        if not pypath in sys.path: sys.path.insert(0, pypath)
    
        from scheduler import Scheduler, RelativeEntropy, SwapRate
    
        burnin, thin = -20, 1
        
        #LOG_L=np.array([log_L[:,i]/beta[i] for i in range(25)])
        LOG_L = (log_L/beta)
        E = -LOG_L[burnin::thin]#E = -log_L[burnin::thin]
        
        q = np.multiply.outer(beta, E.flatten())
    
        wham = WHAM(q.shape[1], q.shape[0])
        wham._N[...] = E.shape[0]
    
        wham.run(q, niter=int(1e5), tol=1e-10, verbose=1)
    
        dos = DOS(E.flatten(), wham.s)
    
        ensemble = BoltzmannEnsemble(dos=dos) 
        entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)
    
        entropy.find_schedule(2.0, 1e-2, 1., verbose=True)
    
        plt.figure()
        plt.plot(np.linspace(0.,1.,len(beta)),beta)
        plt.plot(np.linspace(0.,1.,len(entropy.schedule)),entropy.schedule)
     
        beta = entropy.schedule.array
        beta[-1] = 1.
     