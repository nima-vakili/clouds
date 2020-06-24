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

from clouds.replica import ReplicaExchangeMC, print_rates

try:
    from csbplus.bio.structure import BeadsOnStringViewer as Viewer
    pymol = Viewer()
except:
    from csbplus.bio.structure import Viewer
    pymol = Viewer('pymol')

n_iter     = 100
replicas   = []
posteriors =[]
dataset    = 'toy_tau=1'
K          = 3
dataset    = 'toy_tau=1_K=5'
K          = 5
beta       = np.linspace(0.15,1.,51)
beta       = np.logspace(-3,0,51)

try:
    beta       = load('/tmp/beta2')
    print 'reading schedule from file'
except:
    beta = np.array([ 0.001     ,  0.00182204,  0.00300299,  0.00429244,  0.00562339,
                      0.00735676,  0.00917122,  0.01076005,  0.01230383,  0.01384999,
                      0.01516907,  0.01657517,  0.01794459,  0.01897781,  0.02005689,
                      0.02133335,  0.02263471,  0.02372622,  0.02504237,  0.02653227,
                      0.02788304,  0.029056  ,  0.03062976,  0.03197696,  0.0333532 ,
                      0.03489182,  0.03664962,  0.03873729,  0.04038808,  0.04212328,
                      0.04456229,  0.0468102 ,  0.04912077,  0.05156231,  0.05383642,
                      0.0565233 ,  0.0593926 ,  0.06298603,  0.06599909,  0.06917655,
                      0.07247479,  0.07536343,  0.07911125,  0.08264617,  0.0871459 ,
                      0.09090626,  0.09567244,  0.10103185,  0.10556234,  0.11152449,
                      0.11872626,  0.12457428,  0.12981043,  0.13710167,  0.14519157,
                      0.15466822,  0.16522844,  0.17388561,  0.18531165,  0.20005489,
                      0.21241326,  0.22380724,  0.23729834,  0.2525066 ,  0.27191632,
                      0.29526802,  0.31343729,  0.33197509,  0.3548813 ,  0.37751293,
                      0.40408015,  0.43132895,  0.47128915,  0.50841155,  0.54677133,
                      0.58193675,  0.61388456,  0.63976383,  0.68127363,  0.73800311,
                      0.79971087,  0.8455801 ,  0.90367172,  0.98370124,  1.        ])

    beta = np.array([ 0.001     ,  0.00177598,  0.00272502,  0.00386314,  0.00562125,
                      0.00753441,  0.00906306,  0.01033414,  0.01148671,  0.01293112,
                      0.01442147,  0.01579022,  0.01694744,  0.01823132,  0.01952921,
                      0.02091781,  0.02212828,  0.02329565,  0.02452948,  0.02601076,
                      0.02748397,  0.02863792,  0.03013777,  0.03193276,  0.03308142,
                      0.03465188,  0.0364784 ,  0.03835454,  0.03998162,  0.04156121,
                      0.04391765,  0.04594948,  0.04819464,  0.0510771 ,  0.05390961,
                      0.05639111,  0.05857351,  0.06173452,  0.06459763,  0.06743357,
                      0.07050397,  0.07450644,  0.07937924,  0.08284081,  0.08749598,
                      0.0921005 ,  0.09635189,  0.10055146,  0.10577262,  0.11069118,
                      0.11592046,  0.12166423,  0.12815435,  0.1338864 ,  0.1421899 ,
                      0.15023125,  0.15895703,  0.16809353,  0.17804634,  0.19416928,
                      0.20514173,  0.21912607,  0.23133804,  0.24917496,  0.26804493,
                      0.28490497,  0.30652808,  0.32484855,  0.34200339,  0.36115185,
                      0.38657189,  0.41746023,  0.44311918,  0.46837245,  0.51157218,
                      0.55450184,  0.59319781,  0.63230557,  0.68799447,  0.73577799,
                      0.78802493,  0.83330153,  0.86901071,  0.914433  ,  0.96655501,
                      1.        ])


params = utils.setup(dataset, K)
params.beta = 1.0
params.tau  = 1.0

rotations   = params.R.copy()
coords      = params.mu.copy()

L = clouds.Likelihood(params=params)
posterior_Z = clouds.PosteriorAssignments(params, L)
posterior_Z.sample()

Z     = params.Z.copy()
L_max = L.log_prob()

## setup replicas

for i in range(len(beta)):
    
    params = utils.setup(dataset, K)
    params.beta = beta[i]
    params.tau  = 1.0

    ## create Posterior
    
    priors = [clouds.PriorMeans(params=params),
              clouds.PriorPrecision(params=params),
              clouds.PriorAssignments(params=params),
              clouds.PriorRotations(params=params)]
    
    L = clouds.Likelihood(params=params)
    
    for prior in priors[-1:]: prior.sample() 

    posteriors.append(clouds.Posterior(params, L, *priors)) 

    ## create Gibbs sampler
    
    posterior_Z   = clouds.PosteriorAssignments(params, L)
    posterior_tau = clouds.PosteriorPrecision(params, L)
    posterior_mu  = clouds.PosteriorMeans(params, L)
    posterior_R   = clouds.PosteriorRotations(params, L, verbose=False)
    posterior_R._sampler.stepsize = 0.1
    posterior_R._sampler.n_steps = 10#0
    
    samplers = [posterior_Z, posterior_mu, posterior_tau, posterior_R][:]
    samplers = [posterior_Z, posterior_R][:]

    posterior_R._sampler.n_trials = 1000
    posterior_R._sampler._optimizer = 'powell'

    replicas.append(clouds.GibbsSampler(samplers))

rex = ReplicaExchangeMC(posteriors, replicas)
n_samples = 1000

states = [get_state(replica.pdfs[0].params) for replica in replicas]
initial = map(deepcopy, states)

results = threaded(rex.run, states, n_samples, verbose=True)

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

    fig, ax = subplots(2,3,figsize=(12,8))
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

    colors = cm.viridis(np.linspace(0.,1.,len(beta)))
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

    for a in ax:
        sns.despine(offset=10, trim=True, ax=a)
    
    fig.tight_layout()
    fig2 = utils.show_projections(params, n_rows=2, n_cols=5, thin=1)[0]


    fig.savefig('/tmp/rex.pdf')    
    fig2.savefig('/tmp/projections.pdf')
    
if False:

    from csbplus.statmech.wham import WHAM
    from csbplus.statmech.dos import DOS
    from csbplus.statmech.ensembles import BoltzmannEnsemble

    pypath = os.path.expanduser('~/projects/adarex/py')
    if not pypath in sys.path: sys.path.insert(0, pypath)

    from scheduler import Scheduler, RelativeEntropy, SwapRate

    burnin, thin = -100, 10

    E = -log_L[burnin::thin]
    q = np.multiply.outer(beta, E.flatten())

    wham = WHAM(q.shape[1], q.shape[0])
    wham._N[...] = E.shape[0]

    wham.run(q, niter=int(1e5), tol=1e-10, verbose=1)

    dos = DOS(E.flatten(), wham.s)

    ensemble = BoltzmannEnsemble(dos=dos) 
    entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)

    entropy.find_schedule(1.0, beta.min(), 1., verbose=True)

    plt.figure()
    plt.plot(np.linspace(0.,1.,len(beta)),beta)
    plt.plot(np.linspace(0.,1.,len(entropy.schedule)),entropy.schedule)

    beta = entropy.schedule.array
    beta[-1] = 1.
