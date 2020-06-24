import os
import utils
import numpy as np

from csbplus.bio.structure import BeadsOnStringViewer as Viewer

pymol = Viewer()

iteration = 2
path      = '/tmp/it{}_results.npz'
results   = np.load(path.format(iteration))

dataset   = 'dilutedData_1aon'
params    = utils.setup(dataset, 50, None)
L_max     = -167555.63284774363

posterior = results['posterior']
prior     = results['prior']
beta      = results['beta']
tau       = results['tau']
mu        = results['mu']
R         = results['R']
rates     = results['rates']

likelihood = posterior - prior

fig, ax = subplots(2,3,figsize=(10,6))
ax = list(ax.flat)

ax[0].plot(rates)
ax[0].set_ylim(0., 1.02)

ax[1].plot(beta)
ax[1].set_ylabel(r'$\beta_i$')

ax[2].plot(tau)
ax[2].set_ylabel(r'$\tau_i$')

ax[3].plot(likelihood.sum(1))
ax[4].plot(posterior[:,-1])
ax[5].plot(likelihood[:,-1])
ax[5].axhline(L_max, ls='--', color='r', lw=3)

for a in ax[2:]:
    a.yaxis.get_major_formatter().set_powerlimits((0, 1))

fig.tight_layout()

params.mu = mu[-1]
params.R  = R[-1]

utils.show_projections(params, thin=1)

if False:

    from csbplus.statmech.wham import WHAM
    from csbplus.statmech.dos import DOS
    from csbplus.statmech.ensembles import BoltzmannEnsemble

    pypath = os.path.expanduser('~/projects/adarex/py')
    if not pypath in sys.path: sys.path.insert(0, pypath)

    from scheduler import Scheduler, RelativeEntropy, SwapRate

    burnin, thin = -200, 5

    E = -likelihood[burnin::thin]
    q = np.multiply.outer(beta, E.flatten())

    wham = WHAM(q.shape[1], q.shape[0])
    wham._N[...] = E.shape[0]

    wham.run(q, niter=int(1e5), tol=1e-10, verbose=1)

    dos = DOS(E.flatten(), wham.s)

    ensemble = BoltzmannEnsemble(dos=dos) 
    entropy  = Scheduler(ensemble, RelativeEntropy(), np.greater)

    entropy.find_schedule(1.0, 1e-3, 1., verbose=True)

    plt.figure()
    plt.plot(np.linspace(0.,1.,len(beta)),beta)
    plt.plot(np.linspace(0.,1.,len(entropy.schedule)),entropy.schedule)

    beta = entropy.schedule.array
    beta[-1] = 1.

    iteration += 1

    np.savez('/tmp/it{}_beta'.format(iteration), beta=beta)

if False:

    eps = results['stepsizes']

    fig, ax = subplots(1,2,figsize=(8,4))
    for i, a in enumerate(ax):
        a.plot(eps[-10:,:,i].mean(0))
    
