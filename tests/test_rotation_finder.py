"""
Testing various strategies to find the rotation matrix that maximizes the
matrix Bingham-Fisher-von Mises distribution.
"""
import clouds
import numpy as np

from clouds import euler, frobenius
from clouds.core import take_time
from csb.numeric import euler_angles

def random_model(d=3, beta=10.):
    """
    Generate a random matrix Bingham-Fisher-von Mises distribution for
    testing.
    """
    A = np.random.random((d,d)) * beta
    B = np.random.random((d,d)) * beta
    C = np.random.random((d,d)) * beta

    return clouds.MatrixBingham(A, np.dot(B.T,B), np.dot(C.T,C))

def print_ranking(methods, log_prob):

    ranking = np.argsort(log_prob)[::-1]

    for rank, i in enumerate(ranking,1):
        print '{0}. {1:12s} {2:10.2f}'.format(rank,methods[i],log_prob[i])

if __name__ == '__main__':

    beta      = 1.
    bingham   = random_model(beta=beta)
    finder    = clouds.RotationFinder()
    maximizer = clouds.RotationMaximizer()
    sampler   = clouds.RotationSampler(n_steps=10, stepsize=1e-2)
    methods   = ('random','nedler-mead','powell','bfgs')

    ## study the effect of the number of rotations used in the random
    ## search for the rotation matrix that maximizes the BFvM distribution

    for n_trials in [10,100,1000,10000,100000]:

        finder.n_trials = maximizer.n_trials = n_trials

        angles = []

        ## random search

        with take_time('random search (n_trials={})'.format(n_trials)):
            angles.append(finder.run(bingham))

        ## optimization

        for method in methods[1:]:
            maximizer._optimizer = method        
            with take_time('optimization ({})'.format(method)):
                angles.append(maximizer.run(bingham))

        R = [euler(*a) for a in angles]
        log_p = np.array(map(bingham.log_prob, R))

        print 'n_trials={}'.format(n_trials)
        print_ranking(methods, log_p)
        print

    ## store optimal rotation matrix

    R_opt = R[-1]

    ## run Monte Carlo sampler

    samples = []
    sampler.n_steps = int(1e4)
    sampler.stepsize = 0.1 / beta
    sampler.run(bingham, verbose=True, samples=samples)
    samples = np.array(samples)
    samples = samples # % (2 * np.pi)

    ## evaluate rotations and Frobenius distance to optimal rotation

    R = np.rollaxis(euler(*samples.T),2)
    d = np.array([frobenius(R_opt,RR) for RR in R])
    log_p = np.array(map(bingham.log_prob, R))

    fig, axes = subplots(1,2,figsize=(10,5))
    axes[0].plot(log_p)
    axes[0].axhline(bingham.log_prob(R_opt),ls='--',color='r')
    axes[0].set_ylabel('log prob of Bingham')
    axes[1].plot(d)
    axes[1].set_ylabel('Frobenius distance to R_opt')
    fig.tight_layout()

    ## plot marginal distributions over the Euler angles

    limits  = samples.min(), samples.max(), 50
    bins    = np.linspace(*limits)
    angles_opt = euler_angles(R_opt)

    kw_hist = dict(bins=bins,normed=True,histtype='stepfilled')

    fig, axes = subplots(1,3,figsize=(20,6),subplot_kw=dict(xlim=limits[:2]))
    for k, ax in enumerate(axes.flat):
        ax.hist(samples[:,k],**kw_hist)
        ax.axvline(angles_opt[k],ls='--',color='r')
