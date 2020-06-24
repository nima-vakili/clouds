import clouds
import utils
import numpy as np
import pylab as plt

from clouds.marginal import RotationSampler, RotationMaximizer
from clouds.rotation import random_euler, Poses, Pose, euler_angles, euler
from clouds import hmc, frobenius

from scipy import optimize

def check_gradient_likelihood(params, eps=1e-8):

    for index in range(params.n_projections):

        L = clouds.MarginalLikelihood_R(index, params=params)
#        L = clouds.FasterMarginalLikelihood_R(index, 3, params=params)

        angles = euler_angles(params.R[index])
        grad   = L.gradient()
        log_p  = L.log_prob()
        num    = grad * 0

        for i in range(3):
            for j in range(3):
                params.R[index,i,j] += eps
                num[i,j] = (L.log_prob() - log_p) / eps
                params.R[index,i,j] -= eps
        
#        print 'Testing balltree with {} nearest neighbors'.format(L.n_neighbors)
        print '-' * 25 + ' projection direction {0:d} '.format(index) + '-' * 25
        print 'analytical gradient'
        print np.round(grad.flatten(), 3)

        print 'numerical gradient'
        print np.round(num.flatten(), 3)
        print 
    
def check_gradient_energy(params, eps):

    for index in range(params.n_projections):

        energy = hmc.EulerEnergy(index, params=params)
        angles = np.array(euler_angles(params.R[index]))
        grad   = energy.gradient(angles)
        num    = grad * 0.
        E      = energy.energy(angles)
        
        for i in range(3):
            angles[i] += eps
            num[i]     = (energy.energy(angles)-E)/eps
            angles[i] -= eps

        print index, np.fabs(grad-num).max()

def check_gradient_energy2(energy, n=1e3):

    results = []

    for i in xrange(int(n)):

        angles = random_euler()

        a = energy.gradient(angles)
        b = optimize.approx_fprime(angles, energy.energy, 1e-8)

        results.append(np.fabs(a-b).m+
                       ax())

    return np.array(results)

if __name__ == '__main__':

    dataset     = 'toy_tau=1_K=5'
    params      = utils.setup(dataset, 5)
    params.beta = 0.1
    truth       = params.R.copy()

    if not False:
        eps  = 1e-7
        check_gradient_likelihood(params, eps)
        check_gradient_energy(params, eps)

    index  = 0
    energy = hmc.EulerEnergy(index, params=params) 
    start  = np.array(random_euler())
    best   = optimize.fmin_bfgs(energy.energy, start, fprime=energy.gradient, disp=False)

    print 'norms before/after optimization:', \
          frobenius(truth[index], euler(*start)),  \
          frobenius(truth[index], euler(*best))

    energy.__class__.gradient_counter = 0
    
    sampler = hmc.HMC(energy, 1e-2, 50)
    sampler.adapt_stepsize = True

    angles  = np.array(random_euler())
    true_angles = np.array(euler_angles(truth[index]))
    samples = sampler.run(angles, 1000)
    
    E = np.array(map(energy.energy, samples))
    norms = [frobenius(truth[index], euler(*angles)) for angles in samples]

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(E)
    ax[0].axhline(energy.energy(true_angles),ls='--',color='r',lw=3)    
    ax[1].plot(norms)

    ## 2nd version

    sampler  = hmc.RotationSampler(50, 1e-2)
    samples2 = [np.array(random_euler())]

    while len(samples2) < 100:
        samples2.append(sampler.run(energy, angles=samples2[-1]))
        print sampler.stepsize

    norms2 = [clouds.frobenius(truth[index], euler(*angles)) for angles in samples2]

    plt.figure()
    plt.plot(norms2)
    
if False:

    ## trying to visualize energy landscape

    n = 50
    a = np.linspace(0., 2.*np.pi, n)
    b = np.linspace(0., np.pi, n)
    c = np.linspace(0., 2*np.pi, n)

    f = []
    for aa in a:
        for bb in b:
            for cc in c:
                f.append(energy.energy(np.array([aa,bb,cc])))

    beta = 0.01
    f = np.reshape(f, (n,n,n))
    f = np.exp(-beta * (f-f.min()))
