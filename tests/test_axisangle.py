import utils
import clouds
import numpy as np
import pylab as plt

from clouds import hmc

from clouds.marginal import RotationSampler, RotationMaximizer
from clouds.rotation import random_euler, Poses, Pose, euler_angles, euler
from clouds import hmc, frobenius

from scipy import optimize

def check_gradient_energy(params, eps):

    for index in range(params.n_projections):

        energy = hmc.AxisAngleEnergy(index, params=params)
        x      = energy._mapping.params
        grad   = energy.gradient(x)
        num    = grad * 0.
        E      = energy.energy(x)
        
        for i in range(3):
            x[i]  += eps
            num[i] = (energy.energy(x)-E)/eps
            x[i]  -= eps

        print index, np.fabs(grad-num).max()

def rotation(params):
    x = clouds.ExponentialMap(params)
    return x.rotation

if __name__ == '__main__':

    rot  = clouds.ExponentialMap(np.random.random(3))
    rot2 = clouds.ExponentialMap(rot.rotation)

    print rot.params
    print rot2.params

    v   = np.random.random(3)

    print np.dot(rot.rotation, v), rot.rotate(v)

    a = rot.gradient()
    b = a * 0
    R = rot.rotation.copy()
    eps = 1e-7

    for i in range(3):
        rot.params[i] += eps
        b[i,...] = (rot.rotation - R) / eps
        rot.params[i] -= eps

    print np.fabs(a-b).max()

    dataset = 'toy_tau=1_K=5'
    params  = utils.setup(dataset, 5)
    truth   = params.R.copy()
    
    check_gradient_energy(params, 1e-8)

    index  = 0
    energy = hmc.AxisAngleEnergy(index, params=params) 
    start  = np.random.random(3)
    best   = optimize.fmin_bfgs(energy.energy, start, fprime=energy.gradient, disp=False)

    print 'norms before/after optimization:', \
          frobenius(truth[index], euler(*start)),  \
          frobenius(truth[index], euler(*best))

    energy.__class__.gradient_counter = 0
    
    sampler = hmc.HMC(energy, 1e-2, 50)
    sampler.adapt_stepsize = True

    start       = np.random.random(3)
    true_params = clouds.ExponentialMap.from_rotation(truth[index])
    samples     = np.array(sampler.run(start, 200))

    ## map to inner ball

    norms   = np.sum(samples**2,1)**0.5
    samples = samples.T / norms
    samples*= norms % (2*np.pi)
    samples = samples.T
    
    E = np.array(map(energy.energy, samples))
    norms = [frobenius(truth[index], rotation(s)) for s in samples]

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(E)
    ax[0].axhline(energy.energy(true_params),ls='--',color='r',lw=3)    
    ax[1].plot(norms)

if False:
    ## 2nd version

    sampler  = hmc.RotationSampler(50, 1e-2)
    samples2 = [np.random.random(3)]

    while len(samples2) < 100:
        samples2.append(sampler.run(energy, angles=samples2[-1]))
        print sampler.stepsize

    norms2 = [clouds.frobenius(truth[index], rotation(angles)) for angles in samples2]

    plt.figure()
    plt.plot(norms2)
    
