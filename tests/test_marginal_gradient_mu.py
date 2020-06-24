import clouds
import utils
import numpy as np
import pylab as plt
from clouds.marginal_fasterVersion_ballTree import  FasterMarginalLikelihood_mu
from clouds.marginal import RotationSampler, RotationMaximizer, MarginalLikelihood_mu
from clouds.prior import PriorMeans
from clouds.rotation import random_euler, Poses, Pose, euler_angles, euler
from clouds import  frobenius, hmc
from clouds import hmc_mu_R
import time

from scipy import optimize

## without prior_mu

def check_gradient_energy(params, eps, K, D):

    energy = hmc.MeanEnergy(params=params)
    samples = np.random.random((K, D+1))
    grad   = energy.gradient(samples)
    num    = grad * 0.
    E      = energy.energy(samples)
    
    for i in range(len(samples)):
        for j in range(samples.shape[1]):
            samples[i,j] += eps
            num[i,j]      = (energy.energy(samples)-E)/eps
            samples[i,j] -= eps

    print np.fabs(grad-num).max()

## with prior

def check_gradient2_energy(params, pdfs, eps, K, D):

    energy = hmc.MeanEnergy(params, pdfs)
    samples = np.random.random((K, D+1))
    grad   = energy.gradient(samples)
    num    = grad * 0.
    E      = energy.energy(samples)
    
    for i in range(len(samples)):
        for j in range(samples.shape[1]):
            samples[i,j] += eps
            num[i,j]      = (energy.energy(samples)-E)/eps
            samples[i,j] -= eps

    print np.fabs(grad-num).max()
    
    
def check_gradient_ballTree(params, pdfs, eps, K, D):

    energy = hmc_mu_R.MeanEnergy(params, pdfs)
    samples = np.random.random((K, D+1))
    grad   = energy.gradient(samples)
    num    = grad * 0.
    E      = energy.energy(samples)
    
    for i in range(len(samples)):
        for j in range(samples.shape[1]):
            samples[i,j] += eps
            num[i,j]      = (energy.energy(samples)-E)/eps
            samples[i,j] -= eps

    print np.fabs(grad-num).max()


if __name__ == '__main__':

    dataset     ='dilutedData1aon_newMethod_50'#'toy_tau=1_K=5'#'1aon'
    K = 50
    D = 2
 
    eps  = 1e-7
    params      = setup(dataset, K)
    params.beta = 0.1
    truth       = params.mu.copy()

    
    pdfs  = [MarginalLikelihood_mu(params=params), PriorMeans(params=params)]
 
    pdfs2 = [FasterMarginalLikelihood_mu(params=params), PriorMeans(params=params)] 
        
    
#    check_gradient_energy(params, eps, K, D)
    
    
    a1=time.clock()
#    check_gradient_ballTree(params, pdfs2, eps, K, D)
    a2=time.clock()
    print 'time for ball tree is:', a2-a1
    
#    a3=time.clock()
#    TR = check_gradient2_energy(params, pdfs, eps, K, D)
#    a4=time.clock()
#    print 'time for main method', a4-a3
    

