import clouds
import utils
import numpy as np
from clouds.marginal import RotationSampler, RotationMaximizer
from clouds.rotation import random_euler, Poses, Pose
from csb.numeric import euler, euler_angles
import pylab as plt
from clouds import hmc, frobenius
import time
            
def compute_norms(samples, truth):
    
    return np.array([frobenius(truth, euler(*angles)) 
                     for angles in samples])
    

if __name__ == '__main__':
    

    dataset = 'toy_tau=1_K=5'
    params  = utils.setup(dataset, 5)
    truth   = params.R.copy()
    params.beta = 1. #0.1
    L1         = clouds.MarginalLikelihood(1, params=params)
    L2         = hmc.EulerEnergy(1, params=params)
    
    start  = np.array(random_euler())
    
    n_samples = 1000
    n_trials  = 1000000/n_samples
    
    sampler1   = RotationSampler(n_samples, 1e-2)
    sampler2   = hmc.RotationSampler(n_samples/10, 1e-2)
    
    sampler1.adapt_stepsize = not False
    sampler2.adapt_stepsize = not False

    T1 = []
    T2 = []
    t3 =  0.
    t33 = 0.
    
    log_prob_randomWalk = []
    log_prob_hmc = []
    angles   = []
    samples1 = {}
    samples2 = {}
 
    for _ in xrange(int(n_trials)):

        angles.append(np.array(random_euler()))
        
    for index in range(params.n_projections)[:]:

        L1.projection_index = index
        L2.projection_index = index
        
        true_angles = np.array(euler_angles(truth[index]))
              
        samples1[index] = []
        samples2[index] = []

        t1 = time.clock()
        
        for angle in angles:
            samples = []
            sampler1.run(L1, angles=angle.copy(), optimize_first=False, samples=samples)
            samples1[index].append(np.array(samples))
            
        t1 = time.clock() - t1
        T1.append(t1)
        print "RandomWallk time:", t1
        
        log_prob_randomWalk.append(L1.log_prob())
        
        t2 = time.clock()

        for angle in angles:
            samples = []
            sampler2.run(L2, angles=angle.copy(), optimize_first=False, samples=samples)
            samples2[index].append(np.array(samples))
            
        t2 = time.clock() - t2
        T2.append(t2)
        print "HMC time:", t2
        print"------------------------------------------------------"
        log_prob_hmc.append(L2.log_prob())
 
    ## plot norms for all projections
    
    kw = dict(normed=True,alpha=0.7,color='k',bins=30,histtype='stepfilled')

    fig, ax = plt.subplots(3,3,figsize=(12,12))
    ax = ax.flat
    for k, n in enumerate(samples1.keys()):
        if k > 8: break
        angles = np.array(samples1[n])[:,-1]
        norms1 = compute_norms(angles, truth[n])
        angles = np.array(samples2[n])[:,-1]
        norms2 = compute_norms(angles, truth[n])
        kw['color'] = 'k'
        ax[k].hist(norms1, label='RWMC', **kw)
        kw['color'] = 'r'
        ax[k].hist(norms2, label='HMC', **kw)
        ax[k].set_xlim(0.,1.5)
        ax[k].set_title('projection {0}'.format(n))
        ax[k].legend()
            
    threshold = 1e-1
        
    for n in samples1:
        angles = np.array(samples1[n])[:,-1]
        norms1 = compute_norms(angles, truth[n])
        angles = np.array(samples2[n])[:,-1]
        norms2 = compute_norms(angles, truth[n])
        print n, np.mean(norms1<threshold), np.mean(norms2<threshold)
        
    E1 = []
    E2 = []
    for n in samples1:
        L2.projection_index = n
        angles = np.reshape(samples1[n], (n_trials*n_samples,3))
        E1.append(np.reshape(map(L2.energy, angles), (n_trials,n_samples)))
    
        angles = np.reshape(samples2[n], (n_trials*n_samples/10,3))
        E2.append(np.reshape(map(L2.energy, angles), (n_trials,n_samples/10)))
    
    E1 = np.array(E1)
    E2 = np.array(E2)
    
    fig, ax = plt.subplots(3,3,figsize=(12,12))
    ax = ax.flat
    for k in range(len(E1)):
        if k > 8: break
        kw['color'] = 'k'
        ax[k].hist(E1[k,:,-1], label='RWMC',**kw)     
        kw['color'] = 'r'
        ax[k].hist(E2[k,:,-1], label='HMC' ,**kw)
        ax[k].legend()
        ax[k].yaxis.set_visible(False)
    fig.tight_layout()
  
    ## use more samples to compute the auto-correlation
    
    from csb.statistics import autocorrelation
    
    tau = 50
    k = 2 
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    for x in np.array(samples1[0])[:,:,k]:

        ax[0].plot(autocorrelation(x[::10],tau), color='k', lw=2, alpha=0.1)
    ax[0].plot(x.mean(0), lw=3,color='r')
    ax[0].axhline(0.,ls='--',color='r',lw=3)
   
    for x in np.array(samples2[0])[:,:,k]:

        ax[1].plot(autocorrelation(x,tau), color='k', lw=2, alpha=0.1)      
    ax[1].plot(x.mean(0), lw=3,color='r')
    ax[1].axhline(0.,ls='--',color='r',lw=3)
        
        
        
        
        
        
        
        
    