import clouds
import utils, utils2
import numpy as np

from clouds.marginal import RotationSampler, RotationMaximizer
from clouds.rotation import random_euler, Poses, Pose

from csb.numeric import euler, euler_angles

if __name__ == '__main__':

#    dataset = 'toy_tau=1_K=5'
    n_iter     = 100
    K          = 50
    dataset = ('1aon')
    params      = utils.setup(dataset, K)
    params.beta = 1.0
#    params  = utils.setup(dataset, 5)
    truth   = params.R.copy()
    
    L         = clouds.MarginalLikelihood(1, params=params)
    sampler   = RotationSampler(n_steps=100, stepsize=0.03)
    sampler.adapt_stepsize = False
    
    n_trials  = 10e1
    
    for index in range(params.n_projections):
        
        params = utils.setup(dataset, K)
#        params  = utils.setup(dataset, 5)
        params.beta = 1.0
        L = clouds.MarginalLikelihood(index, params=params)
        optimizer = RotationMaximizer(L, n_trials)  

        poses = optimizer.run(return_poses=True)
        poses.prune()

        angles = optimizer.optimize(euler_angles(truth[index]))

        norms = [clouds.frobenius(truth[index], pose.R) for pose in poses]

        print index, norms[0], min(norms), min(poses.score), optimizer(angles)

        params.R[index,...] = poses[0].R

        ## MCMC starting from best pose

        angles = sampler.run(L, angles=poses[0].params)
        pose   = Pose(angles, optimizer(angles))

        print clouds.frobenius(truth[index], pose.R)
        
        
#        np.savez('/home/nvakili/Documents/PYTHON_CODES/weekly_experiments/23.15.2017/marginalR/R', params.R)
#        np.savez('/home/nvakili/Documents/PYTHON_CODES/weekly_experiments/23.15.2017/marginalR/pose', pose)
        
    if  False:

        samples = []
        norms   = []

        for _ in xrange(100):

            angles = sampler.run(L, verbose=not True, angles=angles)
            samples.append(angles.copy())
            norms.append(clouds.frobenius(truth[L.projection_index], euler(*angles)))

            print norms[-1], sampler.stepsize

#        np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/marginalR/norms', norms)
#        np.savez('/home/nvakili/Documents/clouds/results_hmc_GroEl_index/marginalR/samples', samples)