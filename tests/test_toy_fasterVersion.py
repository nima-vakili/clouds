import clouds
import utils
import numpy as np

from clouds.marginal import RotationSampler, RotationMaximizer
from clouds.rotation import random_euler, Poses, Pose
from clouds.core import take_time
from csb.numeric import euler, euler_angles

if __name__ == '__main__':

    dataset ='toy_tau=1_K=5' #'dilutedData_1aon'
    K = 5.
    n_neighbors = 2.
    params1  = utils.setup(dataset, K)
    params2 = utils.setup(dataset, K)

    truth   = params1.R.copy()

    
    L1         = clouds.MarginalLikelihood_R(1, params=params1)
    L2         = clouds.FasterMarginalLikelihood_R(1, n_neighbors, params=params2)
    
    sampler   = RotationSampler(n_steps=10, stepsize=0.03)
    sampler.adapt_stepsize = False
    
    n_trials  = 10e1
    
    with take_time("new version"):
        
        for index in range(params1.n_projections):
            
            params1  = utils.setup(dataset, K)
            params2  = utils.setup(dataset, K)
            params1.beta = 1.0
            params2.beta = 1.0
            L1 = clouds.MarginalLikelihood_R(index, params=params1 )
            L2 = clouds.FasterMarginalLikelihood_R(index, n_neighbors, params2)
            optimizer_1 = RotationMaximizer(L1, n_trials)  
            optimizer_2 = RotationMaximizer(L2, n_trials)
            
            poses_1 = optimizer_1.run(return_poses=True)
            poses_2 = optimizer_2.run(return_poses=True)
            poses_1.prune()
            poses_2.prune()
    
            angles_1 = optimizer_1.optimize(euler_angles(truth[index]))
            angles_2 = optimizer_2.optimize(euler_angles(truth[index]))
    
            norms_1 = [clouds.frobenius(truth[index], pose.R) for pose in poses_1]
            norms_2 = [clouds.frobenius(truth[index], pose.R) for pose in poses_2]
            
            print index, norms_1[0], norms_2[0], min(norms_1), min(norms_2), min(poses_1.score)
            print min(poses_2.score), optimizer_1(angles_1), optimizer_2(angles_2)
    
            params1.R[index,...] = poses_1[0].R
            params2.R[index,...] = poses_2[0].R
                    
#            ## MCMC starting from best pose
#    
#            angles = sampler.run(L, angles=poses[0].params)
#            pose   = Pose(angles, optimizer(angles))
#    
#            print clouds.frobenius(truth[index], pose.R)
#        
