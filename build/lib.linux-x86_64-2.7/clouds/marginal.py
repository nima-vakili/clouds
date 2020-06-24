"""
Marginal likelihood that is not using explicit assignment variables
"""
import numpy as np

from .model import Likelihood, Posterior
from .prior import PriorRotations
from .rotation import random_euler, Poses, Pose
from .core import take_time

from csb.numeric import log_sum_exp, euler, euler_angles
from scipy import optimize

class MarginalLikelihood_R(Likelihood):
    """MarginalLikelihood

    Marginal likelihood for a single projection. No assignment variables
    are used. 
    """
    def __init__(self, projection_index, name=None, params=None):

        super(MarginalLikelihood_R, self).__init__(name, params)

        self.projection_index = int(projection_index)

    @property
    def log_prob_components(self):
        """
        Evaluates log probabilities for all combinations of data points
        from i-th projection and all components
        """
        params = self.params

        i  = self.projection_index

        R  = params.R[i][:2]
        x  = params.data[i]
        mu = params.mu
        
        log_p = params.distance(x, mu, R)
        log_norm = 0.5 * params.n_dimensions * np.log(params.tau / (2*np.pi))
      
        return log_norm - 0.5 * params.tau * log_p

    def log_prob(self):

        log_p = self.log_prob_components
        log_p = log_sum_exp(log_p.T, 0)

        return self.params.beta * log_p.sum()

    def gradient(self):
        """
        Gradient with respect to rotation matrix
        """
        params = self.params 

        i  = self.projection_index
        mu = params.mu
        x  = params.data[i]
        R  = params.R[i][:2]
        D  = params.distance(x, mu, R)
       
        log_prob = -0.5 * params.tau * D + \
                    0.5 * params.n_dimensions * np.log(params.tau / (2*np.pi)) - \
                    np.log(params.n_components)

        log_norm = log_sum_exp(log_prob.T, 0)
        log_prob = (log_prob.T - log_norm).T

        prob = np.exp(log_prob)

        A = np.dot(x.T, np.dot(prob, mu))
        B = np.dot(np.dot(mu, R.T).T * prob.sum(0), mu)

        return params.beta * params.tau * np.dot(params.P.T, A - B) 
   
    
class MarginalLikelihood_mu(Likelihood):


    def __init__(self, name=None, params=None):

        super(MarginalLikelihood_mu, self).__init__(name, params)

    def log_prob(self):

        params = self.params
        mu = params.mu
        
        log_prob = 0.

        for i in range(params.n_projections):        

            R  = params.R[i,:2]
            x  = params.data[i]
            D  = params.distance(x, mu, R) ##np.dot(mu, R.T))**2
            
            log_p = 0.5 * (params.n_dimensions * np.log(params.tau / (2*np.pi)) -params.tau * D)
            log_prob += log_sum_exp(log_p.T, 0).sum()
            
        return self.params.beta * log_prob
    
    def gradient(self):
        """
        Gradient with respect to mean 
        """
        params = self.params 

        mu = params.mu
        E = 0.
        for i in range(params.n_projections):
            
            x  = params.data[i]
            R  = params.R[i,:2]
            D  = params.distance(x, mu, R)
            
            log_prob = -0.5 * params.tau * D + \
                        0.5 * params.n_dimensions * np.log(params.tau / (2*np.pi)) - \
                        np.log(params.n_components)

        
            log_norm = log_sum_exp(log_prob.T, 0)
            log_prob = (log_prob.T - log_norm).T

            prob = np.exp(log_prob)

            A = np.dot(x.T, prob)
            B = np.dot(mu, R.T).T * prob.sum(0)
            E += np.dot(R.T, B-A)
            
        return params.beta * params.tau * E.T


class RotationMaximizer(object):
    """
    For optimizing the rotational parameters
    """
    def __init__(self, marginal_likelihood, n_trials=20):
        self.marginal_likelihood = marginal_likelihood
        self.n_trials = int(n_trials)
        
    def __call__(self, angles):

        L = self.marginal_likelihood
        i = L.projection_index
        R = L.params.R[i].copy()

        L.params.R[i,...] = euler(*angles)
        log_p = L.log_prob()
        L.params.R[i,...] = R

        return -log_p

    def optimize(self, angles=None):

        if angles is None: angles = random_euler()

        return optimize.fmin(self, angles, disp=False)

    def run(self, angles=None, return_poses=False):

        poses = Poses(n_max=10)        

        for _ in xrange(self.n_trials):

            angles = self.optimize()
            energy = self(angles)

            poses.add(Pose(angles, energy))

        poses.prune()

        if return_poses:
            return poses
        else:
            return poses[0].params
        

class RotationSampler(object):
    """RotationSampler

    Monte Carlo sampler for a single rotation
    """
    def __init__(self, n_steps=10, stepsize=0.1):
        """
        Parameters:
        -----------

        n_steps  : number of Monte Carlo steps
        stepsize : stepsize used in the random walk proposal 
        """
        self.n_steps  = int(n_steps)
        self.stepsize = float(stepsize)

        self.adapt_stepsize = True

    def run(self, marginal_likelihood, verbose=False, samples=None, angles=None,
            optimize_first=False):
        """
        Run random walk Metropolis Monte Carlo to estimate the unknown
        rotation 
        """
        if angles is None: angles = np.array(random_euler())

        optimizer = RotationMaximizer(marginal_likelihood)

        if optimize_first: angles = optimizer.run(angles)

        log_prob = -optimizer(angles)
        n_accept = 0        

        for i in range(self.n_steps):

            ## propose new angles

            angles_new   = angles + self.stepsize * np.random.uniform(-1, 1., angles.shape)
            log_prob_new = -optimizer(angles_new)

            ## accept / reject new rotation according to Metropolis criterion

            accept = np.log(np.random.random()) < (log_prob_new - log_prob)

            if accept:
                angles, log_prob = angles_new, log_prob_new
                n_accept += 1

            if self.adapt_stepsize:

                self.stepsize *= 1.02 if accept else 0.98
                self.stepsize = min(1., self.stepsize)
            
            if samples is not None: samples.append(angles)

        if verbose:
            print 'acceptance rate:', float(n_accept) / self.n_steps

        return angles


class MarginalPosteriorRotations(Posterior, PriorRotations):

    """MarginalPosteriorRotations

    Sampler for all unknown projection directions. To estimate each projection's

    unknown rotation matrix, Monte Carlo samples are drawn according to the

    marginal likelihood. 

    """

    def __init__(self, params, likelihood, verbose=False):

        super(MarginalPosteriorRotations, self).__init__(params, likelihood)

        ## create marginal likelihoods for each projection direction

        self._likelihoods = [MarginalLikelihood_R(i, params=params)
                             for i in range(self.params.n_projections)]
        self._sampler = RotationSampler(n_steps=10, stepsize=1e-1)
        self.verbose  = verbose        

    def sample(self, verbose=False):

        likelihoods, sampler = self._likelihoods, self._sampler

        ## now loop over all input images and sample their orientations

        for i in range(self.params.n_projections):

            angles = np.array(euler_angles(self.params.R[i]))

            if self.verbose:

                print 'beta={0:.2e}'.format(self.params.beta)

            angles = sampler.run(likelihoods[i], angles=angles, verbose=self.verbose)

            self.params.R[i,...] = euler(*angles)
            
            
            
            
