"""
Prior probabilities
"""
import numpy as np

from . import Probability
from .rotation import random_rotation

from scipy.special import gammaln

class PriorMeans(Probability):
    """PriorMeans

    Probabilistic model implementing a Gaussian prior for the component
    means. All means are generated from the same Gaussian centered at
    mu_0 with inverse variance tau * tau_0.
    """
    def __init__(self, name=r'Pr({mu_k}|tau,mu_0,tau_0)', params=None):
        super(PriorMeans, self).__init__(name, params)
        
    def log_prob(self):

        params = self.params

        mu, tau, mu_0, tau_0, K, D = [getattr(params, attr) for attr in
                                      ('mu','tau','mu_0','tau_0',
                                       'n_components', 'n_dimensions')]

        log_norm = 0.5 * K * (D+1) * np.log(tau_0 * tau / (2 * np.pi))
        
        return log_norm - 0.5 * tau * tau_0 * np.sum((mu - mu_0)**2)

    def sample(self):

        params = self.params
        shape  = (params.n_components, params.n_dimensions + 1)
        var    = 1 / (params.tau_0 * params.tau)
        
        params.mu[...] = np.random.standard_normal(shape) * var**0.5 + params.mu_0  
                 
    
    def gradient(self):
        
        params = self.params 
        
        return params.tau * params.tau_0 * (params.mu-params.mu_0)
                 
                 
class PriorMeansVersion2(PriorMeans):

    def log_prob(self):

        params = self.params

        mu, mu_0, tau_0, K, D = [getattr(params, attr) for attr in
                                      ('mu','mu_0','tau_0',
                                       'n_components', 'n_dimensions')]

        log_norm = 0.5 * K * (D+1) * np.log(tau_0 / (2 * np.pi))
        
        return log_norm - 0.5 * tau_0 * np.sum((mu - mu_0)**2)

    def sample(self):

        params = self.params
        shape  = (params.n_components, params.n_dimensions + 1)
        var    = 1 / (params.tau_0)
        
        params.mu[...] = np.random.standard_normal(shape) * var**0.5 + params.mu_0  
                 
    def gradient(self):
        
        params = self.params 
        
        return params.tau_0 * (params.mu-params.mu_0)
                
                 
class PriorPrecision(Probability):
    """PriorPrecision

    Gamma distribution for the precision parameter tau. The parameters of
    the Gamma distribution are 'alpha_0' and 'beta_0'
    """
    def __init__(self, name=r'Pr(tau|alpha_0,beta_0)', params=None):
        super(PriorPrecision, self).__init__(name, params)
        
    def log_prob(self):

        params = self.params

        tau, a, b = [getattr(params, attr) for attr in ('tau','alpha_0','beta_0')]
        log_norm  = a * np.log(b) - gammaln(a)
        
        return log_norm + (a-1) * np.log(tau) - b * tau

    def sample(self):

        params = self.params
        params.tau = np.random.gamma(params.alpha_0) / params.beta_0
        

class PriorAssignments(Probability):
    """PriorAssignments

    Categorical distribution for the binary assignment variables Z_{mnk} where
    index m enumerates projection directions, index n enumerates data points per
    projetion image, and index k enumerates the number of components.

    A priori all components are equally likely, i.e. the vectors Z_{mn} follow a
    multinomial distribution with probabilities 1 / K (where K is the number of
    components).
    """
    def __init__(self, name='Pr(Z_{mnk})', params=None):
        super(PriorAssignments, self).__init__(name, params)
        
    def log_prob(self):
        params = self.params
        return - np.log(params.n_components) * params.n_points * params.n_projections
        return - np.sum(self.params.Z * np.log(self.params.n_components))

    def sample(self):

        params = self.params
        prob   = np.ones(params.n_components if params.n_neighbors is None else params.n_neighbors)
        prob  /= prob.sum()
        
        for m in range(params.n_projections):

            params.Z[m,...] = np.array([np.random.multinomial(1, prob)
                                         for _ in range(params.n_points)])
            
class PriorRotations(Probability):
    """PriorRotations

    Uniform prior for the rotation matrices.
    """
    def __init__(self, name='Pr({R_m})', params=None):
        super(PriorRotations, self).__init__(name, params)
        
    def log_prob(self):
        return 0.

    def sample(self):

        d = self.params.n_dimensions

        if d == 1:

            theta = np.random.random(self.params.n_projections) * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
                
            self.params.R[...] = np.rollaxis(np.array([[c,-s],[s,c]]),2)

        elif d == 2:

            self.params.R[...] = random_rotation(self.params.n_projections)

        else:
            msg = 'Only one- and two-dimensional data supported'
            raise ValueError(msg)

