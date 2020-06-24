import numpy as np

from . import Probability
from .prior import PriorAssignments, PriorPrecision, PriorMeans, PriorRotations, PriorMeansVersion2
from .rotation import ProjectedBinghamFast as ProjectedBingham, RotationSampler, euler, euler_angles
from .gibbs import set_state

from csb.numeric import log_sum_exp

class Likelihood(Probability):
    """Likelihood

    A mixture model for the projected point clouds.
    """
    @property
    def log_prob_components(self):
        """
        Evaluates log probabilities for all combinations of data points and
        components
        """
        params   = self.params
        x, mu, R = [getattr(self.params, attr) for attr in ('data','mu','R')]

        log_p = np.array([params.distance(x[i], mu, R[i][:2]) for i in range(params.n_projections)])
             
        log_norm = 0.5 * params.n_dimensions * np.log(params.tau / (2*np.pi))
        
        return params.beta * (log_norm - 0.5 * params.tau * log_p)

    def log_prob(self):

        if self.params.Z is not None:
            return np.sum(self.params.Z * self.log_prob_components)

        else:
            ## complete likelihood accumulated from all marginal likelihoods
            
            beta = self.params.beta
            self.params.beta = 1.
            log_p = self.log_prob_components.reshape(-1,self.params.n_components)
            log_p = log_sum_exp(log_p.T, 0)
            self.params.beta = beta

            return beta * log_p.sum()
            
    def sample(self):        

        params = self.params
        D = params.n_dimensions + 1
        P = params.P
        sigma = params.sigma
        
        for i in range (params.n_projections) :

            R = np.dot(P, params.R[i])

            for n in range(params.n_points):
                k = params.Z[i,n].argmax()
                x = np.random.standard_normal(D) * sigma + params.mu[k]

                params.data[i,n,...] = np.dot(R,x)


class Posterior(Probability):

    @property
    def _Prior(self):
        return self.__class__.__bases__[-1]

    @property
    def likelihood(self):
        return self._likelihood

    def __init__(self, params, likelihood, *priors):
        self._Prior.__init__(self, params=params)
        
        self.check_params(likelihood, *priors)        
        
        self._likelihood = likelihood
        self._priors = list(priors)

#    def log_prob(self):
#        return self._Prior.log_prob(self) + self.likelihood.log_prob()
    def log_prob(self, x):
        
        set_state(self.likelihood.params, x)    
        return np.sum([p.log_prob() for p in [self.likelihood] + self._priors])       

    def check_params(self, *pdfs):
        
        for pdf in pdfs:
            if self.params != pdf.params:
                msg = 'Inconsistent parameters in posterior in likelihood/priors'
                raise ValueError(msg)


class PosteriorAssignments(Posterior, PriorAssignments):

    def sample(self):        

        ## evaluate component-wise probabilities

        log_prob = self.likelihood.log_prob_components - \
                   np.log(self.params.n_components)

        ## normalize probabilities
        
        log_norm = log_sum_exp(log_prob.T, 0)
        log_prob = (log_prob.T - log_norm).T

        prob = np.exp(log_prob)

        N = self.params.n_points
        M = self.params.n_projections

        Z = [[np.random.multinomial(1,prob[i,n]/prob[i,n].sum()) for n in xrange(N)] for i in xrange(M)]
            
#        self.params.Z[...] = [[np.random.multinomial(1,prob[i,n]/prob[i,n].sum()) for n in xrange(N)] 
#                      for i in xrange(M)]
        
        if self.params.Z is not None:
            self.params.Z[...] = Z
                         
        return np.array(Z) 
    
      
class PosteriorPrecision_NoneZ(Posterior, PriorPrecision):

    def sample(self):        
        
        ##### Gibbs sampler for generationg Assignments
       
        n_gibbs = 1
        posterior_Z= PosteriorAssignments(self.params, self.likelihood)
        
        for i in range(n_gibbs):
            Z = posterior_Z.sample()
            
        params = self.params    
        x, mu, R, d, k = params.get('data', 'mu', 'R', 'n_dimensions', 'n_components')

        n = params.n_points * params.n_projections
        
        beta  = 0.5 * np.sum([np.sum(Z[i] * params.distance(x[i], mu, R[i][:2]))
                              for i in range(params.n_projections)])
            
        beta *= params.beta
        beta += 0.5 * params.tau_0 * np.sum(((mu - params.mu_0)**2))
        beta += params.beta_0
        
        alpha = params.alpha_0 + 0.5 * (params.beta * n * d + k * (d+1))
        
        params.tau = np.random.gamma(alpha) / beta

        return params.tau   
##################################################################    
        
class PosteriorPrecision(Posterior, PriorPrecision):

    def sample(self):        

        params = self.params
        x, mu, Z, R, d, k = params.get('data', 'mu', 'Z', 'R', 'n_dimensions', 'n_components')

        n = params.n_points * params.n_projections
        
        beta  = 0.5 * np.sum([np.sum(Z[i] * params.distance(x[i], mu, R[i][:2]))
                              for i in range(params.n_projections)])
            
        beta *= params.beta
        beta += 0.5 * params.tau_0 * np.sum(((mu - params.mu_0)**2))
        beta += params.beta_0
        
        alpha = params.alpha_0 + 0.5 * (params.beta * n * d + k * (d+1))
        
        params.tau = np.random.gamma(alpha) / beta

        return params.tau       


class PosteriorPrecisionVersion2(Posterior, PriorPrecision):

    def sample(self):        

        params = self.params
        x, mu, Z, R, d, k = params.get('data', 'mu', 'Z', 'R', 'n_dimensions', 'n_components')

        n = params.n_points * params.n_projections
        R = R[:,:-1]
        
        beta  = 0.5 * np.sum([np.sum(Z[i] * distance_matrix(x[i], np.dot(mu,R[i].T))**2)
                              for i in range(params.n_projections)])
        beta *= params.beta
        beta += params.beta_0
        
        alpha = params.alpha_0 + 0.5 * (params.beta * n * d)
        
        params.tau = np.random.gamma(alpha) / beta
        return params.tau   
    

class PosteriorPrecisionVersion2_NoneZ(Posterior, PriorPrecision):

    def sample(self):        
        
        ##### Gibbs sampler for generationg Assignments
   
        params = self.params
        n_gibbs = 1
        posterior_Z= PosteriorAssignments(params, self.likelihood)
        
        for i in range(n_gibbs):
            Z = posterior_Z.sample()
            
        x, mu, R, d, k = params.get('data', 'mu', 'R', 'n_dimensions', 'n_components')

        n = params.n_points * params.n_projections
        
        beta  = 0.5 * np.sum([np.sum(Z[i] * params.distance(x[i], mu, R[i][:2]))
                              for i in range(params.n_projections)])
        beta *= params.beta
        beta += params.beta_0
        
        alpha = params.alpha_0 + 0.5 * (params.beta * n * d)
        
        params.tau = np.random.gamma(alpha) / beta
        return params.tau                            

class PosteriorMeans(Posterior, PriorMeans):

    def sample(self):        

        params  = self.params        
        x, Z, R = params.get('data', 'Z', 'R')

        N = Z.sum(1)
        D = params.n_dimensions + 1
        R = R[:,:-1]
        
        A = np.array([np.dot(RR.T,RR) for RR in R]).reshape(-1,D**2)
        A = np.dot(N.T,A).reshape(-1,D,D) * params.beta
        A+= params.tau_0 * np.eye(D)
        
        ## calculate back projections

        x = np.array([np.dot(x[i], R[i]) for i in range(params.n_projections)]).reshape(-1,D)
        Z = Z.reshape(-1,params.n_components)
        b = np.dot(Z.T,x) * params.beta
        b+= params.tau_0 * params.mu_0

        for k in range(params.n_components):    

            Sigma = np.linalg.inv(A[k]) 
            mu = np.dot(Sigma, b[k])
            
            params.mu[k,...] = np.random.multivariate_normal(mu, Sigma / params.tau)
            
            
class PosteriorMeansVersion2(Posterior, PriorMeansVersion2):

    def sample(self):        

        params  = self.params        
        x, Z, R = params.get('data', 'Z', 'R')

        N = Z.sum(1)
        D = params.n_dimensions + 1
        R = R[:,:-1]
        
        A = np.array([np.dot(RR.T,RR) for RR in R]).reshape(-1,D**2)
        A = np.dot(N.T,A).reshape(-1,D,D) * params.beta * params.tau
        A+= params.tau_0 * np.eye(D)
        
        ## calculate back projections

        x = np.array([np.dot(x[i], R[i]) for i in range(params.n_projections)]).reshape(-1,D)
        Z = Z.reshape(-1,params.n_components)
        b = np.dot(Z.T,x) * params.beta * params.tau
        b+= params.tau_0 * params.mu_0

        for k in range(params.n_components):    

            Sigma = np.linalg.inv(A[k]) 
            mu = np.dot(Sigma, b[k])
            
            params.mu[k,...] = np.random.multivariate_normal(mu, Sigma)
                     
            
class PosteriorMeans2(Posterior, PriorMeans):
    """PosteriorMeans2

    Slower(?) version of PosteriorMeans
    """
    def sample(self):        

        params  = self.params        
        x, Z, R = params.get('data', 'Z', 'R')

        N = Z.sum(1)
        D = params.n_dimensions + 1
        R = R[:,:-1]
        
        for k in range(params.n_components):    

            ## contribution from prior

            A = params.tau_0 * np.eye(D)
            b = params.tau_0 * params.mu_0 

            for i in range(params.n_projections):
                
                A += N[i,k] * np.dot(R[i].T, R[i])
                b += np.dot(R[i].T, np.dot(Z[i,:,k], x[i]))           

            Sigma = np.linalg.inv(A) 
            mu = np.dot(Sigma, b)                            

            params.mu[k,...] = np.random.multivariate_normal(mu, Sigma / params.tau)
            
               
class PosteriorRotations(Posterior, PriorRotations):

    def __init__(self, params, likelihood, verbose=False):

        super(PosteriorRotations, self).__init__(params, likelihood)

        ## statistics of the conditional posterior distributions
        ## of all rotation matrices

        self.A = self.params.R[:,:-1] * 0.
        self.B = self.params.R * 0.        

        ## helper classes for sampling rotations

        self._sampler = RotationSampler(n_steps=10, stepsize=1e-2)
        self._bingham = ProjectedBingham(self.A[0], self.B[0])
        self.verbose  = verbose
        
    def update_statistics(self):
        """
        Updates A_i and B_i matrices
        """
        params = self.params
        x, mu, Z, P = params.get('data','mu','Z','P')

        N = Z.sum(1)
        D = params.n_dimensions + 1
        M = np.array([np.multiply.outer(mu[k],mu[k]) for k in range(params.n_components)]).reshape(-1,D*D)

        self.B[...] = np.dot(N,M).reshape(-1,D,D)
        self.B *= -0.5 * params.tau * params.beta
        
        for i in range (params.n_projections):  
            self.A[i,...] = params.tau * np.dot(np.dot(x[i].T, Z[i]), mu) * params.beta

    def sample(self, verbose=False):

        bingham, sampler = self._bingham, self._sampler

        ## update A_i and B_i
        
        self.update_statistics()

        ## now loop over all input images and sample their orientations

        for i in range(self.params.n_projections):

            bingham.A, bingham.B = self.A[i], self.B[i]

            angles = np.array(euler_angles(self.params.R[i]))
            if self.verbose:
                print 'beta={0:.2e}'.format(self.params.beta)
            angles = sampler.run(bingham, angles=angles, verbose=self.verbose, beta=self.params.beta)
            
            self.params.R[i,...] = euler(*angles)
     
