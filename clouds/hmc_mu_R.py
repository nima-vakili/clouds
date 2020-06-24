"""
HMC for sampling rotations in clouds
"""
import numpy as np
from clouds.model_fasterVersion_ballTree import Posterior
from .prior import PriorRotations, PriorMeans, PriorMeansVersion2
from .rotation import random_euler, grad_euler, Poses, Pose
from .rotation import ExponentialMap
from .marginal_fasterVersion_ballTree import  RotationMaximizer,FasterMarginalLikelihood_R, FasterMarginalLikelihood_mu
from csb.numeric import euler, euler_angles

from scipy import optimize


class HarmonicPotential(object):

    def energy(self, x):
        return np.sum(x**2) / 2.

    def gradient(self, x):
        return x


class EulerEnergy(FasterMarginalLikelihood_R):

    n_energy_evaluations = 0
    n_gradient_evaluations = 0
    
    def set_params(self, angles):
        self.params.R[self.projection_index, ...] = euler(*angles)
            
    def gradient(self, angles):
        
        self.set_params(angles)
        grad = super(EulerEnergy, self).gradient().flatten()

        self.__class__.n_gradient_evaluations += 1

        return - np.dot(grad_euler(*angles).reshape(3,-1), grad)

    def energy(self, angles):
        self.set_params(angles)
        self.__class__.n_energy_evaluations += 1
        return - self.log_prob()


class MeanEnergy(object):
    
    def __init__(self, params, pdfs):
        self.params = params
        self.pdfs = pdfs
    
    def gradient(self, mu):
        mu_old = self.params.mu.copy()
        self.params.mu[...] = mu              
        grad = np.sum([pdf.gradient() for pdf in self.pdfs], 0)
        self.params.mu[...] = mu_old
        return grad    
    
    def energy(self, mu):
        mu_old = self.params.mu.copy()
        self.params.mu[...] = mu
        E = - np.sum([pdf.log_prob() for pdf in self.pdfs], 0)
        self.params.mu[...] = mu_old
        return E


class AxisAngleEnergy(EulerEnergy):

    def __init__(self, *args, **kw):
        super(AxisAngleEnergy, self).__init__(*args, **kw)
        self._mapping = ExponentialMap(self.params.R[self.projection_index])

    def set_params(self, params):
        self._mapping.params[...] = params
        self.params.R[self.projection_index,...] = self._mapping.rotation

    def gradient(self, params):
        
        self.set_params(params)
        grad = super(EulerEnergy, self).gradient().flatten()

        return - np.dot(self._mapping.gradient().reshape(3,-1), grad)
    
    
class RotationSampler(object):

    def __init__(self, n_steps=10, stepsize=0.1):

        self.n_steps  = int(n_steps)
        self.stepsize = float(stepsize)
        self.adapt_stepsize = True
        
    def run(self, energy, verbose=False, samples=None, angles=None,
            optimize_first=False):

        if angles is None: angles = np.array(random_euler())

        optimizer = RotationMaximizer(energy)

        if optimize_first: angles = optimizer.run(angles)

        hmc = HMC(energy, self.stepsize)
        hmc.adapt_stepsize = self.adapt_stepsize
       
#        for i in range(self.n_steps):   
    
        states = hmc.run(angles, self.n_steps, verbose=False)

        if samples is not None:
            samples += states 

        angles = states[-1]
          
        self.stepsize = hmc.stepsize
 
        return angles
    
    
class MeanSampler(object):

    def __init__(self, n_steps=10, stepsize=0.1):

        self.n_steps  = int(n_steps)
        self.stepsize = float(stepsize)
        self.adapt_stepsize = True
        
    def run(self, energy, samples):
        
        hmc = HMC(energy, self.stepsize)
        hmc.adapt_stepsize = self.adapt_stepsize
    
        states = hmc.run(samples, self.n_steps, verbose=False)
        samples = states[-1]
        self.stepsize = hmc.stepsize
        
        return samples
    
    
class HMC(object):
    
    def __init__(self, system, stepsize, n_leaps=10):

        self.potential = system
        self.kinetic   = HarmonicPotential()   
        self.stepsize  = float(stepsize)
        self.n_leaps   = int(n_leaps)
        self.adapt_stepsize = False

    def propose(self, (q0, p0)):

        eps = self.stepsize
        p = p0 - eps * self.potential.gradient(q0) / 2.
        q = q0 + eps * self.kinetic.gradient(p)
            
        for i in range(self.n_leaps-1):
            
            p -= eps * self.potential.gradient(q)
            q += eps * self.kinetic.gradient(p)
                        
        p -= eps * self.potential.gradient(q) / 2.
                 
        return q, p           

    def hamiltonian(self, (q,p)):
        
        return self.kinetic.energy(p) + self.potential.energy(q) 
              
    def accept(self, x_current, x_proposed):
     
        dH  = self.hamiltonian(x_current) - self.hamiltonian(x_proposed)
        
        return np.log(np.random.random()) < dH
    
    def run(self, q, n_samples=2, verbose=True):

        n_acc =  0.
        samples = [q.copy()]
   
        while len(samples) < n_samples:
            
            p = np.random.standard_normal(q.shape)            
            x = self.propose((q,p))  
            accept = self.accept((q,p), x)
            
            if accept:
                q = x[0]
                n_acc += 1

            if self.adapt_stepsize:
                self.stepsize *= 1.02 if accept else 0.98
                
            if samples is not None: samples.append(q.copy())
        
        if verbose: print n_acc / n_samples
    
        return samples
    
    
class MarginalPosteriorRotations_hmc(Posterior, PriorRotations):

    def __init__(self, params, likelihood, verbose=False):

        super(MarginalPosteriorRotations_hmc, self).__init__(params, likelihood)

        self._likelihoods = [EulerEnergy(i, params=params)
                             for i in range(self.params.n_projections)]
        self._sampler = RotationSampler(n_steps=10, stepsize=1e-1)
        self.verbose  = verbose
        self.optimize = False
        
    def sample(self, verbose=False):

        likelihoods, sampler = self._likelihoods, self._sampler

        for i in range(self.params.n_projections):

            angles = np.array(euler_angles(self.params.R[i]))
            if self.verbose:
                print 'beta={0:.2e}'.format(self.params.beta)

            angles = sampler.run(likelihoods[i], angles=angles, verbose=self.verbose,
                                 optimize_first=self.optimize)
            
            self.params.R[i,...] = euler(*angles)
        

class MarginalPosteriorMeans_hmc(Posterior, PriorMeans):

    def __init__(self, params, likelihood):

        super(MarginalPosteriorMeans_hmc, self).__init__(params, likelihood)
    
        pdfs = [FasterMarginalLikelihood_mu(params=params), PriorMeans(params=params)]
        self._likelihoods = MeanEnergy(params, pdfs)
        self._sampler = MeanSampler(n_steps=10, stepsize=1e-1)
        
    def sample(self):
        
        likelihoods, sampler = self._likelihoods, self._sampler
        samples = self.params.mu
        samples = sampler.run(likelihoods, samples=samples)
        
        self.params.mu[...] = samples
      

