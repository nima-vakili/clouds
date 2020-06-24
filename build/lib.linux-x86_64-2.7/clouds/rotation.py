"""
Functions for optimizing and sampling rotations
"""
import numpy as np

from scipy import optimize
from csb.numeric import euler, euler_angles

def is_symmetric(A):
    return np.all(A==A.T)

def random_euler(n=None):
    """
    Generate random Euler angles
    """
    alpha = np.random.random(n) * 2 * np.pi
    gamma = np.random.random(n) * 2 * np.pi
    u     = np.random.uniform(-1.,1.,size=n)
    beta  = np.arccos(u)

    return alpha, beta, gamma

def random_rotation(n=None):
    """
    Generate random rotations
    """
    alpha, beta, gamma = random_euler(n)

    rotation = euler(alpha,beta,gamma)

    if n:
        return np.rollaxis(rotation,2)
    else:
        return rotation

def grad_euler(a, b, c):
    
    ca, cb, cc = map(np.cos, (a, b, c))
    sa, sb, sc = map(np.sin, (a, b, c))
    
    dR_a = np.array([[ -cc * cb * sa - sc * ca,  cc * cb * ca - sc *  sa, 0],
                     [  sc * cb * sa - cc * ca, -sc * cb * ca - cc * sa, 0],
                     [ -sb * sa , sb * ca , 0]])
    
    dR_b = np.array([[ -cc * sb * ca  , -cc * sb * sa , -cc * cb],
                     [  sc * sb * ca  ,  sc * sb * sa ,  sc * cb],
                     [  cb * ca  , cb * sa , -sb ]])
    
    dR_c = np.array([[ -sc * cb * ca - cc * sa , -sc * cb * sa + cc * ca, sc * sb],
                     [ -cc * cb * ca + sc * sa , -cc * cb * sa - sc * ca, cc * sb],
                     [     0, 0, 0 ]])
   
    return np.array([dR_a , dR_b , dR_c])
 
def skew_matrix(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])

class MatrixBingham(object):
    """MatrixBingham

    The matrix Bingham-Fisher-von Mises distribution

    R ~ etr(A^T R + B^T R^T C R)
    """
    def __init__(self, A, B, C):

        if not is_symmetric(B) or not is_symmetric(C):
            msg = 'Matrices B and C must be symmetric'
            raise ValueError(msg)

        self.A, self.B, self.C = A, B, C

    def log_prob(self, R):

        if R.ndim == 3:

            return np.array(map(self.log_prob, R))

        elif R.ndim == 2:

            return np.sum(self.A * R) + np.sum(self.B * np.dot(R.T, np.dot(self.C, R)))

        else:
            msg = 'Argument must be rotation matrix or array of rotation matrices'
            raise ValueError(msg)

class ProjectedBingham(MatrixBingham):
    """ProjectedBingham

    Special version of the matrix Bingham-Fisher-von Mises distribution
    that occurs in tomographic reconstruction problems. In this case, the
    matrices A and C have a particular structure.
    """
    def __init__(self, A, B):

        C = np.eye(len(A))
        C[-1,-1] = 0
        
        super(ProjectedBingham, self).__init__(A, B, C)

    def log_prob(self, R):

        if R.ndim == 3:

            return np.array(map(self.log_prob, R))

        elif R.ndim == 2:

            log_p = np.trace(self.B) - np.dot(R[-1],np.dot(self.B,R[-1]))
            log_p+= np.sum([np.dot(self.A[i],R[i]) for i in range(len(R)-1)])

            return log_p

        else:
            msg = 'Argument must be rotation matrix or array of rotation matrices'
            raise ValueError(msg)

class ProjectedBinghamFast(ProjectedBingham):
    """ProjectedBinghamFast

    Specialized class for 3d reconstruction problems. Uses a spectral
    decomposition of the B matrix to speed up computations.
    """
    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, B):

        self._B = B
        self._v, self._U = np.linalg.eigh(-B)
        self._A = np.dot(self.A, self._U)
        self._traceB = np.trace(B)
        
    def log_prob(self, R):

        if R.ndim == 2:
            x1, x2, x3 = np.dot(R, self._U)
        elif R.ndim == 3:
            x1, x2, x3 = np.dot(R, self._U).swapaxes(0,1)
        else:
            msg = 'Argument must be rotation matrix or array of rotation matrices'
            raise ValueError(msg)
        
        a1, a2 = self._A
        
        return np.dot(x1,a1) + np.dot(x2,a2) + self._traceB + np.dot(x3**2,self._v)
    
class RotationFinder(object):
    """RotationFinder

    Class for finding the rotation that maximizes the Matrix Bingham-
    Fisher-von Mises distribution. Picks the rotation that maximizes
    a BFvM distribution among a set of random rotation matrices
    parameterized by Euler angles.
    """
    def __init__(self, n_trials=1e4):

        self.n_trials = int(n_trials)

    def run(self, bingham):

        angles = np.transpose(random_euler(self.n_trials))
        random_rotations = np.rollaxis(euler(*angles.T),2)
        log_prob = bingham.log_prob(random_rotations)

        return angles[log_prob.argmax()]

class RotationMaximizer(RotationFinder):
    """RotationMaximizer

    Uses an optimizer to find the rotation achieving maximum
    probability.
    """
    optimizers = ('nedler-mead','powell','bfgs')

    def __init__(self, n_trials=1e4, optimizer='bfgs'):

        super(RotationMaximizer, self).__init__(n_trials)
        if not optimizer in RotationMaximizer.optimizers:
            msg = 'Optimizer "{}" not supported'.format(optimizer)
            raise ValueError(msg)

        self._optimizer = optimizer
    
    def run(self, bingham):

        ## first run a random search
        
        x = super(RotationMaximizer, self).run(bingham)

        ## then refine the result by running an optimizer
        
        f = lambda x , log_p=bingham.log_prob : -log_p(euler(*x))

        if self._optimizer == 'nedler-mead':
            return optimize.fmin(f, x, disp=False)

        elif self._optimizer == 'powell':
            return optimize.fmin_powell(f, x, disp=False)

        elif self._optimizer == 'bfgs':
            result = optimize.minimize(f, x, args=(), method='BFGS', jac=None, hess=None,
                                       hessp=None, tol=None)
            return result['x']
        
        else:
            msg = 'This should not happen!'
            raise Exception(msg)

class RotationSampler(RotationMaximizer):
    """RotationSampler

    Monte Carlo Sampler for the matrix Bingham distribution. The
    sampler uses Euler angles as parameters and starts from the
    most likely rotation.
    """
    def __init__(self, n_steps=1e2, stepsize=1e-1, n_trials=1e4):

        super(RotationSampler, self).__init__(n_trials)
        
        self.n_steps = int(n_steps)
        self.stepsize = float(stepsize)

    def run(self, bingham, verbose=False, samples=None, angles=None, beta=1.):
        """
        Rotations according to a matrix Bingham-Fisher-von Mises
        dsitribution using Metropolis Monte Carlo. 
        """         
        ## find initial rotations (Euler angles) for all projection images

        if angles is None:
            angles = super(RotationSampler, self).run(bingham)

        log_prob = bingham.log_prob(euler(*angles))
        n_accept = 0        

        for i in range(self.n_steps):

            angles_new = angles + self.stepsize * np.random.uniform(-1, 1., angles.shape)
            log_prob_new = bingham.log_prob(euler(*angles_new))

            accept = np.log(np.random.random()) < (log_prob_new - log_prob) * beta

            if accept:

                angles, log_prob = angles_new, log_prob_new
                n_accept += 1

            self.stepsize *= 1.02 if accept else 0.98
            self.stepsize  = min(1., self.stepsize)
            
            if samples is not None: samples.append(angles)

        if verbose:
            print 'acceptance rate:', float(n_accept) / self.n_steps

        return angles
        
class Pose(object):

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        if x.ndim == 2:
            self._params = np.array(euler_angles(x))
        else:
            self._params = np.array(x)

    @property
    def R(self):
        return euler(*self.params)

    def __init__(self, params, score):
        self.params = params
        self.score  = score

class Poses(object):

    def __init__(self, poses=None, n_max=None):
        self._items = [] if poses is None else list(poses)
        self.n_max  = n_max
        
    def sort(self):
        self._items.sort(lambda a, b: cmp(a.score, b.score))

    def add(self, pose):
        self._items.append(pose)

    def prune(self):
        if self.n_max is not None:
            self.sort()
            self._items = self._items[:self.n_max]

    @property
    def score(self):
        return np.array([pose.score for pose in self])

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

class ExponentialMap(object):
    """ExponentialMap

    Parameterization of rotations in terms of the exponential map. 
    """
    @classmethod
    def from_rotation(cls, R):
        a = np.arccos(np.clip(0.5 * (np.trace(R)-1), -1, 1))
        if a != 0:
            n = 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            return n / np.sin(a) * a
        else:
            return np.zeros(3)
        
    def __init__(self, params):
        if params.ndim == 2:
            params = ExponentialMap.from_rotation(params)
            
        self.params = np.array(params)

    @property
    def rotation(self):
        """
        Rotation matrix
        """
        n, a = self.axisangle
        
        return np.cos(a) * np.eye(3) + np.sin(a) * skew_matrix(n) + \
               (1-np.cos(a)) * np.multiply.outer(n,n)

    @property
    def axisangle(self):
        a = np.linalg.norm(self.params)
        n = self.params / a

        return n, a

    def rotate(self, v):
        """
        Rodrigues formula
        """
        n, a = self.axisangle

        return np.cos(a) * v + np.sin(a) * np.cross(n, v) + (1-np.cos(a)) * np.dot(n, v) * n

    def gradient(self):
        norm = np.linalg.norm(self.params)
        v = self.params

        A = skew_matrix(v)
        R = self.rotation
        B = np.cross(v, R - np.eye(3))

        return np.array([np.dot(v[i] * A + skew_matrix(B[:,i]), R)
                         for i in range(3)]) / (norm**2 + 1e-100)

