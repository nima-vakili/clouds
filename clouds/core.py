"""
Defines the most fundamental classes 'Parameters' and 'Probability'.
Parameters stores all model parameters and allows subclasses of
Probability to share them. In principle, Parameters can be thought
of as a Singleton.
"""
import time
import contextlib
import numpy as np

from threading import Event
from thread import start_new_thread

from csb.bio.utils import distance_matrix
from csb.core import validatedproperty

from scipy.spatial import distance

from sklearn.neighbors import BallTree 

class DistanceMatrix(object):
    """DistanceMatrix

    Class for computing and storing *squared* distances
    """
    use_csb = not True
    
    def __init__(self):
        self._distances = None
        
    def __call__(self, x, mu, R):
        self.update(x, mu, R)
        return self._distances

    @property
    def shape(self):
        
        if self._distances is None:
            msg = 'Distances have not been computed yet'
            raise Exception(msg)

        return self._distances.shape
    
    @property
    def distances(self):

        if self._distances is None:
            msg = 'Distances have not been computed yet'
            raise Exception(msg)

        return self._distances

    @property
    def indices(self):

        indices = np.zeros(self.distances.shape, 'i')
        indices[...] = np.arange(indices.shape[1])

        return indices

    def update(self, x, mu, R):
        if self.use_csb:
            self._distances = distance_matrix(x, np.dot(mu, R[:2].T))**2
        else:
            self._distances = distance.cdist(x, np.dot(mu, R[:2].T), 'sqeuclidean')
            
class NearestNeighborDistanceMatrix(DistanceMatrix):

    def __init__(self, k):
        super(NearestNeighborDistanceMatrix, self).__init__()
        self._indices = None
        self.k = int(k)

    @property
    def indices(self):
        
        if self._indices is None:
            msg = 'Distances have not been computed yet'
            raise Exception(msg)

        return self._indices

    def update(self, x, mu, R):

        tree = BallTree(np.dot(mu, R[:2].T))
        
        self._distances, self._indices = tree.query(x, self.k, return_distance=True)
        self._distances **=2
        
        
def format_time(t):

    units = [(1.,'s'),(1e-3,'ms'),(1e-6,'us'),(1e-9,'ns')]
    for scale, unit in units:
        if t > scale or t==0: break
        
    return '{0:.1f} {1}'.format(t/scale, unit)

@contextlib.contextmanager
def take_time(desc, mute=False):
    t0 = time.clock()
    yield
    dt = time.clock() - t0
    if not mute:
        print '{0} took {1}'.format(desc, format_time(dt))

class Parameters(object):
    """Parameters

    Class holding all model parameters, data and hyper parameters.
    This class is shared among all probabilities to make sure that
    the probabilities always use the same parameters.
    """

    @property
    def n_components(self):
        return self.mu.shape[0]

    @property
    def n_points(self):
        return self.data.shape[1]

    @property
    def n_projections(self):
        return self.data.shape[0]

    @property
    def n_dimensions(self):
        return self.data.shape[2]

    @property
    def P(self):
        d = self.n_dimensions
        return np.eye(d+1)[:-1]

    @property
    def sigma(self):
        return 1 / self.tau**0.5

    @validatedproperty
    def beta(value):
        """
        Inverse temperature used for annealing and exchange simulations
        """
        value = float(value)
        if value < 0.:
            raise ValueError(value)
        return value
        
    def __init__(self, K=3, N=100, M=1, D=2, beta=1., n_neighbors=None):
        """
        Parameters
        ----------

        K : positive integer
          number of Gaussian components

        N :
          number of data points per image

        D : positive integer
          dimension of input data (default D=2)

        M : positive integer
          number of projections

        beta : positive float
          inverse temperature used for annealing
          
        """
        self.data = None
       
        K, N, M, D = map(int, [K, N, M, D])

        ## component means, precision, assignments, rotations

        self.mu  = np.random.random((K, D+1))
        self.tau = 1.
        self.Z   = np.random.randint(0, 2, (M, N, K if n_neighbors is None else n_neighbors))
        self.R   = np.array([np.eye(D+1) for _ in xrange(M)])
        self.n_neighbors = n_neighbors

        ## hyper parameters 

        self.mu_0    = np.zeros(D+1)
        self.tau_0   = 1e-1        
        self.alpha_0 = 1e-1
        self.beta_0  = 1e-1

        ## algorithmic parameters

        self.beta = beta
        
        self.distance = DistanceMatrix() if n_neighbors is None else NearestNeighborDistanceMatrix(n_neighbors)

    def get(self, *attrs):
        """
        Convenience method for retrieving a list of parameters
        """
        return [getattr(self,attr) for attr in attrs]

class Probability(object):
    """Probability

    Generic class that will be subclassed by all probabilistic
    models that are needed to describe projection data.
    """
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name) if name is not None else name

    def __init__(self, name=None, params=None):
        self.name = name
        self.params = params or Parameters()
    
    def log_prob(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

class MyResult(object):

    def __init__(self):

        object.__init__(self)

        self.__dict__['event'] = Event()
        self.__dict__['value'] = None

    def wait(self, timeout = None):

        self.event.wait(timeout)

        if timeout is not None:
            
            if not self.isSet():
                raise TimeoutError
        
        return self.value

    def __call__(self):
        return self.wait()

    def isSet(self):
        return self.event.isSet()

    def set(self):
        return self.event.set()

    def clear(self):
        return self.event.clear()
    
def _wrapper(f, result, *args, **kw):

    result.value = f(*args, **kw)
    result.set()

def threaded(f, *args, **kw):

    result = MyResult()
    
    start_new_thread(_wrapper, (f, result) + args, kw)
    
    return result

