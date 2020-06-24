import numpy as np

def frobenius(A,B):
    """
    Frobenius distance between two rotation matrices
    """
    return 1 - np.sum(A*B) / len(A)
                
from .core import Probability, Parameters
from .prior import PriorRotations, PriorMeans, PriorAssignments, PriorPrecision, PriorMeansVersion2
from .model import Likelihood, PosteriorRotations, PosteriorAssignments
from .model import PosteriorMeans, PosteriorPrecision, Posterior, PosteriorMeansVersion2, PosteriorPrecisionVersion2, PosteriorPrecisionVersion2_NoneZ, PosteriorPrecision_NoneZ

#from .model_fasterVersion_ballTree import FasterLikelihood, Posterior, FasterPosteriorPrecisionVersion2, FasterPosteriorAssignments, FasterPosteriorPrecision

from .gibbs import GibbsSampler
from .rotation import MatrixBingham, RotationFinder, RotationMaximizer
from .rotation import ProjectedBingham, ProjectedBinghamFast, RotationSampler, euler
from .rotation import ExponentialMap
from .marginal import MarginalLikelihood_R, MarginalLikelihood_mu, MarginalPosteriorRotations

#from .marginal_fasterVersion_ballTree import  FasterMarginalLikelihood_R, FasterMarginalLikelihood_mu
