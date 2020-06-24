import utils
import clouds
import numpy as np

from csbplus.bio.structure import Viewer

pymol = Viewer('pymol')

from clouds import prior

## setup model

K = 50
params = utils.setup('1aon',K)

prior_mu  = prior.PriorMeans()
prior_tau = prior.PriorPrecision() 
prior_Z   = prior.PriorAssignments()
prior_R   = prior.PriorRotations()

priors = (prior_mu, prior_tau, prior_Z)

for prior in priors:
    prior.sample()

for prior in priors:
    print '{0:25s} : {1:8.2e}'.format(str(prior), prior.log_prob())

## some tests

print 'Some tests'
print 'Do Z_{mnk} satisfy constraints? - ', np.all(params.Z.sum(-1)==1)

