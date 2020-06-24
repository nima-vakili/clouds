"""
Estimate rotation matrix for a small synthetic problem
"""
import clouds
import numpy as np

from clouds.core import take_time
from clouds.rotation import random_rotation, euler, euler_angles 

from scipy import optimize

i = 30

data       = np.load('./data/1aon.npz')
mu         = data['coords']
projection = data['projections'][i]
rotation   = data['rotations'][i]

## add noise

sigma = 1.
projection += np.random.standard_normal(projection.shape) * sigma

B = -0.5 * np.dot(mu.T,mu)
A = np.dot(projection.T,mu)

bingham   = clouds.ProjectedBingham(A,B)
maximizer = clouds.RotationMaximizer()

print bingham.log_prob(rotation)

angles = maximizer.run(bingham)

print clouds.frobenius(rotation, euler(*angles))
print np.round(np.rad2deg(angles),1)
print np.round(np.rad2deg(euler_angles(rotation)),2)

## run Monte Carlo sampler

samples = []
sampler = clouds.RotationSampler(n_steps=1e4, stepsize=5e-4)
sampler.run(bingham, verbose=True, samples=samples)
samples = np.array(samples)

## show results

kw_hist = dict(bins=50,normed=True,histtype='stepfilled')

fig, axes = subplots(1,3,figsize=(20,6))
for k, ax in enumerate(axes.flat):
    ax.hist(samples[:,k],**kw_hist)
    ax.axvline(angles[k],ls='--',color='r')

bingham2 = clouds.ProjectedBinghamFast(bingham.A, bingham.B)

R = random_rotation(int(1e4))

with take_time(bingham.__class__.__name__):
    log_p  = bingham.log_prob(R)

with take_time(bingham2.__class__.__name__):
    log_p2 = bingham2.log_prob(R)

print 'Max diff between slow and fast version:', np.fabs(log_p - log_p2).max()
print

for p in (bingham,bingham2):
    with take_time('Maximizing ' + p.__class__.__name__):
        angles = maximizer.run(p)
    print 'achieved log prob: {0:.2e}'.format(p.log_prob(euler(*angles)))
    print

def target(x, p=bingham2):

    x1,x2,x3 = euler(*x)
    a1,a2 = p._A
    
    return -(np.dot(a1,x1)+np.dot(a2,x2)+np.dot(p._v,x3**2))

def target2(x, p=bingham2):

    return -bingham2.log_prob(euler(*x))

for f in (target, target2):
    with take_time(f.__name__):
        x = optimize.fmin(f, np.zeros(3), disp=False)
        R = euler(*x)
        if f is target: R = np.dot(R, bingham2._U.T)
    print 'achieved log prob: {0:.2e}'.format(bingham2.log_prob(R))
    print 

