# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:48:09 2019

@author: nima
"""

from csb.statistics.rand import random_rotation  
import clouds
from clouds.data import projection_series
import pylab as plt
import os
import numpy as np
import pointmatching as pm
from pointmatching.cloud import show
from scipy.ndimage.measurements import sum as partial_sum
from csb.bio.io import mrc
from csb.bio.io import StructureParser
from csb.numeric import log_sum_exp    
from scipy.spatial import distance    

class EM(object):
   
    @property
    def n_points(self):
        return len(self.coords)

    def __init__(self, target, n_points, shrink=0.8):

        lower, upper = target.bounding_box
        
        length = shrink * (upper - lower) / 2.
        lower  = target.center - length
        upper  = target.center + length
        
        self.target  = target
        self.coords  = np.random.uniform(lower, upper, (n_points,3))
        self.weights = np.ones(n_points)
        self.weights/= self.weights.sum()
        self.sigma   = 1.
        self.update_labels()        
        
    def update_labels(self):
        self.labels  = distance.cdist(self.target.coords, self.coords).argmin(1)

    def log_prob(self, normed=True, return_distances=False):

        d = distance.cdist(self.target.coords, self.coords, 'sqeuclidean')
        p = -0.5 * d / self.sigma**2 + np.log(self.weights) - np.log(2*np.pi*self.sigma**2)
        if normed:
            p = (p.T - log_sum_exp(p.T,0)).T

        if not return_distances:
            return p
        else:
            return p, d

    @property
    def log_likelihood(self):
        return np.dot(self.target.weights, log_sum_exp(self.log_prob(normed=False).T,0))

    def next(self):

        p, d = self.log_prob(normed=True, return_distances=True)
        p = np.exp(p)
        
#        self.weights = np.dot(self.target.weights, p) / self.target.weights.sum()
        self.coords  = np.dot(self.target.coords.T * self.target.weights, p) / np.dot(self.target.weights,p)
        self.coords  = self.coords.T
        self.sigma   = np.sqrt(np.sum(np.dot(self.target.weights, p * d)) / np.sum(self.target.weights) / self.coords.shape[1])

    def run(self, n_iter=100, tol=1e-10):

        self.L = []

        def relerror(a,b):
            return abs(a-b) / (abs(a) + abs(b))

        for i in xrange(int(n_iter)):

            next(self)
            self.L.append(self.log_likelihood)
            print i, self.L[-1]

            if len(self.L) >= 2 and relerror(*self.L[-2:]) < tol:
                break
            

def random_projection(coords):
    
    d = coords.shape[1]
    R = random_rotation(np.zeros((d,d)))
    return np.dot(coords, R.T)[:,:-1], R
    
def dp(coords, radius, i_iter):

    dpmeans   = pm.FastDPMeans(coords, 2 * radius)
    dpmeans.run(n_iter=1000, verbose=1)
    weights   = partial_sum(dpmeans.weights, dpmeans.labels, index=np.arange(dpmeans.n_clusters))
    cloud     = pm.PointCloud(dpmeans.centers, weights)
    return cloud

def em(cloud, n_atoms, n_iter):
    
    em = EM(cloud, n_atoms)
    em.coords[...] = cloud.coords[np.random.permutation(len(cloud))[:n_atoms]]
    em.update_labels()
    em.run(n_iter)
    cloud = pm.PointCloud(em.coords, em.weights)
    return cloud, em.sigma

def sample_data(coords, rotation, sigma, n_data):
    
    mu = np.dot(coords, rotation.T)[:,:-1]
    prob = np.ones(len(coords))
    prob/= prob.sum()
    indices = np.random.multinomial(1,prob,size=n_data).argmax(1)
    
    return np.random.standard_normal((n_data,mu.shape[1])) * sigma + mu[indices]
    
    
if __name__=='__main__':       

    try:
        from csbplus.bio.structure import BeadsOnStringViewer as Viewer
        pymol = Viewer()
    except:
        from csbplus.bio.structure import Viewer
        pymol = Viewer('pymol')
        
    name      = '1vor'#,'1i3q'#'1aon'
    radius    = 2.5
    struct    = StructureParser('1vor.pdb'.format(name)).parse()
    coords    = struct.get_coordinates()
    n_atoms   = 50#100 #200
    n_iter    = 1000
    n_data    = 1000
    
    if True: coords -= coords.mean(0)
    cloud = pm.PointCloud(coords[::10])
    em_cloud, sigma = em(cloud, n_atoms, n_iter)
    em_cloud.coords -= em_cloud.center
    raise
    n_projections = 35
    projections = []
    rotations = []
    
    for _ in range(n_projections):
    
        projection, rotation = random_projection(em_cloud.coords)
        projection = sample_data(em_cloud.coords, rotation, sigma, n_data)
        projections.append(projection)
        rotations.append(rotation)
    
    params = clouds.Parameters()  
    params.data = np.array(projections)
    params.mu = em_cloud.coords
    params.tau = 1/sigma**2
    params.R = np.array(rotations)
 
if False:   
    np.savez('../../clouds2/tests/data/dilutedData1vor_newMethod_50.npz', 
             projections=params.data,
             rotations=params.R,
             precisions=params.tau,
             coords=params.mu)    
    
#

#if False:
#    name = os.path.basename(mrcfile).split('.')[0]
#    np.savez('./data/{0}_{1}A.npz'.format(name, radius), coords=cloud.coords, weights=cloud.weights)
#    np.savez('./data/{0}_{1}A.npz'.format(name, radius), coords=cloud.coords, weights=cloud.weights)
#sampler.stop=1

if False:
    
    n_rows=5
    n_cols=7
    
    projections = np.array(projections)
    rotations = np.array(rotations)
    limits = projections.min(), projections.max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,2*n_rows),
                             subplot_kw=dict(xlim=limits, ylim=limits))

    for k, ax in enumerate(list(axes.flat)):

        ax.scatter(*projections[k].T, color='b', s=5, alpha=0.9)
        ax.scatter(*np.dot(em_cloud.coords, rotations[k].T)[:,:2].T,color='r',s=2,alpha=.7)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


    
    
    
    
    