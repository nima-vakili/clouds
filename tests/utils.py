"""
Collection of convenience functions used by various test scripts 
"""
import clouds
import numpy as np
import pylab as plt

def setup(pdbcode, n_components, n_neighbors=None, path='./data/{}.npz'):
    """
    Sets up the parameters and data for a 3d reconstruction task generated
    from a PDB file

    Parameters
    ----------

    pdbcode : string
      four-letter PDB code
      
    n_components : positive integer
      number of Gaussian components used to a approximate the reconstructed
      object
    """
    data = np.load(path.format(pdbcode))

    ## input data

    projections = data['projections']

    ## setup model

    params = clouds.Parameters(n_components, projections.shape[1],
                               projections.shape[0], projections.shape[2], n_neighbors=n_neighbors)
    params.data   = projections

    if 'coords' in data and len(data['coords']) == n_components:
        params.mu[...] = data['coords']
    
    if 'rotations' in data:
        params.R[...] = data['rotations']

    if 'assignments' in data:
        params.Z[...] = data['assignments']
        
    if 'precisions' in data:
        params.tau = data['precisions']

    ## set parameters globally in all subclasses of Probability

    ## clouds.Probability.set_params(params)

    return params

def show_projections(params, n_rows=5, n_cols=7, thin=10):
    """
    Visualize the observed and estimated projections
    """
    projections = params.data
    limits = projections.min(), projections.max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,2*n_rows),
                             subplot_kw=dict(xlim=limits, ylim=limits))

    for k, ax in enumerate(list(axes.flat)):

        ax.scatter(*projections[k][::thin].T, color='k', s=5, alpha=0.3)
        ax.scatter(*np.dot(params.mu, params.R[k].T)[:,:2].T,color='r',s=25,alpha=.7)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig, axes
