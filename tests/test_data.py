"""
Generating synthetic data from PDB entries
"""
from clouds.data import projection_series
import pylab as plt

## GroEL/ES

pdbcode = '1aon' 

n_rows, n_cols = 5, 5
n_projections  = n_rows * n_cols 

coords, projections, rotations = projection_series(
    pdbcode, n_projections, center_coords=True, atoms=['CA'])

## visualize the projections

limits = projections.min(), projections.max()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols,3*n_rows),
                     subplot_kw=dict(xlim=limits, ylim=limits))

for k, ax in enumerate(list(axes.flat)):

    ax.scatter(*projections[k][::10].T, color='k', s=5, alpha=0.3)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

## save data

if False:

    filename = './data/{}'.format(pdbcode)
    np.savez(filename, coords=coords, projections=projections, rotations=rotations)
