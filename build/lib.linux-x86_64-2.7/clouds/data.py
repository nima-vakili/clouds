"""
Some utility functions for creating synthetic data.
"""
import numpy as np

from csb.bio.io.wwpdb import RemoteStructureProvider as PDB
from csb.statistics.rand import random_rotation

pdb = PDB()

def load_coords(pdbcode, atoms=None):

    entry = pdb.get(pdbcode)

    if atoms is None:
        return entry.get_coordinates()
    else:
        return np.array([residue[atom].vector for chainid in entry for residue in entry[chainid]
                         if residue.has_structure for atom in residue if atom in atoms])

def random_projection(coords):

    d = coords.shape[1]
    R = random_rotation(np.zeros((d,d)))

    return np.dot(coords, R.T)[:,:-1], R

def projection_series(pdbcode, n_projections, center_coords=True, atoms=None):
    """projection_series
    --------------------

    Create a series of random projections of a 3d point clouds obtained from
    a PDB entry.

    Parameters
    ----------

    pdbcode:
      4-letter code indicating the PDB entry

    n_projections: 
      integer specifying the desired number of random projections

    center_coords:
      flag specifying if the coordinates should be centered before projecting
      them (default: center_coords=True)

    atoms:
      if not None, this iterable lists the atom names whose coordinates will be
      included in the coordinate array
    """
    coords = load_coords(pdbcode, atoms=atoms)

    if center_coords: coords -= coords.mean(0)

    projections = []
    rotations = []

    for _ in range(n_projections):

        projection, rotation = random_projection(coords)
        projections.append(projection)
        rotations.append(rotation)

    return coords, np.array(projections), np.array(rotations)
