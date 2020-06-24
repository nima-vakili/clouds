import os
import numpy as np
import pylab as plt
from isd.Ensemble import Ensemble


path = os.curdir
samples = np.load(os.path.join(path, 'samples.npz'))['arr_0']
E = Ensemble(samples)
traj = np.load(os.path.join(path, 'traj.npz'))['arr_0']    
cc = np.load(os.path.join(path, 'cc.npz'))['arr_0']
var = np.load(os.path.join(path, 'var.npz'))['arr_0']               

n = 1 

fig, ax = plt.subplots(1,3,figsize=(10,5))
ax[0].plot(E.E[1:][2:] if len(E) < n else E.E[n:][2:],'k',linewidth=3.0)
ax[0].set_xlabel('Step', fontsize=20)
ax[0].set_ylabel('Energy function', fontsize=20)
ax[1].plot(cc,'k',linewidth=3.0)
ax[1].set_xlabel('Step', fontsize=20)
ax[1].set_ylabel('Cross-Correlation', fontsize=20)
ax[2].plot(var[5:],'k',linewidth=2)
ax[2].set_xlabel('Step', fontsize=20)
ax[2].set_ylabel('Precision', fontsize=20)


    
