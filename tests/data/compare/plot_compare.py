import os
import numpy as np
import pylab as plt


path = os.curdir

corr = np.load(os.path.join(path, 'corr.npz'))['arr_0']
like = np.load(os.path.join(path, 'like.npz'))['arr_0']   
R_true = np.load(os.path.join(path, 'R_true.npz'))['arr_0']
mu_true = np.load(os.path.join(path, 'mu_true.npz'))['arr_0']
L_max  = np.load(os.path.join(path, 'L_max.npz'))['arr_0']
all_results = np.load(os.path.join(path, 'all_results.npz'))['arr_0']
init_R  = np.load(os.path.join(path, 'init_R.npz'))['arr_0']
  

MH    = [corr[i][0] for i in range(1000)]
HMC   = [corr[i][1] for i in range(1000)]  
Grid  = [corr[i][2] for i in range(1000)]  
 


fig, ax = plt.subplots(1,2,figsize=(10,5))
ax = ax.flat

ax[0].hist(MH, 15, facecolor='k', alpha=0.35, label='Metropolis-Hasting')
ax[0].hist(HMC, 15, facecolor='r', alpha=0.27, label='HMC')
ax[0].axvline(Grid[0], ls='--', lw=3, color='k', label='Grid-Search')
ax[0].set_xlabel('Kernel Correlation', fontsize=20)
ax[0].legend(fancybox=True, framealpha=0.5)

MH2    = [like[i][0] for i in range(1000)]
HMC2   = [like[i][1] for i in range(1000)]
Grid2  = [like[i][2] for i in range(1000)]

ax[1].hist(MH2, 15, facecolor='k', alpha=0.35, label='Metropolis-Hasting')
ax[1].hist(HMC2, 15, facecolor='r', alpha=0.27, label='HMC')
ax[1].axvline(Grid2[0], ls='--', lw=3, color='k', label='Grid-Search')
ax[1].axvline(L_max, ls='--', lw=3, color='c', label='True Value')
ax[1].set_xlabel('Log Likelihood', fontsize=20)
ax[1].legend(fancybox=True, framealpha=0.5)
ax[1].xaxis.get_major_formatter().set_powerlimits((0, 0))

fig.tight_layout()


plt.figure()  
plt.scatter(MH,HMC, color='black'); plt.plot([0,1],[0,1], 'r')
plt.xlim([.7,1])
plt.ylim([.7,1])
plt.xlabel("Metropolis-Hasting", fontsize=20)
plt.ylabel("HMC", fontsize=20)
  
  
from spin import EulerAngles  
  
EulerAngles._from_matrix(init_R[0])
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

