"""
Test variants of the matrix Bingham distribution that into account the
special structure of the A and C matrix
"""
from clouds import MatrixBingham
from clouds.core import take_time
from clouds.rotation import random_rotation

import numpy as np

class ProjectedBingham(MatrixBingham):

    def __init__(self, A, B):

        C = np.eye(len(A))
        C[-1,-1] = 0
        super(ProjectedBingham, self).__init__(A, B, C)

    def log_prob(self, R):

        log_p = np.trace(self.B) - np.dot(R[-1],np.dot(self.B,R[-1]))
        log_p+= np.sum(self.A * R)

        return log_p

class ProjectedBingham2(ProjectedBingham):

    def log_prob(self, R):

        log_p = np.trace(self.B) - np.dot(R[-1],np.dot(self.B,R[-1]))
        log_p+= np.dot(self.A.flatten(), R.flatten())

        return log_p

class ProjectedBingham3(ProjectedBingham):

    def log_prob(self, R):

        log_p = np.trace(self.B) - np.dot(R[-1],np.dot(self.B,R[-1]))
        log_p+= np.sum([np.dot(self.A[i],R[i]) for i in range(len(R)-1)])

        return log_p

if __name__ == '__main__':

    from test_rotation_finder import random_model

    p = random_model()
    p.A[-1] = 0.
    
    q = ProjectedBingham(p.A,p.B)
    r = ProjectedBingham2(p.A,p.B)
    s = ProjectedBingham3(p.A,p.B)

    p.C[...] = q.C

    n_test = int(1e5)

    R = random_rotation(n_test)

    with take_time('MatrixBingham'):
        a = np.array(map(p.log_prob, R))

    with take_time('ProjectedBingham'):
        b = np.array(map(q.log_prob, R))

    with take_time('ProjectedBingham2'):
        c = np.array(map(r.log_prob, R))

    with take_time('ProjectedBingham3'):
        d = np.array(map(r.log_prob, R))

    print np.fabs(a-b).max(), np.fabs(a-c).max(), np.fabs(b-c).max(), np.fabs(b-d).max()

