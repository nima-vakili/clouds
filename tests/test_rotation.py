import numpy as np

from csb.numeric import euler, euler_angles, rotation_matrix

def R_x(angle):
    return rotation_matrix(np.eye(3)[0], angle)

def R_y(angle):
    return rotation_matrix(np.eye(3)[1], angle)

def R_z(angle):
    return rotation_matrix(np.eye(3)[2], angle)

def dR_y(angle):
    return np.array([[-np.sin(angle), 0.,-np.cos(angle)],
                     [0., 0., 0.],
                     [ np.cos(angle), 0.,-np.sin(angle)]])

def dR_z(angle):
    return np.array([[-np.sin(angle), +np.cos(angle), 0.],
                     [-np.cos(angle), -np.sin(angle), 0.],
                     [0., 0., 0.]])

def euler2(a, b, c):
    return np.dot(R_z(c), np.dot(R_y(b), R_z(a)))

def grad_euler2(a, b, c):
    return np.array([np.dot(R_z(c), np.dot(R_y(b), dR_z(a))),
                     np.dot(R_z(c), np.dot(dR_y(b), R_z(a))),
                     np.dot(dR_z(c), np.dot(R_y(b), R_z(a)))])

