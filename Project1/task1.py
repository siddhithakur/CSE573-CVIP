###############
##Design the function "findRotMat" to  return
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz
#It is ok to add other functions if you need
###############

import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    #......
    a_rd=np.deg2rad(alpha)
    b_rd=np.deg2rad(beta)
    g_rd=np.deg2rad(gamma)
    z_dash = np.matrix([[np.cos(a_rd), -np.sin(a_rd),0], [np.sin(a_rd), np.cos(a_rd), 0],[0,0,1]])
    x_dash=np.matrix([[1,0,0], [0, np.cos(b_rd), -np.sin(b_rd)],[0,np.sin(b_rd),np.cos(b_rd)]])
    z_dash_dash=np.matrix([[np.cos(g_rd), -np.sin(g_rd),0], [np.sin(g_rd), np.cos(g_rd), 0],[0,0,1]])
    #rotMat1=np.dot(z_dash,x_dash)
    #rotMat1=np.dot(rotMat1,z_dash_dash)
    #
    rotMat1=z_dash_dash * x_dash *z_dash
    rotMat2 = np.transpose(rotMat1)
    return rotMat1,rotMat2


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 60
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)
    print(rotMat1)
    print(rotMat2)
