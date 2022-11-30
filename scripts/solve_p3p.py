from cmath import inf
import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    # Invoke Procrustes function to find R, t
    # select the R and t that could transoform all 4 points correctly. 
    Pc_3d = np.zeros((4,3))
    Pc_3d[0] = [Pc[0,0]-K[-1,0], Pc[0,1]-K[-1,1], (K[0,0]+K[1,1])/2]
    Pc_3d[1] = [Pc[1,0]-K[-1,0], Pc[1,1]-K[-1,1], (K[0,0]+K[1,1])/2]
    Pc_3d[2] = [Pc[2,0]-K[-1,0], Pc[2,1]-K[-1,1], (K[0,0]+K[1,1])/2]
    Pc_3d[3] = [Pc[3,0]-K[-1,0], Pc[3,1]-K[-1,1], (K[0,0]+K[1,1])/2]

    # Pc_3d = np.linalg.inv(K)@Pc_3d[0:3]
    # Pc_3d = Pc_3d[0:3]

    calibrated_pnts = (np.linalg.inv(K) @ np.hstack((Pc[1:,:], np.ones((np.shape(Pc[1:,:])[0],1)))).T).T
    
    j1 = calibrated_pnts[0,:] / np.linalg.norm(calibrated_pnts[0,:])
    j2 = calibrated_pnts[1,:] / np.linalg.norm(calibrated_pnts[1,:])
    j3 = calibrated_pnts[2,:] / np.linalg.norm(calibrated_pnts[2,:])

    ca = np.dot(j2,j3)
    cb = np.dot(j1,j3)
    cg = np.dot(j1,j2)

    d13 = np.linalg.norm(Pw[1]-Pw[3])
    d23 = np.linalg.norm(Pw[2]-Pw[3])
    d12 = np.linalg.norm(Pw[1]-Pw[2])

    term_acb = (d23**2 - d12**2)/d13**2
    term_acb_2 = (d23**2 + d12**2)/d13**2
    term_bcb = (d13**2 - d12**2)/d13**2
    term_bab = (d13**2 - d23**2)/d13**2

    A4 = (term_acb - 1)**2 - 4*d12**2*(ca)**2/(d13**2)
    A3 = 4*( term_acb*(1-term_acb)*cb - (1-term_acb_2)*ca*cg + 2*(d12**2/d13**2)*ca**2*cb)
    A2 = 2*(term_acb**2 - 1 + 2*term_acb**2*cb**2 + 2*term_bcb*ca**2 - 4*term_acb_2*ca*cb*cg + 2*term_bab*cg**2)
    A1 = 4*(-term_acb*(1+term_acb)*cb + 2*d23**2*cg**2*cb/d13**2 - (1 - term_acb_2)*ca*cg)
    A0 = (1 + term_acb)**2 - 4*d23**2*cg**2/d13**2

    coeff = [A4, A3, A2, A1, A0]
    v = np.roots(coeff)
    # print(coeff)
    # print(v)

    R = np.zeros((3,3))
    t = np.zeros(3)
    closest_pnt = inf

    for i in range(4):

        if(np.isreal(v[i]) and v[i] > 0):
            u = ( (-1 + term_acb)*v[i]**2 - 2*term_acb*cb*v[i] + 1 + term_acb )/ (2*(cg - v[i]*ca))
            # print(u)
            if(u>0):

                d1 = np.sqrt( d23**2 / ( u**2 + v[i]**2 - 2*u*v[i]*ca))
                d2 = u*d1
                d3 = v[i]*d1

                # Pc_3d_scaled = np.zeros(np.shape(Pc_3d))
                # Pc_3d_scaled[0] = d1*Pc_3d[0]/np.linalg.norm(Pc_3d[0])
                # Pc_3d_scaled[1] = d2*Pc_3d[1]/np.linalg.norm(Pc_3d[1])
                # Pc_3d_scaled[2] = d3*Pc_3d[2]/np.linalg.norm(Pc_3d[2])
                p = np.vstack((d1*j1, d2*j2, d3*j3))
                R_,t_ = Procrustes(Pw[1:,:], p)

                # R_,t_ = Procrustes(Pw[1:4], Pc_3d_scaled[0:3])
                Y = K@ (R_@np.transpose(Pw[0,:]) + t_)
                Y = (Y / Y[-1])[:-1]
                # Y = (np.hstack(R_.T, -R_.T*t_.reshape(3,1)) @ np.vstack((Pw[0].T, np.ones(np.shape(Pw[0]))))).T
                # Y = np.transpose(Y)
                # dist = (Y[0] - Pc[0,0])**2 + (Y[1] - Pc[0,1])**2 
                dist = np.linalg.norm(Y - Pc[0,:])
                if(dist < closest_pnt):
                    closest_pnt = dist
                    R = R_
                    t = t_

    return np.linalg.inv(R), -np.linalg.inv(R)@t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    A_bar = np.mean(X, axis=0)
    B_bar = np.mean(Y, axis=0)
    
    A = np.transpose(X - A_bar)
    B = np.transpose(Y - B_bar)
    
    [U, S , Vt ] = np.linalg.svd(B@np.transpose(A))
    V = np.transpose(Vt)
    det_UVt = np.linalg.det(V@np.transpose(U))
    B_ = [[1,0,0      ],
          [0,1,0      ],
          [0,0,det_UVt]]
    R = (U@B_) @ Vt
    t = B_bar - R@A_bar

    return R, t
