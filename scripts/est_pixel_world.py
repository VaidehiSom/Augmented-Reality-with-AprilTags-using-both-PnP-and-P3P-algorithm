import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    # Pc = np.ones((np.shape(pixels)[0],3))
    # Pw = np.zeros((np.shape(pixels)[0],3))
    # Pc[:,0:2] = pixels
    # Rcw = np.transpose(R_wc)
    # tcw = -Rcw@(t_wc)

    # projmatrix = np.zeros((3,3))
    # projmatrix[:,[0,1]] = Rcw[:,[0,1]]
    # projmatrix[:,2] = tcw
    # for i in range(np.shape(pixels)[0]):
    #   Pw[i,:] = np.matmul(np.linalg.inv(projmatrix),np.matmul(np.linalg.inv(K),Pc[i,:]))
    #   Pw[i,0] = Pw[i,0]/Pw[i,2]
    #   Pw[i,1] = Pw[i,1]/Pw[i,2]

    # N = np.shape(pixels)[0]
    # Pw = np.zeros((N,3))
    # Pc = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    # R = np.linalg.inv(R_wc)
    # t = 
    # for i in range(N):
    #     pixels_3d = [pixels[i][0], pixels[i][1], 1]
    #     # Rt = [ [R_wc[0][0], R_wc[0][1], R_wc[0][2], t_wc[0] ],
    #     #        [R_wc[1][0], R_wc[1][1], R_wc[1][2], t_wc[1] ],
    #     #        [R_wc[2][0], R_wc[2][1], R_wc[2][2], t_wc[2] ] ]
    #     # A = np.matmul(K, Rt)
    #     # print(np.shape(A))
    #     # print(np.shape(pixels_3d))
    #     # Pw[i] = np.matmul( np.linalg.inv(A), pixels_3d )
    #     # mat = K@np.hstack((R_wc[:,:], t_wc[:, None]))
    #     # Pw[i] = np.linalg.inv(mat)@pixels_3d
    #     Pw[i,:] = np.linalg.inv(R_wc)@np.linalg.inv(K)@pixels_3d - np.linalg.inv(R_wc)@np.linalg.inv(K)@K@t_wc
    #     Pw[i,:] = Pw[i,:] / Pw[i,-1]
    #     Pw[i,-1] = 0

    Pw = np.zeros([Pc.shape[0],3])
    Pc = np.hstack((pixels, np.ones([pixels.shape[0],1])))
    R_wc = np.linalg.inv(R_wc)
    t_wc = -np.matmul(R_wc,t_wc).reshape(3,1)

    Rt = np.hstack((R_wc[:,0:-1],t_wc))

    for i in range(pixels.shape[0]):
        Pw[i,:] = np.transpose(np.linalg.inv(K@Rt)@np.transpose(Pc[i,:]))
        Pw[i,:] = Pw[i,:]/Pw[i,-1]
        Pw[i,-1] = 0

    return Pw
