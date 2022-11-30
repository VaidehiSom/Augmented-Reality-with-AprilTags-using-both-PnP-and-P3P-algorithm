# from ssl import VERIFY_ALLOW_PROXY_CERTS
from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    # Pw_ = np.zeros((4,2))
    # for i in range(4):
    #     Pw_[i][0] = Pw[i][0]
    #     Pw_[i][1] = Pw[i][1]

    # # Pw = Pw[:][0:2]

    H = est_homography(Pw[:,0:2], Pc)
    H = H/H[-1,-1]
    # Homography Approach
    # Following slides: Pose from Projective Transformation
    # h1 = (np.transpose(K) * H)[0]
    # h2 = (np.transpose(K) * H)[1]
    # h3 = (np.transpose(K) * H)[2]
    # A = [h1, h2, h3]

    A = np.linalg.inv(K) @ H
    # print("A", np.shape(A))
    h1 = A[:,0]
    h2 = A[:,1]
    h3 = A[:,2]
    h_prime = np.hstack((h1[:, None], h2[:, None], np.cross(h1,h2)[:, None]))
    # print("h", h_prime)
    [U, _, Vt] = np.linalg.svd(h_prime)

    det_UVt = np.linalg.det(U@Vt)
    # B = [ [1,0,0      ],
    #       [0,1,0      ],
    #       [0,0,det_UVt]]
    B = np.eye(3,dtype=np.float64)
    B[-1,-1] = det_UVt
    R = U @ B @ Vt
    R = np.linalg.inv(R)
    t = h3 / np.linalg.norm(h1)

    # print("R", np.shape(R))
    # print("t", np.shape(t))

    return R, -R@t
