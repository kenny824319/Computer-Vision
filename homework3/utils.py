import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((N*2, 9), dtype=np.float32)
    for i in range(N):
        A[2*i] = [u[i][0], u[i][1], 1, 0, 0, 0, -1.0*v[i][0]*u[i][0], -1.0*v[i][0]*u[i][1], -1.0*v[i][0]]
        A[2*i+1] = [0, 0, 0, u[i][0], u[i][1], 1, -1.0*v[i][1]*u[i][0], -1.0*v[i][1]*u[i][1], -1.0*v[i][1]]
    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    H = np.reshape(VT[-1], (3,3))
    return H


def warping(u, v, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in u(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in v(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param u: source image
    :param v: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_u, w_u, ch = u.shape
    h_v, w_v, ch = v.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xc, yc = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1), sparse = False)
    xrow = xc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    yrow = yc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    onerow =  np.ones(( 1,(xmax-xmin)*(ymax-ymin) ))
    M = np.concatenate((xrow, yrow, onerow), axis = 0)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        Mbar = H_inv @ M
        Mbar = np.divide(Mbar, Mbar[-1,:]) 
        uy = np.round( Mbar[1,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)
        ux = np.round( Mbar[0,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        h_mask = (0<uy)&(uy<h_u)
        w_mask = (0<ux)&(ux<w_u)
        mask = h_mask&w_mask
        # TODO: 6. assign to destination image with proper masking
        v[yc[mask], xc[mask]] = u[uy[mask], ux[mask]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        Mbar = H @ M
        Mbar = np.divide(Mbar, Mbar[-1,:])
        vy = np.round(Mbar[1,:].reshape(ymax-ymin,xmax-xmin)).astype(int)
        vx = np.round(Mbar[0,:].reshape(ymax-ymin,xmax-xmin)).astype(int)
        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing
        v[np.clip(vy, 0, v.shape[0]-1), np.clip(vx, 0, v.shape[1]-1)] = u

    return v
