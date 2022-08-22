import numpy as np
import cv2.ximgproc as xip
import cv2

def construct_table(img):
    pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0, 0, 0))
    cube = np.stack((pad_img[0:-2, 0:-2],
                    pad_img[0:-2, 1:-1], 
                    pad_img[0:-2, 2:], 
                    pad_img[1:-1, 0:-2],  
                    pad_img[1:-1, 2:], 
                    pad_img[2:, 0:-2], 
                    pad_img[2:, 1:-1], 
                    pad_img[2:, 2:]), axis=-1)
    return cube <= np.expand_dims(img, axis=-1)

def cost_comput_and_aggre(img_l, img_r,  max_disp):
    table_l, table_r = construct_table(img_l), construct_table(img_r)
    h, w, c, _ = table_l.shape
    cost_l = np.zeros((h, w, max_disp+1), dtype = "float32")
    cost_r = np.zeros((h, w, max_disp+1), dtype = "float32")
    for d in range(max_disp+1):
        shift_left, shift_right = table_l[:, d:], table_r[:, :w-d]
        hamming_dist = np.sum(shift_left ^ shift_right, axis=(2, 3))
        pad_left, pad_right = np.tile(hamming_dist[:, [0]], (1, d)), np.tile(hamming_dist[:, [-1]], (1, d))
        cost_l[:, :, d] = np.hstack((pad_left, hamming_dist))
        cost_l[:, :, d] = xip.jointBilateralFilter(img_l, cost_l[:, :, d], 30, 5, 5)
        cost_r[:, :, d] = np.hstack((hamming_dist, pad_right))
        cost_r[:, :, d] = xip.jointBilateralFilter(img_r, cost_r[:, :, d], 30, 5, 5)
    return cost_l, cost_r

def check_consist(disparity_l, disparity_r):
    h, w = disparity_l.shape
    for i in range(h):
        for j in range(w):
            if j - disparity_l[i, j] >= 0 and disparity_l[i, j] != disparity_r[i, j-disparity_l[i, j]]:
                disparity_l[i, j] = -1
    return disparity_l

def hole_fill(hole_img, max_disp):
    h, w = hole_img.shape
    for i in range(h):
        for j in range(w):
            if hole_img[i, j] == -1:
                l = r = 0
                while j - l >= 0 and hole_img[i, j-l] == -1:
                    l += 1
                if j - l < 0:
                    FL = max_disp
                else:
                    FL = hole_img[i, j-l]
                while j+r <= w-1 and hole_img[i, j+r] == -1:
                    r += 1
                if j+r > w-1:
                    FR = max_disp
                else:
                    FR = hole_img[i, j+r]
                hole_img[i, j] = min(FL, FR)
    return hole_img


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    cost_l, cost_r = cost_comput_and_aggre(Il, Ir, max_disp)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disparity_l, disparity_r = cost_l.argmin(axis=-1), cost_r.argmin(axis=-1)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    hole_img = check_consist(disparity_l, disparity_r)
    fin_img = hole_fill(hole_img, max_disp)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), fin_img.astype(np.uint8), 18, 1)



    return labels.astype(np.uint8)
    