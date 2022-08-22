import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]      
  
        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        goodu = []
        goodv = []

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                goodu.append(kp1[m.queryIdx].pt)
                goodv.append(kp2[m.trainIdx].pt)
        goodu = np.array(goodu)
        goodv = np.array(goodv)

        # TODO: 2. apply RANSAC to choose best H
        times = 5000
        threshold = 4
        inlineNmax = 0
        HNmax = np.eye(3)
        index = np.arange(len(goodu))
        for i in range(0, times+1):
            np.random.shuffle(index)
            random_u = goodu[index[0:5]]
            random_v = goodv[index[0:5]]
            H = solve_homography(random_v, random_u)
            
            onerow = np.ones((1,len(goodu)))
            M = np.concatenate( (goodv.T, onerow), axis=0)
            W = np.concatenate( (goodu.T, onerow), axis=0)         
            Mbar = H @ M
            Mbar = np.divide(Mbar, Mbar[-1,:])
            
            err  = np.linalg.norm((Mbar-W)[:-1,:], ord=1, axis=0)
            inlineN = sum(err<threshold)
            
            if inlineN > inlineNmax:
                inlineNmax = inlineN
                HNmax = H

        # TODO: 3. chain the homographies    
        # TODO: 4. apply warping
        last_best_H = last_best_H @ HNmax
        output = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    return output


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)