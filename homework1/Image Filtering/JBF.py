import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        height, width, channel = img.shape
        #GS table
        gs = np.zeros((self.wndw_size, self.wndw_size))

        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                gs[i, j] = np.exp(np.divide(np.square(i - self.pad_w) + np.square(j - self.pad_w), -2 * self.sigma_s**2))

        gs = gs.reshape(-1, 1, 1)

        #GR table
        gr_table = np.zeros(256)

        for i in range(256):
            gr_table[i] = np.exp(np.divide((i / 255)**2, -2 * self.sigma_r**2))


        if len(guidance.shape) == 3:
            cube_r = []
            cube_g = []
            cube_b = []
            img_cube_r = []
            img_cube_g = []
            img_cube_b = []
            
            for i in range(0, self.wndw_size**2):
                row = i // self.wndw_size
                col = i % self.wndw_size
                cube_r.append((padded_guidance[row:height+row, col:width+col, 0]-padded_guidance[self.pad_w:height+self.pad_w, self.pad_w:width+self.pad_w, 0]))
                cube_g.append((padded_guidance[row:height+row, col:width+col, 1]-padded_guidance[self.pad_w:height+self.pad_w, self.pad_w:width+self.pad_w, 1]))
                cube_b.append((padded_guidance[row:height+row, col:width+col, 2]-padded_guidance[self.pad_w:height+self.pad_w, self.pad_w:width+self.pad_w, 2]))
                img_cube_r.append(padded_img[row:height+row, col:width+col, 0])
                img_cube_g.append(padded_img[row:height+row, col:width+col, 1])
                img_cube_b.append(padded_img[row:height+row, col:width+col, 2])



            cube_r = np.abs(np.array(cube_r))
            cube_g = np.abs(np.array(cube_g))
            cube_b = np.abs(np.array(cube_b))
            img_cube_r = np.array(img_cube_r)
            img_cube_g = np.array(img_cube_g)
            img_cube_b = np.array(img_cube_b)
            gr = np.multiply(gr_table[cube_r], gr_table[cube_g])
            gr = np.multiply(gr, gr_table[cube_b])

            W = np.multiply(gs, gr)

            
            output = np.zeros(img.shape)

            output[:, :, 0] = np.divide(np.multiply(W, img_cube_r).sum(axis = 0), W.sum(axis = 0))
            output[:, :, 1] = np.divide(np.multiply(W, img_cube_g).sum(axis = 0), W.sum(axis = 0))
            output[:, :, 2] = np.divide(np.multiply(W, img_cube_b).sum(axis = 0), W.sum(axis = 0))

        else:
            cube = []
            img_cube_r = []
            img_cube_g = []
            img_cube_b = []
            
            for i in range(0, self.wndw_size**2):
                row = i // self.wndw_size
                col = i % self.wndw_size
                cube.append((padded_guidance[row:height+row, col:width+col]-padded_guidance[self.pad_w:height+self.pad_w, self.pad_w:width+self.pad_w]))
                img_cube_r.append(padded_img[row:height+row, col:width+col, 0])
                img_cube_g.append(padded_img[row:height+row, col:width+col, 1])
                img_cube_b.append(padded_img[row:height+row, col:width+col, 2])



            cube = np.abs(np.array(cube))
            img_cube_r = np.array(img_cube_r)
            img_cube_g = np.array(img_cube_g)
            img_cube_b = np.array(img_cube_b)
            gr = gr_table[cube]

            W = np.multiply(gs, gr)

            
            output = np.zeros(img.shape)

            output[:, :, 0] = np.divide(np.multiply(W, img_cube_r).sum(axis = 0), W.sum(axis = 0))
            output[:, :, 1] = np.divide(np.multiply(W, img_cube_g).sum(axis = 0), W.sum(axis = 0))
            output[:, :, 2] = np.divide(np.multiply(W, img_cube_b).sum(axis = 0), W.sum(axis = 0))


        
        return np.clip(output, 0, 255).astype(np.uint8)