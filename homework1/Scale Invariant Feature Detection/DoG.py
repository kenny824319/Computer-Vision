import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.dog_images = []


    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        height, width = image.shape
        # print("height = ", height)
        # print("width = ", width)
        
        gaussian_images.append(image)
        
        for i in range(1, 5):
            gaussian_images.append(cv2.GaussianBlur(image, (0, 0), self.sigma**i))
        
        #down sample on fifth image
        new_image = cv2.resize(gaussian_images[4], (width // 2, height // 2), interpolation = cv2.INTER_NEAREST)
        
        new_image_h, new_image_w = new_image.shape
        # print("new_image_h = ", new_image_h)
        # print("new_image_w = ", new_image_w)

        gaussian_images.append(new_image)

        for i in range(1, 5):
            gaussian_images.append(cv2.GaussianBlur(new_image, (0, 0), self.sigma**i))

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        #前五張經過GaussianBlur的圖兩兩相減
        for i in range(1, 5):
            self.dog_images.append(cv2.subtract(gaussian_images[i], gaussian_images[i-1]))
        #後五張經過GaussianBlur的圖兩兩相減
        for i in range(6, 10):
           self.dog_images.append(cv2.subtract(gaussian_images[i], gaussian_images[i-1]))




        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(1, 3):
            for h in range(1, height-1):
                for w in range(1, width-1):
                    if abs(self.dog_images[i][h][w]) > self.threshold:
                        cube = np.array([self.dog_images[i+1][h-1:h+2,w-1:w+2], 
                                        self.dog_images[i][h-1:h+2,w-1:w+2], 
                                        self.dog_images[i-1][h-1:h+2,w-1:w+2]])
                        if self.dog_images[i][h][w] > 0:
                            maximum = np.max(cube)
                            if maximum == self.dog_images[i][h][w]:
                                keypoints.append([h, w])
                        else:
                            minimum = np.min(cube)
                            if minimum == self.dog_images[i][h][w]:
                                keypoints.append([h, w])

        for i in range(5, 7):
            for h in range(1, height // 2 - 1):
                for w in range(1, width // 2 - 1):
                    if abs(self.dog_images[i][h][w]) > self.threshold:
                        cube = np.array([self.dog_images[i+1][h-1:h+2,w-1:w+2], 
                                        self.dog_images[i][h-1:h+2,w-1:w+2], 
                                        self.dog_images[i-1][h-1:h+2,w-1:w+2]])
                        if self.dog_images[i][h][w] > 0:
                            maximum = np.max(cube)
                            if maximum == self.dog_images[i][h][w]:
                                keypoints.append([h*2, w*2])
                        else:
                            minimum = np.min(cube)
                            if minimum == self.dog_images[i][h][w]:
                                keypoints.append([h*2, w*2])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.array(keypoints)
        keypoints = np.unique(keypoints, axis = 0)



        # # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints

    def save_DoG_res(self):
        for i in range(1, 5):
            dst = cv2.normalize(self.dog_images[i-1], None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite('DoG1-' + str(i) +'.png', dst)

        for idx, i in enumerate(range(5, 9)):
            dst = cv2.normalize(self.dog_images[i-1], None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite('DoG2-' + str(idx+1) +'.png', dst)

        return

