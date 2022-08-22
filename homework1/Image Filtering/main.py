import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter




def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_no', default='2', help='which img')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img_rgb)

    ### TODO ###
    f = open(args.setting_path, 'r')
    data = []
    for line in f:
        line = line.strip()
        line = line.split(',')
        line = [float(item) for item in line]
        data.append(line)

    RGB = np.array(data[0:-2])
    sigma_s = int(data[-2][0])
    sigma_r = data[-1][0]
    min_val = [2**64-1, -1]
    max_val = [0, -1]

    for i in range(6):
        if i:
            img_gray = RGB[i-1, 0] * img_rgb[:, :, 0] + RGB[i-1, 1] * img_rgb[:, :, 1] + RGB[i-1, 2] * img_rgb[:, :, 2]
        JBF = Joint_bilateral_filter(sigma_s, sigma_r)
        bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        cost = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        if cost < min_val[0]:
            min_val[0] = cost
            min_val[1] = i
        if cost > max_val[0]:
            max_val[0] = cost
            max_val[1] = i
        print(cost)

    print(min_val[0], min_val[1], max_val[0], max_val[1])
    min_img_gray = RGB[min_val[1]-1, 0] * img_rgb[:, :, 0] + RGB[min_val[1]-1, 1] * img_rgb[:, :, 1] + RGB[min_val[1]-1, 2] * img_rgb[:, :, 2]
    min_jbf_out = JBF.joint_bilateral_filter(img_rgb, min_img_gray).astype(np.uint8)
    min_jbf_out = cv2.cvtColor(min_jbf_out,cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.image_no+'_min_gray.png', min_img_gray)
    cv2.imwrite(args.image_no+'_min_JBF.png', min_jbf_out)
    max_img_gray = RGB[max_val[1]-1, 0] * img_rgb[:, :, 0] + RGB[max_val[1]-1, 1] * img_rgb[:, :, 1] + RGB[max_val[1]-1, 2] * img_rgb[:, :, 2]
    max_jbf_out = JBF.joint_bilateral_filter(img_rgb, max_img_gray).astype(np.uint8)
    max_jbf_out = cv2.cvtColor(max_jbf_out,cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.image_no+'_max_gray.png', max_img_gray)
    cv2.imwrite(args.image_no+'_max_JBF.png', max_jbf_out)

         
    f.close()
    


if __name__ == '__main__':
    main()