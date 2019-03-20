import cv2
import sys
from PIL import Image
import argparse
import os,os.path
import numpy as np
import math
import matplotlib.pyplot as plt
import random

out_dir = './output/'
valid_image = ['.jpg','.png']

# TODO : read shutter speed by read txt
shutter_speed_list = np.array([0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024])
shutter_speed_list = np.reciprocal(shutter_speed_list)
weight_function = np.zeros(256, dtype=np.float64)
ZMIN = 0
ZMAX = 255
ZAVG = 127


def read_img(path):
    # read all image in path folder
    image_list = []
    count = 0
    # sort image by file name
    for filename in sorted(os.listdir(path)):
        ext = os.path.splitext(filename)[1]
        # only read .jpg / .png
        if ext.lower() in valid_image:
            im = cv2.imread(os.path.join(path,filename))
            image_list.append(np.array(im))
            count += 1
    image_list = np.array(image_list)
    # image_list shape : (number_of_image, image.shape[0], image.shape[1], 3)
    return image_list

def make_weight_function():
    # calcalate weighting function
    global weight_function
    for i in range(256):
        if i <= ZAVG:
            # sys.float_info.epsilon : avoid sum = 0 in construct radiance image
            weight_function[i] = i - ZMIN + sys.float_info.epsilon
        else:
            weight_function[i] = ZMAX - i + sys.float_info.epsilon

def sampleIntensities(imgs):
    # sample intensity in 0 ~ 255
    # input shape : (number_of_image, image.shape[0], image.shape[1])

    num_layer = len(imgs)
    num_sample = ZMAX - ZMIN + 1

    mid_img = imgs[(num_layer // 2)]
    intensity = np.zeros((num_sample, num_layer), dtype=np.uint8)

    for i in range(ZMIN, ZMAX + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            r, c = rows[idx], cols[idx]
            for j in range(num_layer):
                intensity[i, j] = imgs[j][r, c]

    # output shape : (256,number_of_image)
    return intensity
    
def gslove(img_list, sample_list, lamdba, rgb):
    # ppt page 44 / 45
    n = 256
    A = np.zeros((img_list.shape[0]*sample_list.shape[0]+n-1,n+sample_list.shape[0]), dtype = np.float64)
    B = np.zeros((A.shape[0],1), dtype = np.float64)
    k = 0
    
    for i in range(sample_list.shape[0]):
        for j in range(img_list.shape[0]):
            wij = weight_function[sample_list[i][j]]
            A[k][sample_list[i][j]] = wij
            A[k][n+i] = -wij
            B[k][0] = wij * math.log(shutter_speed_list[j])
            k += 1
    A[k][ZAVG] = 1
    k += 1
    for i in range(1,n-1):
        A[k][i-1] = lamdba * weight_function[i]
        A[k][i] = -2*lamdba * weight_function[i]
        A[k][i+1] = lamdba * weight_function[i]
        k += 1
    
    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, B)
    g = x[0:n]
    lE = x[n:x.shape[0]]

    plt.figure()
    plt.plot(g)
    plt.savefig(out_dir+'response_curve_'+str(rgb)+'.png')
    return g


def reconstruct_radiance_map(image_list, response_curve, rgb):
    # ppt page 47
    rad =  np.zeros((image_list.shape[1],image_list.shape[2]), dtype=np.float64)

    for i in range(image_list.shape[1]):
        for j in range(image_list.shape[2]):
            # g = np.array([response_curve[image_list[x][i][j][rgb]] for x in range(image_list.shape[0])])
            w = np.array([weight_function[image_list[x][i][j][rgb]] for x in range(image_list.shape[0])])
            w_sum = np.nansum(w)
            if w_sum >= 0:
                rad[i][j] = (np.nansum(weight_function[image_list[:,i,j,rgb]] * (response_curve[image_list[:,i,j,rgb]]-np.log(shutter_speed_list)))) / w_sum
            else:
                rad[i][j] = g[image_list.shape[0] // 2] - math.log(shutter_speed_list[image_list.shape[0] // 2])
    return rad
                # rad[i,j,channel] = weight_function[image_list[x][i][j][channel]]

def tone_mapping(image, gamma):
    return cv2.pow(image/255., 1.0/gamma)

if __name__ == '__main__':
    # set args
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="path to images directory")
    args = vars(ap.parse_args())
    path = args["path"]

    # radiance image
    radiance_img = []

    # read image 
    image_list = read_img(path)

    # aglin image
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(image_list, image_list)

    # weighting function : ppt page 32
    make_weight_function()

    # recover response curve by channel
    for i in range(3):

        # sample point to avoid out of memory
        sample_int = sampleIntensities([x[:,:,i] for x in image_list])

        # recover response curse
        response_curve = gslove(image_list, sample_int, 100, i)

        # reconstruct radiance map
        radiance_img.append(reconstruct_radiance_map(image_list, response_curve, i))

    radiance_img = np.array(radiance_img)
    
    # output : reshape radiance image to rgb image
    output = np.zeros((radiance_img.shape[1],radiance_img.shape[2],3))
    for i in range(3):
        output[:,:,i] = radiance_img[i]
    
    # TODO : tonemapping

    # cv2.imwrite(out_dir+'test_out.hdr',output)
    # output = tone_mapping(output, 1.5)
    # output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite(out_dir+'test_out.png',output)
