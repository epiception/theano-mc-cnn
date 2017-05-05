import os
import numpy as np
import cv2
import argparse
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d, relu

from tqdm import tqdm
import glob

mPath = os.path.dirname(os.path.abspath("Deploy_Depth.py")) + "/Deploy/"
wPath = os.path.dirname(os.path.abspath("Deploy_Depth.py")) + "/Weights/"

parser = argparse.ArgumentParser(description='Deploy Neural Network to find disparity map from Image pairs')
parser.add_argument('path_1',metavar='-path_left',help='path to left image',type=str)
parser.add_argument('path_2',metavar='-path_right',help='path to right image',type=str)
parser.add_argument('disparity',metavar='-disp',help='Disparity Range',type=int)
args = parser.parse_args()

left_image = cv2.imread(args.path_1,0)
right_image = cv2.imread(args.path_2,0)
disparity_range = args.disparity

weight_files_list = sorted(glob.glob(wPath+"*.npy"))
weight_files_list = weight_files_list[-4:]

f_size = 3
patch_size = 9

assert left_image.shape[0] == right_image.shape[0], "Image Dimensions must match!"
assert left_image.shape[1] == right_image.shape[1], "Image Dimensions must match!"

img_ht = left_image.shape[0]
img_wdt = left_image.shape[1]

patch_width = int(patch_size/2.0)
patch_height = int(patch_size/2.0)

batch_size = 1

X_l = T.ftensor4()
X_r = T.ftensor4()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    fan_in = np.prod(shape[1:])
    fan_out = (shape[0] * np.prod(shape[2:]))

    local=np.random.randn(*shape)
    W_bound = np.sqrt(2.0/(fan_in))

    return theano.shared(floatX(local*W_bound))

def l2_norm_layer(ip):
    norm = T.inv(T.sqrt(((ip**2).sum(axis=(1,2,3)))))
    sq = T.reshape(norm, (batch_size,1,1,1))
    op = ip*sq

    return op

'''layer size specs'''
model_size = []

model_size.append((64,1,f_size,f_size))
model_size.append((64,64,f_size,f_size))
model_size.append((64,64,3,3))
model_size.append((64,64,3,3))

def model(X, w1, w2, w3, w4):

    l1 = relu((conv2d(X,w1, border_mode='full')))
    l2 = relu((conv2d(l1,w2, border_mode='valid')))
    l3 = relu((conv2d(l2,w3,border_mode='full')))
    l4 = conv2d(l3,w4,border_mode='valid')

    output = l2_norm_layer(l4)

    return output

w1 = init_weights(model_size[0])
w2 = init_weights(model_size[1])
w3 = init_weights(model_size[2])
w4 = init_weights(model_size[3])

params = [w1, w2, w3, w4]

nn_output_left = model(X_l, *params)
nn_output_right = model(X_r, *params)


print("Compiling")
test_model = theano.function(inputs=[X_l,X_r], outputs = [nn_output_left, nn_output_right])
print("Compiled Model")

print("Loading latest weights")
layer_idx=0
if not weight_files_list:
    print("No Weights in Weights folder")
    raise AssertionError

else:
    for layer in tqdm(weight_files_list):
        current_layer = np.load(layer)
        params[layer_idx].set_value(current_layer)
        layer_idx+=1

left_image = np.float32(left_image - np.mean(left_image)/(np.std(left_image)))
right_image = np.float32(right_image - np.mean(right_image)/(np.std(right_image)))

left_image = left_image.reshape(1,1,img_ht,img_wdt)
right_image = right_image.reshape(1,1,img_ht,img_wdt)

output_vol = test_model(left_image, right_image)

output_vol_left = output_vol[0][0]
output_vol_right = output_vol[1][0]

Cost_grid = np.zeros((img_ht, img_wdt, disparity_range))

print("\nLeft and Right images passed through Convolution and Normalization Layers")

print("\nNow Computing Sliding Dot Product of Output Volumes")

for i in tqdm(range(patch_height, img_ht - patch_height)):

    for j in range(patch_width, img_wdt - patch_width - 1):

        for disp in range(0, disparity_range):

            q = j-disp
            if(q > patch_width and q < (img_wdt - patch_width)):
                left_patch_volume = output_vol_left[:, i - patch_height: i+ patch_height + 1, j - patch_width:j + patch_width + 1]
                right_patch_volume = output_vol_right[:, i - patch_height: i + patch_height + 1, q-patch_width: q + patch_width + 1]
                cost = (left_patch_volume*right_patch_volume).sum()
                Cost_grid[i,j,disp] = cost

print("Cost grid saved")
print("Creating Disparity Map")
disparity_map = np.zeros((img_ht,img_wdt))
for i in tqdm(range(0, img_ht)):
    for j in range(0, img_wdt):
        cost_max = 0
        disparity_max=0
        for disp in range(0, disparity_range):
            if(Cost_grid[i,j,disp] > cost_max):
                cost_max = Cost_grid[i,j,disp]
                disparity_max = disp

        disparity_map[i,j] = disparity_max

np.savetxt(mPath + 'Cost_grid_map.txt', disparity_map)

print("Disparity Map Saved as Cost_grid_map.txt")

gmap = disparity_map.astype(np.int16)
gmap = cv2.medianBlur(gmap,5)

cv2.imshow('Map',gmap*1.5/128.0)

saving_img = gmap * 1.5
cv2.imwrite("Saved_map.png", saving_img)

cv2.waitKey(0)
