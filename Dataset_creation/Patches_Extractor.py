import os
import cv2
import numpy as np
from random import randint
from random import choice
from PIL import Image

m_path = os.path.dirname(os.path.abspath("Patches_Extractor.py")) + "/Dataset_creation/"
print(m_path)

if not os.path.exists("PositivePatches"):
    os.makedirs("PositivePatches")
if not os.path.exists("NegativePatches"):
    os.makedirs("NegativePatches")
if not os.path.exists("ReferencePatches"):
    os.makedirs("ReferencePatches")
if not os.path.exists("Weights"):
    os.makedirs("Weights")

limit = 400000

img_ht = 375
img_wdt = 1242
window_ht = 9
window_wdt = 9

counter = 0
inner_counter = 0


width = int(window_wdt/2.0)
height = int(window_ht/2.0)
print(width,height)

for k in range(1,8):
    print(m_path + '/ground_%d.png')
    S=(m_path + '/ground_%d.png')%k
    disparity_value_map = cv2.imread(S,-1)
    S=(m_path + '/left_%d.png')%k
    left_image=cv2.imread(S,0)
    S=(m_path + '/right_%d.png')%k
    right_image=cv2.imread(S,0)

    disparity_value_map=(disparity_value_map.astype(np.float)/256.0)
    np.savetxt("Disparity_values_example.txt",disparity_value_map,fmt = '%d')
    patch_positions=[]
    max_disp = (int(disparity_value_map.max()))

    mean_left = (np.mean(left_image))
    std_left = (np.std(left_image))

    left_image = np.float32((left_image - mean_left)/std_left)

    mean_right = (np.mean(right_image))
    std_right = (np.std(right_image))

    right_image = np.float32((right_image - mean_right)/std_right)

    result = Image.fromarray((left_image).astype(np.float32))
    strl = "normalized_left_example_%d.tiff"%k
    result.save(strl)
    result = Image.fromarray((right_image).astype(np.float32))
    strr = "normalized_right_example_%d.tiff"%k
    result.save(strr)

    for i in range(100, img_ht - height):
        for j in range(width, img_wdt - width-1):

            if(disparity_value_map[i,j]>0):
                disp = int(round(disparity_value_map[i,j]))
                offset=choice([-1,1])
                q=j-disp+offset*randint(4,10)
                q_2 = j-disp
                if(q > width and q < (img_wdt - width) and q_2 >width and q_2 < (img_wdt - width) ):
                    left_patch = left_image[i-height:i+height+1,j-width:j+width+1]
                    result = Image.fromarray((left_patch).astype(np.float32))
                    s=(m_path + '../ReferencePatches/%d.tiff')%counter
                    result.save(s)

                    right_patch_negative=right_image[i-height:i+height+1,q-width:q+width+1]
                    result = Image.fromarray((right_patch_negative).astype(np.float32))
                    s=(m_path + '../NegativePatches/%d.tiff')%counter
                    result.save(s)
                    print(k,counter,i,j,disp,j-disp,q)

                    right_patch_positive=right_image[i-height:i+height+1,q_2-width:q_2+width+1]
                    result = Image.fromarray((right_patch_positive).astype(np.float32))
                    s=(m_path + '../PositivePatches/%d.tiff')%counter
                    result.save(s)

                    counter = counter+1;
                    if(counter == limit):
                        quit()
