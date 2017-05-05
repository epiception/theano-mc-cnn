## THEANO-MC-CNN ##

Simple Theano Implementation of https://arxiv.org/abs/1510.05970 (fast architecture) for creating Depth Maps by training a Siamese Convolutional Network on Image Patches. 

Original Source Code (in Torch/Lua) and results can be found [here](https://github.com/jzbontar/mc-cnn).



---

## Requirements
* Theano
* OpenCV
* Numpy
* Python Imaging Library (PIL)
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [glob](https://docs.python.org/2/library/glob.html)

---

## Usage

    $git clone https://github.com/epiception/theano-mc-cnn.git
    $cd ~/path/to/theano-mc-cnn
    
**Create Dataset:**
    
    $python ./Dataset_creation/Patches_Extractor.py
The maximum number of patches has been set to 4,00,000 patches.

**Training**
    
    $python ./Model/MC_CNN_Fast.py
Weights will be saved for each layer and epoch in the Weights folder in the following format:      
`weights_epoch_{epoch_number}_layer_{layer_no}.npy`. 

Training and Test Accuracy can be viewed in the `Epoch_stats_training.txt` and `Epoch_stats_testing.txt` files in the Model folder respectively.

To load or retrain from weights from a specific epoch, change the value of the *start_epoch* variable in `MC_CNN_Fast.py`. 
It will load the weights from the previous epoch and continue training.

**Finding Depth Map**
To save Disparity Map after training:
    
    $python ./Deploy/Deploy_Depth.py -path_to_left_image -path_to_right_image -disparity_range
For instructions
    
    $python ./Deploy/Deploy_Depth.py -h/--help 
This saves the Disparity values in `Cost_grid_map.txt` and displays the Disparity Map
(Since the dot product layer is not multi-threaded and runs in single core, this will be slow)


