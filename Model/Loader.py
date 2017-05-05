import numpy as np
import os
import theano

from PIL import Image
from tqdm import tqdm

mPath = m_path = os.path.dirname(os.path.abspath("Loader.py"))
refPath = mPath + '/ReferencePatches/'
PosPath = mPath + '/PositivePatches/'
NegPath = mPath + '/NegativePatches/'

def loadAllImages(n):
	imgsList = []
	all_ref = np.zeros((n,1,9,9))
	all_pos = np.zeros_like(all_ref)
	all_neg = np.zeros_like(all_ref)

	print("Loading Patches")
	for id in tqdm(range(n)):
		nameR = refPath + str(id) + ".tiff"
		nameP = PosPath + str(id) + ".tiff"
		nameN = NegPath + str(id) + ".tiff"

		imgR = np.asarray(Image.open(nameR), dtype = np.float32)
		imgP = np.asarray(Image.open(nameP), dtype = np.float32)
		imgN = np.asarray(Image.open(nameN), dtype = np.float32)

		all_ref[id,0,:,:] = imgR
		all_pos[id,0,:,:] = imgP
		all_neg[id,0,:,:] = imgN

	return [all_ref,all_pos,all_neg]
