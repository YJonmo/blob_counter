import numpy as np
from skimage.feature import blob_dog, blob_log

def get_gaussian_blobs(img:np.ndarray, min_sigma:int=5, max_sigma:int=30, 
						method_name:str='Difference_of_Gaussian')->list[np.ndarray, str]:
	blobs = np.empty(0)
	if method_name=='Difference_of_Gaussian':
		# %% using the the Difference of Gaussian
		blobs = blob_dog(img , min_sigma=min_sigma, sigma_ratio=1.6, max_sigma=max_sigma)
	else:
		# %% using the the Laplasian of Gaussian
		blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma, 
							num_sigma=10, threshold=.5, overlap=0.1)
	
	blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
	
	return blobs, method_name
