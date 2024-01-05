import numpy as np
from utils.utils import *

def np_contours(img:np.ndarray, thresh_by_area:bool=False, thresh:int=128,
				method_name:str='Numpy_Contour')->list[np.ndarray, str]:

	vertex_connect_high = 'high'
	segments = get_contour_segments(img, float(thresh), 
									vertex_connect_high == 'high', mask=None)

	contours = assemble_contours(segments)
	areas = np.zeros(len(contours), dtype=np.float32)

	correct_ind = np.ones(len(areas), dtype=bool)

	if thresh_by_area:
		# make a rough estimation of the contour areas
		for index, contour in enumerate(contours):
			areas[index] =  max(contours[index][:,0]) - min(contours[index][:,0])
			areas[index] *= max(contours[index][:,1]) - min(contours[index][:,1])
		# remove the countours that correspond to areas samller than the mean of the all areas
		correct_ind = np.where(areas > np.mean(areas))[0]


	return np.array(contours, dtype=object)[correct_ind], method_name