import sys
sys.path.insert(0, './pybind/build')
import cpp_blob_counter as CPC
from matplotlib import pyplot as plt
detector = CPC.PyCppBlobCounter('./data/blobs1.tif', False, False, False, False, 200)

detected_blobs = detector.detect_blobs()
plt.imshow(detected_blobs)
plt.show()
print('Done!')