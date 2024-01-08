import argparse
import tifffile as tf
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import pre_process
from utils.numpy_blob import np_contours
from utils.gaussian_blober import get_gaussian_blobs
from threading import Thread
from queue import Queue

# from pathlib import Path

def worker_function(func, result_queue, *args):
    result = func(*args)
    result_queue.put(result)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image_path', default='./data/blobs2.tif', type=str, help='path to the image')
    parser.add_argument(
        '--is_noisy', default=True, type=bool, help='is the image noisy?')
    parser.add_argument(
        '--thresh_by_area', default=False, type=bool, help='rejecting the blobs smaller than the average of the blob areas')
    parser.add_argument(
        '--kernel_size', type=int, default=7, help='kernel size for image denoising')
    parser.add_argument(
        '--min_sigma', type=int, default=6, help='min sigma for the Gaussian blob detector')
    parser.add_argument(
        '--max_sigma', type=int, default=25, help='max sigma for the Gaussian blob detector')
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    methods = ['Numpy_Contour', 'Difference_of_Gaussian', 'Laplasian_of_Gaussian']
    args = parse_args()
    # read the test image and preprocess it
    img = tf.imread(args.image_path)
    processed_img = pre_process(img, args.is_noisy, kernel_size=args.kernel_size)

    # %% create threads for multithreading
    result_queue = Queue()
    thread_list = [[]]*3
    # Create threads for each function
    thread_list[0] = Thread(target=worker_function, args=(np_contours, result_queue, processed_img, args.thresh_by_area), 
                                                    kwargs={}, daemon=True, name="np_thread", group=None)
    thread_list[1] = Thread(target=worker_function, args=(get_gaussian_blobs, result_queue, processed_img, args.min_sigma, args.max_sigma), 
                                                    kwargs={}, daemon=True, name="dog_thread", group=None)
    thread_list[2] = Thread(target=worker_function, args=(get_gaussian_blobs, result_queue, processed_img, args.min_sigma, args.max_sigma, methods[2]), 
                                                    kwargs={}, daemon=True, name="log_thread", group=None)

    [thread.start() for thread in thread_list]
    [thread.join()  for thread in thread_list]


    # %% download the results
    blobs = [[]]*len(thread_list)
    returned_methods = [[]]*len(thread_list)
    counter = 0
    while not result_queue.empty():
        blobs[counter], returned_methods[counter] = result_queue.get()
        counter += 1

    # %% plot the results
    fig, ax = plt.subplots(1,3)
    for blobs_ind, (method, blob) in enumerate(zip(returned_methods, blobs)):
        ax[blobs_ind].imshow(processed_img, cmap=plt.cm.gray)
        for i, sub_blob in enumerate(blob):
            if method == 'Numpy_Contour':
                ax[blobs_ind].plot(sub_blob[:, 1], sub_blob[:, 0], linewidth=3)
                x, y = np.mean(sub_blob[:, 1]), np.mean(sub_blob[:, 0])
            else:
                y, x, r = sub_blob
                c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
                ax[blobs_ind].add_patch(c)
            ax[blobs_ind].text(x, y, str(i), color='red', fontsize = 12) 
        ax[blobs_ind].set_axis_off()
        ax[blobs_ind].set_title(f"{method=} detected {len(blob)} blobs")
    plt.tight_layout()
    plt.show()
    print('')
