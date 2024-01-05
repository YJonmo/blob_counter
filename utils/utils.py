
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import deque

def pre_process(img:np.ndarray, is_noisy:bool=False, kernel_size:int=7, thresh:int=128)-> np.ndarray:
    # img = img[:,:,0]
    img = img.astype(np.float64)
    if img.ndim > 2:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    binary_img = np.ma.masked_array(img > thresh)

    if is_noisy:
        # img = img/255
        # img = img[:,:] < 0.5
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        # kernel_dial = np.zeros((kerargs.kernel_size_siz, kerargs.kernel_size_siz), dtype=bool)
        # kernel_dial[args.kernel_size//2, :] = 1
        # kernel_dial[:, args.kernel_size//2] = 1
        padded = np.pad(binary_img, kernel_size//2)
        windows = sliding_window_view(padded, window_shape=kernel.shape)
        eroded = np.all(windows | ~kernel, axis=(-1, -2))
        dilated = conv2d(eroded, kernel)
        img *= dilated
    return img


def assemble_contours(segments):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}

    for from_point, to_point in segments:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degenerate vertex will be picked up later by neighboring
        # squares.
        if from_point == to_point:
            continue

        tail, tail_num = starts.pop(to_point, (None, None))
        head, head_num = ends.pop(from_point, (None, None))

        if tail is not None and head is not None:
            # We need to connect these two contours.
            if tail is head:
                # We need to closed a contour: add the end point
                head.append(to_point)
            else:  # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # Remove tail from the detected contours
                    contours.pop(tail_num, None)
                    # Update starts and ends
                    starts[head[0]] = (head, head_num)
                    ends[head[-1]] = (head, head_num)
                else:  # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # Remove head from the detected contours
                    starts.pop(head[0], None)  # head[0] can be == to_point!
                    contours.pop(head_num, None)
                    # Update starts and ends
                    starts[tail[0]] = (tail, tail_num)
                    ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            # We need to add a new contour
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:  # tail is not None
            # tail first element is to_point: the new segment should be
            # prepended.
            tail.appendleft(from_point)
            # Update starts
            starts[from_point] = (tail, tail_num)
        else:  # tail is None and head is not None:
            # head last element is from_point: the new segment should be
            # appended
            head.append(to_point)
            # Update ends
            ends[to_point] = (head, head_num)

    return [np.array(contour) for _, contour in sorted(contours.items())]

def conv2d(a, f, *args):
    # https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
    # s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    s = f.shape + a.shape
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    # padded = np.pad(a, f.shape[0]//2)

    return np.einsum('ij,ijkl->kl', f, subM)


def npy_isnan(x):
    return np.isnan(x)

def _get_fraction(from_value, to_value, level):
    if to_value == from_value:
        return 0
    return (level - from_value) / (to_value - from_value)

def get_contour_segments(array, level, vertex_connect_high, mask=None):
    segments = []

    use_mask = mask is not None
    square_case = 0

    for r0 in range(array.shape[0] - 1):
        for c0 in range(array.shape[1] - 1):
            r1, c1 = r0 + 1, c0 + 1

            if use_mask and not (mask[r0, c0] and mask[r0, c1] and mask[r1, c0] and mask[r1, c1]):
                continue

            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]

            if npy_isnan(ul) or npy_isnan(ur) or npy_isnan(ll) or npy_isnan(lr):
                continue

            square_case = 0
            if ul > level:
                square_case += 1
            if ur > level:
                square_case += 2
            if ll > level:
                square_case += 4
            if lr > level:
                square_case += 8

            if square_case in [0, 15]:
                continue

            top = r0, c0 + _get_fraction(ul, ur, level)
            bottom = r1, c0 + _get_fraction(ll, lr, level)
            left = r0 + _get_fraction(ul, ll, level), c0
            right = r0 + _get_fraction(ur, lr, level), c1

            if square_case == 1:
                segments.append((top, left))
            elif square_case == 2:
                segments.append((right, top))
            elif square_case == 3:
                segments.append((right, left))
            elif square_case == 4:
                segments.append((left, bottom))
            elif square_case == 5:
                segments.append((top, bottom))
            elif square_case == 6:
                if vertex_connect_high:
                    segments.append((left, top))
                    segments.append((right, bottom))
                else:
                    segments.append((right, top))
                    segments.append((left, bottom))
            elif square_case == 7:
                segments.append((right, bottom))
            elif square_case == 8:
                segments.append((bottom, right))
            elif square_case == 9:
                if vertex_connect_high:
                    segments.append((top, right))
                    segments.append((bottom, left))
                else:
                    segments.append((top, left))
                    segments.append((bottom, right))
            elif square_case == 10:
                segments.append((bottom, top))
            elif square_case == 11:
                segments.append((bottom, left))
            elif square_case == 12:
                segments.append((left, right))
            elif square_case == 13:
                segments.append((top, right))
            elif square_case == 14:
                segments.append((left, top))

    return segments