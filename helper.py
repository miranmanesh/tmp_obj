import matplotlib.pyplot as plt
import numpy as np

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    if nrow == 1:
        for i in range(len(img_array)):
            plots[i].imshow(img_array[i])
    else:
        for i in range(len(img_array)):
            plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce
def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks, n_class):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])
    colors = colors[:n_class]

    
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255 
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8) 

instance_masks_colors = np.asarray([list(np.random.choice(range(256), size=3)) for _ in range(200)])

def instancemasks_to_colorimg(masks):
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    colors = instance_masks_colors[:masks.shape[0]]

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)



def masks_to_colorimg2(masks):
    colors = np.asarray([(0, 0, 0),(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56),
                       (101, 58, 14), (22, 107, 10), (100, 52, 5), (10, 172, 128),(6, 234, 232), (16, 19, 250), 
                        (101, 158, 4), (10, 10, 10), (10, 12, 175), (201, 72, 28),(156, 134, 32), (60, 94, 156),
                        (161, 168, 64), (110, 101, 110), (10, 112, 1), (1, 172, 128),(56, 134, 3), (6, 9, 6),
                        (11, 15, 14), (100, 10, 1), (10, 2, 15), (21, 12, 18),(16, 14, 13), (160, 194, 156),
                        (11, 115, 14), (10, 210, 210), (90, 91, 17), (21, 7, 128),(16, 134, 2), (6, 4, 15)]
                       )
    colors = colors[:len(masks)]


    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)



def masks_to_colorimg3(masks):
    colors = np.asarray([(0, 0, 0), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56),
                       (101, 58, 14), (22, 107, 10), (100, 52, 5), (10, 172, 128),(6, 4, 232), (16, 19, 250), 
                        (101, 158, 4), (10, 10, 10), (10, 12, 175), (201, 72, 28),(156, 134, 32), (60, 94, 156),
                        (161, 168, 64), (110, 101, 110), (10, 112, 1), (1, 172, 128),(56, 134, 3), (6, 9, 6),
                        (11, 15, 14), (100, 10, 1), (10, 2, 15), (21, 12, 18),(16, 14, 13), (160, 194, 156),
                        (11, 115, 14), (10, 210, 210), (90, 91, 17), (21, 7, 128),(16, 134, 2), (6, 4, 15)]
                       )
    colors = colors[:len(masks)]


    colorimg = np.ones((masks.shape[0], masks.shape[1], 3), dtype=np.float32) * 255
    height, width = masks.shape

    for y in range(height):
        for x in range(width):
            colorimg[y,x,:] = colors[masks[y,x]]

    return colorimg.astype(np.uint8)
