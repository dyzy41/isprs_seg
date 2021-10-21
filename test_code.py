import numpy as np
import yimage
import cv2

def label_mapping(label_im):
    # colorize = np.zeros([2, 3], dtype=np.int64)
    colorize = np.array([[255, 255, 255],
                         [255, 0, 0],
                         [255, 255, 0],
                         [0, 255, 0],
                         [0, 255, 255],
                         [0, 0, 255],
                         [64, 64, 128],
                         [64, 0, 128],
                         [64, 64, 0],
                         [0, 128, 192],
                         [255, 0, 0]
                         ])
    # colorize[0, :] = [128, 128, 0]
    # colorize[1, :] = [255, 255, 255]
    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label

if __name__ == '__main__':
    p = r'Y:\private\dongsj\0sjcode\code0906_vaiseg\vai_data\train_gt\top_mosaic_09cm_area1.tif'
    image = yimage.io.read_image(p)
    image_vis = label_mapping(image)
    cv2.imwrite('test_gt.tif', image_vis)