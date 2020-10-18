import cv2 
import scipy.ndimage.measurements as m 
import numpy as np 
import imageio 
import os 

def draw_contours(bground, mask, gtruth=None, contour_thickness=2):
    overlay = bground
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    # print(np.unique(np.uint8(mask)), np.unique(np.uint8(gtruth)))
    _, binary_pred = cv2.threshold(np.uint8(mask * 255), 127, 255, cv2.THRESH_BINARY)
    # print(binary_pred.dtype, binary_pred.shape, np.unique(binary_pred))
    contours_pred, _ = cv2.findContours(binary_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_pred, -1, (255, 20, 147), contour_thickness) # pink contours stand for prediction

    if gtruth is not None:
        _, binary_gtruth = cv2.threshold(np.uint8(gtruth * 255), 127, 255, cv2.THRESH_BINARY)
        contours_gtruth, _ = cv2.findContours(binary_gtruth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_gtruth, -1, (0, 255, 0), contour_thickness) # green contours stand for ground truth
        # cv2.putText(overlay, '{:.8f}'.format(IoU(np.uint8(mask), np.uint8(gtruth))), (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    return overlay


def draw_many_slices(slices_in_order, predictions_in_order, labels_in_order, contour_thickness=2):
    overlays_in_order = []
    for i in range(len(slices_in_order)):
        overlay = draw_contours(slices_in_order[i], predictions_in_order[i], labels_in_order[i], contour_thickness)
        overlays_in_order.append(overlay)
    return overlays_in_order


def remove_fragment(volume_in):
    volume, _ = m.label(volume_in)
    labels, count = np.unique(volume, return_counts=True)

    label_keep = np.argsort(count)[-2]

    volume[volume != label_keep] = 0
    volume[volume != 0] = 1
    # print(np.unique(volume))

    return volume

def imwrite(predictions_in_order, overlays_in_order, output_path):
    mask_path = output_path+'mask/'
    overlay_path = output_path+'overlay/'
    # print(mask_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
        os.makedirs(overlay_path)
    for i in range(len(predictions_in_order)):
        # print(predictions_in_order[i].shape)
        imageio.imwrite(mask_path+str(i).zfill(4)+'.png', predictions_in_order[i]*255)
        imageio.imwrite(overlay_path+str(i).zfill(4)+'.png', overlays_in_order[i])

