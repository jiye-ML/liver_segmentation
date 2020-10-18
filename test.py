import os

import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import utils.metrics as m
from model.unet import UNet
from utils.preprocessing_utils import nii2labels, nii2slices
from utils.surface import Surface
from utils.test_utils import (draw_contours, draw_many_slices, imwrite,
                              remove_fragment)

if __name__ == '__main__':
    LITS_data_path = 'LITS/'
    model_path = 'checkpoints/liver_segmentation_U-Net_on_LITS_datasetiter_300000.pth'

    prediction_path = 'results/'

    device = torch.device('cuda:0')
    model = UNet(1, 2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sm = nn.Softmax(dim=1)

    idx_list = []
    dice_list = []
    iou_list = []
    voe_list = []
    rvd_list = []
    assd_list = []
    msd_list = []

    for i in range(31):
        print(i)
        idx_list.append(i)
        volume = nib.load(LITS_data_path+'volume-'+str(i)+'.nii').get_fdata()
        vxlspacing = nib.load(LITS_data_path+'volume-'+str(i)+'.nii').header.get_zooms()[:3]
        segmentation = nib.load(LITS_data_path+'segmentation-'+str(i)+'.nii').get_fdata()
        slices_in_order = nii2slices(volume, 0, 400)
        labels_in_order = nii2labels(segmentation)

        predictions_in_order = []

        for slice in slices_in_order:
            slice = torch.from_numpy(slice).float() / 255.
            output = model(slice.unsqueeze(0).unsqueeze(0).to(device))
            prediction = sm(output)
            _, prediction = torch.max(prediction, dim=1)
            prediction = prediction.squeeze(0).cpu().detach().numpy().astype(np.uint8)
            predictions_in_order.append(prediction)
        
        overlay_in_orders = draw_many_slices(slices_in_order, predictions_in_order, labels_in_order)
        imwrite(predictions_in_order, overlay_in_orders, prediction_path+str(i)+'/')

        # metrics
        v_prediction = np.stack(predictions_in_order).astype(np.uint)
        v_label = np.stack(labels_in_order).astype(np.uint)

        dice = m.dc(v_prediction, v_label)
        dice_list.append(dice)
        print('DICE:', dice)

        iou = m.jc(v_prediction, v_label)
        iou_list.append(iou)
        print('IOU:', iou)

        voe = 1 - iou
        voe_list.append(voe)
        print('VOE:', voe)

        rvd = m.ravd(v_prediction, v_label)
        rvd_list.append(rvd)
        print('RVD:', rvd)

        if np.count_nonzero(v_prediction) ==0 or np.count_nonzero(v_label)==0:
            assd = 0
            msd = 0
        else:
            evalsurf = Surface(v_prediction,v_label,physical_voxel_spacing = vxlspacing,mask_offset = [0.,0.,0.], reference_offset = [0.,0.,0.])
            assd = evalsurf.get_average_symmetric_surface_distance()
            msd = m.hd(v_label,v_prediction,voxelspacing=vxlspacing)

        assd_list.append(assd)
        print('ASSD:', assd)

        msd_list.append(msd)
        print('MSD:', msd)

    metric_data = {'dice': dice_list, 'iou': iou_list, 'voe': voe_list, 'rvd': rvd_list, 'assd': assd_list, 'msd': msd_list}
    csv_data = pd.DataFrame(metric_data, idx_list)
    csv_data.to_csv('metrics.csv')

