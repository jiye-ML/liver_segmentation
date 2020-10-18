import os 
import shutil 
import nibabel as nib 
import glob
import imageio 
import numpy as np 
import multiprocessing
from utils.preprocessing_utils import WL, nii2slices, nii2labels


if __name__ == '__main__':
    LITS_data_path = 'data/LITS/'
    tr_path = 'data/tr/'
    ts_path = 'data/ts/'
    ts_no_label_path = 'data/ts2/'
    raw_path = 'raw/'
    label_path = 'label/'

    for i in range(31):
        print(i)
        volume = nib.load(LITS_data_path+'volume-'+str(i)+'.nii').get_fdata()
        segmentation = nib.load(LITS_data_path+'segmentation-'+str(i)+'.nii').get_fdata()
        slices_in_order = nii2slices(volume, 0, 400)
        labels_in_order = nii2labels(segmentation)
        for n in range(len(slices_in_order)):
            imageio.imwrite(ts_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(ts_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))
    
    for i in range(31, 131):
        print(i)
        volume = nib.load(LITS_data_path+'volume-'+str(i)+'.nii').get_fdata()
        segmentation = nib.load(LITS_data_path+'segmentation-'+str(i)+'.nii').get_fdata()
        slices_in_order = nii2slices(volume, 0, 400)
        labels_in_order = nii2labels(segmentation)
        for n in range(len(slices_in_order)):
            imageio.imwrite(tr_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            imageio.imwrite(tr_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))

    for i in range(0, 70):
        print(i)
        volume = nib.load(LITS_data_path+'test-volume-'+str(i)+'.nii').get_fdata()
        # segmentation = nib.load(LITS_data_path+'segmentation-'+str(i)+'.nii').get_fdata()
        slices_in_order = nii2slices(volume, 0, 400)
        # labels_in_order = nii2labels(segmentation)
        for n in range(len(slices_in_order)):
            imageio.imwrite(ts_no_label_path+raw_path+str(i)+'_'+str(n).zfill(4)+'.png', slices_in_order[n].astype(np.uint8))
            # imageio.imwrite(tr_path+label_path+str(i)+'_'+str(n).zfill(4)+'.png', labels_in_order[n].astype(np.uint8))
