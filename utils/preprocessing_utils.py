import numpy as np 

def WL(data, WC, WW): 
    # WC: 窗位     WW：窗宽
    min = (2*WC - WW) / 2.0
    max = (2*WC + WW) / 2.0
    # print(max, min)
    idx_max = np.where(data > max)
    idx_min = np.where(data < min)
    idx_in = np.where((data >= min) & (data <= max))
    
    data = (data -min) * 254 / (max - min)
    data[idx_max] = 255
    data[idx_min] = 0
    return data

def nii2slices(volume, WC, WW):
    slices_in_order = []
    for i in range(volume.shape[-1]):
        slice = volume[:,:,i]
        slice_post = WL(slice, WC, WW)
        slice_post = slice_post.astype(np.uint8)
        slices_in_order.append(slice_post)
    return slices_in_order

def nii2labels(volume):
    labels_in_order = []
    for i in range(volume.shape[-1]):
        label = volume[:,:,i]
        label[label>0] = 255
        labels_in_order.append(label)
    return labels_in_order