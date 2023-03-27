import pydicom
import numpy as np
import itertools
import operator
import ast
# from PIL import Image
import time

import utils.image_utils as imgu


from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut, apply_windowing, apply_voi

############################################################################################
## OLD NAMING:
############################################################################################

def getOriginalImageFromDicom(dicom):
    return img_original(dicom)


def getOriginalImageFromFileName(fn):
    return img_original(fn)


def getNormalizedImageFromDicom(dicom, ymin=0, ymax=255):
    return img_norm(dicom, ymin=ymin, ymax=ymax)


def getNormalizedImageFromFileName(fn, ymin=0, ymax=255):
    return img_norm(fn, ymin=ymin, ymax=ymax)


def getMaskImgFromDicom(dicom):
    return mask_img(dicom)


############################################################################################
# Main functions:
############################################################################################

def img_original(fn_or_dicom):
    # Corrects the encoding (MONOCHROME) and applies the rescale slope and intercept
    # Using new functions from pydicom library

    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))
        
    img = apply_modality_lut(dicom.pixel_array, dicom).astype(int)
    
    inverse_pls = False
    if 'PresentationLUTShape' in dicom:
        if dicom.PresentationLUTShape == 'INVERSE':
            img = apply_windowing(img, dicom).astype(int)
            inverse_pls = True

    if inverse_pls:
        img = apply_voi(img, dicom) ## no lut
    else:
        img = apply_voi_lut(img, dicom)
            
    if 'PhotometricInterpretation' in dicom:
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img

    # Some images have 3 channels (error images):
    if img.ndim > 2:
        print('WARNING: Image with more than 1 channel. Applying rgb2gray transformation')
        img = imgu.rgb2gray(img)

    return img.astype(int)


def img_norm(fn_or_dicom, ymin=0, ymax=255):
    # Normalizes the image to the range [ymin,ymax] (default [0,255])
    
    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))

    img = img_original(dicom)

    # Windowing: Already performed in img_original by function apply_voi_lut/apply_voi
    c = (img.min() + img.max())/2
    w = img.max() - img.min()
        
    imgNorm = ((img - (c - 0.5))/(w-1) + 0.5)*(ymax-ymin) + ymin

    imgNorm[imgNorm > ymax] = ymax
    imgNorm[imgNorm < ymin] = ymin
    
    return imgNorm.astype(int)




def img_clean(fn_or_dicom, returnMask=False, ymin=0, ymax=255):
    # Computes mask, normalizes the image (default [0,255]), and crops it to the mask

    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))

    img = img_norm(dicom, ymin=ymin, ymax=ymax)
    mask = mask_img(dicom)
    
    img, mask = imgu.img_cropped_to_mask(img=img, mask=mask, prctThr=0.1)

    if returnMask:
        return img, mask

    return img


def mask_img(dicom_or_imgOrig):
    # Extracts a mask with (image: 1, else: 0) based on the histogram of the image

    maskImg = mask_bgfg(dicom_or_imgOrig)

    # mask_bgfg assigns: {bg: -1, image: 0, fg: 1}
    maskImg[maskImg==0] = 2
    maskImg[maskImg<2] = 0
    maskImg[maskImg==2] = 1

    # if the mask is empty, we return a full mask:
    if maskImg.max() == 0:
        maskImg[maskImg == 0] = 1

    return maskImg.astype(np.uint8)


def mask_ROI(dicom_or_imgOrig, ROI_shape):
    # Extracts a mask with (ROI: 1, else: 0)

    if isinstance(dicom_or_imgOrig, pydicom.dataset.FileDataset):
        imgOrig = img_original(dicom_or_imgOrig)
    elif isinstance(dicom_or_imgOrig, np.ndarray):
        imgOrig = dicom_or_imgOrig
    else:
        raise Exception('dicom_or_imgOrig is not a dicom object or numpy ndarray. Type(dicom_or_imgOrig)={}'.format(type(dicom_or_imgOrig)))

    
    # Parse ROI_shape
    d = ast.literal_eval(ROI_shape)
    
    row_start = int(min(d['start']['y'], d['end']['y']))
    row_end = int(max(d['start']['y'], d['end']['y']))
    
    col_start = int(min(d['start']['x'], d['end']['x']))
    col_end = int(max(d['start']['x'], d['end']['x']))
    
    
    mask_ROI = np.zeros(imgOrig.shape, dtype=np.uint8)
    mask_ROI[row_start:row_end, col_start:col_end] = 1

    return mask_ROI.astype(np.uint8)


def isxray(dicom):

    # Based on the SOPClassUID:
    if 'SOPClassUID' in dicom:
        if str(dicom.SOPClassUID).startswith('1.2.840.10008.5.1.4.1.1.'):
            vals = str(dicom.SOPClassUID).split('.')
            if vals[9] in ['1', '12']:
                return True
    
    return False



############################################################################################
# Auxiliary functions:
############################################################################################

def idxs_longest_sequence_of_zeros(A):
    # Finds the longest sequence of quasi 0s in a list A
    # Returns initial and end indices

    if not len(A):
        return -1, -1

    thr = 0.01*sum(A)/len(A) # 1% of the unif distributed data
    A[A<thr] = 0

    if np.argmax(A==0):
        lz = (list(y) for (x,y) in itertools.groupby((enumerate(A)),operator.itemgetter(1)) if x == 0)
        i = max(lz, key=len)
        return i[0][0], i[-1][0]
    else:
        return -1, -1


def hist_min_max(img):
    # Finds min and max values corresponding to the "informative" image based on the histogram
    # It finds the blocks of hist-bins with 0s
    # The objective is to discard low/high values that do not correspond to the image, i.e. are artificially added
    
    vals = img.ravel()

    #vals = vals[::1000] # x100 faster
    
    minImg = vals.min()
    maxImg = vals.max()
    
    minZeros = (0.005*(maxImg - minImg)).astype(int)
    
    if minZeros > 0:
    
        h, b = np.histogram(vals, bins=maxImg-minImg, range=(minImg,maxImg))
        
        p50 = np.percentile(vals, 50)
        i_p50 = np.argmax(b>p50)

        h_inf = h[:i_p50]
        b_inf = b[:i_p50]
        h_sup = h[i_p50:]
        b_sup = b[i_p50:-1]

        hi_min_idx, hi_max_idx = idxs_longest_sequence_of_zeros(h_inf)
        hs_min_idx, hs_max_idx = idxs_longest_sequence_of_zeros(h_sup)

        if (hi_max_idx - hi_min_idx) >= minZeros:
            minImg = b_inf[hi_max_idx]
            
        if (hs_max_idx - hs_min_idx) >= minZeros:
            maxImg = b_sup[hs_min_idx]
    

    return minImg, maxImg


def mask_bgfg(dicom_or_pixelArray):
    # Extracts a mask with (background: -1, foregraound: +1, image: 0) based on the histogram of the image

    if isinstance(dicom_or_pixelArray, pydicom.dataset.FileDataset):
        img = dicom_or_pixelArray.pixel_array
    elif isinstance(dicom_or_pixelArray, np.ndarray):
        img = dicom_or_pixelArray
    else:
        raise Exception('dicom_or_pixelArray is not a dicom object or numpy ndarray. Type(dicom_or_pixelArray)={}'.format(type(dicom_or_pixelArray)))

    minImg, maxImg = hist_min_max(img)

    maskBgFg = np.zeros(img.shape, dtype=np.int8)
    maskBgFg[img <= minImg] = -1
    maskBgFg[img >= maxImg] = 1

    return maskBgFg



############################################################################################
# DEPRECATED functions:
############################################################################################

def img_original_float(fn_or_dicom):
    print('WARNING! - DEPRECATED function img_original_float()')
    print('In case you need this, replicate latest version of img_original()')
    # Corrects the encoding (MONOCHROME) and applies the rescale slope and intercept

    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))

    img = dicom.pixel_array.astype(float)

    if 'PhotometricInterpretation' in dicom:
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            img = img.max() - img

    if 'RescaleIntercept' in dicom:
        ri = dicom.RescaleIntercept
        if not isinstance(ri, (int, float)):
            ri = 0
    else:
        ri = 0
    
    if 'RescaleSlope' in dicom:
        rs = dicom.RescaleSlope
        if not isinstance(rs, (int, float)):
            rs = 1
    else:
        rs = 1

    img = img*rs + ri

    # Some images have 3 channels (error images):
    if img.ndim > 2:
        print('WARNING: Image with more than 1 channel. Applying rgb2gray transformation')
        img = imgu.rgb2gray(img)

    return img


def img_norm_float(fn_or_dicom, ymin=0.0, ymax=1.0):
    print('WARNING! - DEPRECATED function img_norm_float()')
    print('In case you need this, replicate latest version of img_norm()')
    # Normalizes the image to the range [ymin,ymax] (default [0.0,1.0])
    
    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))


    img = img_original_float(dicom)

    if 'WindowCenter' in dicom:
        if dicom.data_element('WindowCenter').VM > 1:
            c = dicom.WindowCenter[0]
            w = dicom.WindowWidth[0]
        else:
            c = dicom.WindowCenter
            w = dicom.WindowWidth
    else:
        c = img.min() + (img.max() - img.min())/2
        w = img.max() - img.min()

    imgNorm = ((img - (c - 0.5))/(w-1) + 0.5)*(ymax-ymin) + ymin

    imgNorm[imgNorm > ymax] = ymax
    imgNorm[imgNorm < ymin] = ymin

    return imgNorm


def img_clean_float(fn_or_dicom, returnMask=False, ymin=0.0, ymax=1.0):
    print('WARNING! - DEPRECATED function img_clean_float()')
    print('In case you need this, replicate latest version of img_clean()')
    # Computes mask, normalizes the image (default [0.0,1.0]), and crops it to the mask

    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception('fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))

    img = img_norm_float(dicom, ymin=ymin, ymax=ymax)
    mask = mask_img(dicom)
    img, mask = imgu.img_cropped_to_mask(img=img, mask=mask, prctThr=0.1)

    if returnMask:
        return img, mask

    return img