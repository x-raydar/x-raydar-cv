import numpy as np
from PIL import Image
from skimage.transform import rescale


############################################################################################
# Main functions:
############################################################################################

def img_cropped_to_mask(img, mask, prctThr=0):
    
    sumCols = np.sum(mask, axis=0)
    sumRows = np.sum(mask, axis=1)

    thrCols = prctThr*max(sumCols)
    thrRows = prctThr*max(sumRows)

    idxCol_min = min(np.where(sumCols > thrCols)[0])
    idxCol_max = max(np.where(sumCols > thrCols)[0])

    idxRow_min = min(np.where(sumRows > thrRows)[0])
    idxRow_max = max(np.where(sumRows > thrRows)[0])

    return img[idxRow_min:idxRow_max+1, idxCol_min:idxCol_max+1], mask[idxRow_min:idxRow_max+1, idxCol_min:idxCol_max+1]


def img_resize(img, shapeImgOut=1024, resizeMethod='padding', resampleMethod='BILINEAR'):
    # It resizes the input image to the size sizeImgOut with:
    # resizeMethod in {padding, span}
    # resampleMethod should be a method accepted by Image.resize:
    # (0) PIL.Image.NEAREST
    # (4) PIL.Image.BOX
    # (2) PIL.Image.BILINEAR
    # (5) PIL.Image.HAMMING
    # (3) PIL.Image.BICUBIC
    # (1) PIL.Image.LANCZOS

    if isinstance(shapeImgOut, int):
        shapeImgOut = (shapeImgOut, shapeImgOut)

    sizeImgOut = shapeImgOut[::-1]
    
    if resizeMethod == 'padding':
        divMax = max(img.shape[0]/shapeImgOut[0], img.shape[1]/shapeImgOut[1])
        sizeResize = (int(img.shape[1]/divMax), int(img.shape[0]/divMax))
    else:
        sizeResize = sizeImgOut

    typeImgIn = img.dtype
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize(sizeResize, resample=getattr(Image, resampleMethod))
    
    if resizeMethod == 'padding':
        out_img = Image.new('L', sizeImgOut)
        out_img.paste(img, ((sizeImgOut[0]-img.size[0])//2,(sizeImgOut[1]-img.size[1])//2))
    else:
        out_img = img

    return np.array(out_img).astype(typeImgIn)


def img_resize_float(img, shapeImgOut=1024):
    # It always uses resizeMethod='padding' and resampleMethod='BILINEAR':

    if isinstance(shapeImgOut, int):
        shapeImgOut = (shapeImgOut, shapeImgOut)

    if img.shape[0] == shapeImgOut[0] and img.shape[1] == shapeImgOut[1]:
        return img
    
    scale = min(shapeImgOut[0]/img.shape[0], shapeImgOut[1]/img.shape[1])

    img = rescale(img, scale, anti_aliasing=True, preserve_range=True).astype(img.dtype)

    padding = (shapeImgOut[0]-img.shape[0], shapeImgOut[1]-img.shape[1])
    
    img = np.pad(img, ((padding[0]//2, padding[0]-padding[0]//2),
                       (padding[1]//2, padding[1]-padding[1]//2)))

    return img


def is_chestxray(img, cutoff=0.5):
    # Take a single sample image and classify as chest X-ray true/false with hardcoded 8x8 filter (obtained from 1,000 images averaged)
    sample = img_resize(img, shapeImgOut=8, resizeMethod='span')

    filter8 = [[ 62,  74,  95, 143, 145,  98,  77,  68],
               [146, 139, 104, 129, 133, 100, 136, 148],
               [171, 139,  78, 125, 137,  77, 132, 171],
               [165, 131,  83, 143, 161,  91, 127, 166],
               [153, 129, 104, 166, 188, 129, 129, 157],
               [143, 140, 139, 187, 203, 166, 140, 148],
               [135, 176, 196, 212, 214, 193, 169, 139],
               [135, 200, 220, 226, 225, 216, 196, 140]]
        
    #print(corr2(filter8, sample))
    return (corr2(filter8, sample) >= cutoff)

############################################################################################
# Auxiliary functions:
############################################################################################

def corr2(a,b):
    a = a - (np.sum(a) / np.size(a))
    b = b - (np.sum(b) / np.size(b))

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum())
    return r


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])