import torch
from torchvision import transforms
import numpy as np
from PIL import Image

import utils.image_utils as imgu
import utils.report_utils as ru
import model_20210820_XNet38MS.XNet38_urg as XNet38_urg



def prepare_data(image_original):
    # PREAPARE DATA:
    mean_imgs = torch.tensor([0.491])
    std_imgs  = torch.tensor([0.271])

    # We need 3 sizes:
    dict_images = dict()
    for imgSize in [299, 512, 1024]:
        # Resize to square image:
        newSize = imgSize
        image = imgu.img_resize(image_original, shapeImgOut=(1024, 1024), resizeMethod='padding', resampleMethod='BILINEAR')

        transform = transforms.Compose([transforms.Resize(newSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean_imgs,
                                                            std=std_imgs)])

        # Transform to Pillow image
        image = Image.fromarray(image.astype(np.uint8))  # Image only accepts uint8

        # Apply transformations:
        image = transform(image)

        # Expand dimension:
        image = image.unsqueeze(0)

        dict_images[imgSize] = image

    return dict_images


def build_model():
    # BUILD MODEL:

    # The model for XNet38MS requires 3 XNet38 models and 3 FCS modules
    dict_models = dict()

    for imgSize in [299, 512, 1024]:   

        model_folder_weights = './model_20210820_XNet38MS/model_weights/direct_multi93_is' + str(imgSize) + '_Rv10_pre00_imagenet'

        # Build model:
        model = XNet38_urg.XNet38_urg()
        # Load model state:
        model.load_state_dict(model_folder_weights)

        print('INFO:', 'Model [', model.__class__.__name__, '] ready')

        model.cuda()
        model.eval()

        dict_models[imgSize] = model

    return dict_models


def test(image, model):
    # TEST MODEL ON IMAGE

    # Now this is done when building the model
    # model.cuda()

    # EVALUATION:
    # Now this is done when building the model
    # model.eval()

    # Only forward
    with torch.no_grad():
        logits_multi, logits_urg = model(image.cuda())

    # index [0] because there is only one input array
    probs_multi = logits_multi[0].cpu().sigmoid()
    probs_urg = logits_urg[0].cpu().softmax(0)

    return probs_multi, probs_urg


def main(image, dict_models):
    # Given an image, returns the prediction

    dict_images = prepare_data(image)

    list_probs_multi = []
    list_probs_urg = []

    for imgSize in dict_images.keys():
        probs_multi, probs_urg = test(dict_images[imgSize], dict_models[imgSize])
        list_probs_multi.append(probs_multi)
        list_probs_urg.append(probs_urg)

    probs_multi_MS = torch.stack(list_probs_multi, 0).mean(0).numpy().ravel()

    urgencies = [1,1,1,1,2,1,1,1,2,2,2,1,1,2,1,2,1,1,1,2,2,3,1,2,2,2,2,2,3,3,3,2,2,1,3,1,2,1]
    urgency_text = ['normal','non-urgent','urgent','critical']
    pred_urg_MS = urgency_text[int(np.max((probs_multi_MS>0.5)*urgencies))]
    
    # Build report:
    report = ru.build_report(probs_multi_MS, pred_urg_MS)

    return report
