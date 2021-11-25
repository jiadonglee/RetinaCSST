#%%
import numpy as np
from PIL import Image
import cv2
from astropy.io import fits
import torch
import torchvision
from torch.utils.data import DataLoader
from utils.tool import collate_fn
from CSST import CSSTimg
from tqdm import tqdm
from torchvision.ops import nms, remove_small_boxes
#%%

def load_retina(model_nme, model_raw, device=torch.device('cuda:0')):
    
    model_raw.load_state_dict(torch.load(model_nme))
    model_raw = model_raw.to(device)
    model_raw.eval()
    for param in model_raw.parameters():
        param.requires_grad = False
        
    return model_raw

def model_to_pred(dataloader, model, device=torch.device('cuda:0')):
    images, targets = [], []
    outputs = []

    for image, target in tqdm(dataloader):

        images+=list(img.to(device) for img in image)
        targets+=[{k:v for k, v in t.items()} for t in target]

        output = model(list(img.to(device) for img in image))
        outputs+=[{k:v.to(torch.device("cpu")) for k,v in t.items()} for t in output]
        del image, target
    return images, targets, outputs

def post_process_box(boxes, scores, labels, score_threshold=0.2, iou_threshold=0.2, min_size=5):
    idx_nms = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
    idx_rmv_small = remove_small_boxes(boxes=boxes, min_size=min_size)
    idx_score = np.nonzero(scores>score_threshold)
    
    idpst = np.intersect1d(idx_nms, idx_rmv_small)
    idpst = np.intersect1d(idpst, idx_score)
    
    return {'boxes':boxes[idpst], 'scores':scores[idpst], 'labels':labels[idpst]}

def pred_result(output, score=0.2, iou=0.3):
    output = post_process_box(
        output['boxes'], output['scores'], output['labels'], 
        score_threshold=score, iou_threshold=iou
    )
    cls_nme = np.array(['Galaxy' if i==0 else 'Star' for i in output['labels'] ])
    pred = np.c_[cls_nme, output['scores'], output['boxes']]
    return pred

def ground_truth(target):
    cls_nme = np.array(['Galaxy' if i==0 else 'Star' for i in target['labels'] ])
    gt = np.c_[cls_nme, target['boxes']]
    return gt

#%%
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model_ft = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model_dir = "/data/jdli/model/"
    model_nme = "Retina1026_mag20.bin"
    model     =  load_retina(model_dir+model_nme, model_ft, device=device)

    img_dir = "/data/jdli/C3test/img/"
    tgt_dir = "/data/jdli/C3test/target20/"
    c3val_data  = CSSTimg(img_dir, tgt_dir)

    dataloader = DataLoader(
        c3val_data, batch_size=8, shuffle=False, num_workers=8,
        collate_fn=collate_fn
    )

    images, targets, outputs = model_to_pred(
        dataloader, model, 
        device=torch.device('cuda:0')
        )

    gts   = [ground_truth(target) for target in targets]
    preds = [pred_result(output, score=0.25, iou=0.5) for output in outputs]

    gt_dir   = "/data/jdli/eval/ground-truth20/"
    pred_dir = "/data/jdli/eval/detection-results-rtn1026mag20/"
    jpg_dir  = "/data/jdli/eval/jpgs/"
    fits_dir = "/data/jdli/eval/fits/"

    """save results to file
    """
    for i in tqdm(range(len(c3val_data))):
        np.savetxt(
            gt_dir+'img_%d.txt'%i, gts[i], fmt="%s", delimiter=' '
            )
        np.savetxt(
            pred_dir+'img_%d.txt'%i, preds[i], fmt="%s", delimiter=' '
            )

        # img = np.array(images[i].cpu()).reshape(-1, 1152)
        # # cv2.imwrite(jpg_dir+'img_%d.jpg'%i, img)
        # img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        # im = Image.fromarray(img_norm.astype(np.uint8), 'L')
        # im.save(jpg_dir+'img_%d.jpg'%i)

        # hdu = fits.PrimaryHDU(img)
        # hdu.writeto(fits_dir+'img_%d.fits'%i, overwrite=True)
    
    

# %%
