#%%
import numpy as np
# import glob
import torch
import torchvision
from torch.utils.data import DataLoader
from utils.tool import Averager, collate_fn
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


def ground_truth(target):
    gt = np.c_[target['boxes'], target['labels']]
    return gt

def post_process_box(boxes, scores, labels, score_threshold=0.2, iou_threshold=0.2, min_size=5):
    idx_nms = nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold)
    idx_rmv_small = remove_small_boxes(boxes=boxes, min_size=min_size)
    idx_score = np.nonzero(scores>score_threshold)
    
    idx_post = np.intersect1d(idx_nms, idx_rmv_small)
    idx_post = np.intersect1d(idx_post, idx_score)
    
    return {'boxes':boxes[idx_post], 'scores':scores[idx_post], 'labels':labels[idx_post]}


def pred_result(output, score=0.2, iou=0.3):
    output = post_process_box(
        output['boxes'], output['scores'], output['labels'], 
        score_threshold=score, iou_threshold=iou
    )
    pred = np.c_[output['boxes'], output['labels'], output['scores']]
    return pred


if __name__ == "__main__":
    device = torch.device('cuda:0')
    model_ft = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model_dir = "/data/jdli/model/"
    model_nme = "Retina1026_mag24.bin"
    model     =  load_retina(model_dir+model_nme, model_ft, device=device)

    img_dir = "/data/jdli/C3test/img"
    tgt_dir = "/data/jdli/C3test/target/"
    c3val_data  = CSSTimg(img_dir, tgt_dir)

    dataloader = DataLoader(
        c3val_data, batch_size=8, shuffle=True, num_workers=8,
        collate_fn=collate_fn
    )

    images, targets, outputs = model_to_pred(
        dataloader, model, 
        device=torch.device('cuda:0')
        )

# %%


