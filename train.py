import time
import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adadelta, Adam, RMSprop, Rprop
from torch.utils.tensorboard import SummaryWriter
from CSST import CSSTimg
from utils.tool import Averager, collate_fn


def train(model, tr_data_loader, num_epochs=20, itr=1, num_iters=50, loss_hist=Averager(), device=torch.device('cuda:0')):
    writer = SummaryWriter("runs/retina1018/")

    for epoch in range(num_epochs):
        loss_hist.reset()
        for images, targets in tr_data_loader:
            start = time.time()
            
            images = list(image.to(device) for image in images)
            targets= [{k: v.to(device) for k,v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_hist.send(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % num_iters == 0:
                end = time.time()
                print(f"Iteration #{itr}  loss: {loss_value}  time:{(end-start)*num_iters}")
                writer.add_scalar('training loss', loss_value, epoch*itr)
            itr+=1
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch}  loss: {loss_hist.value}")


if __name__ == "__main__":

    device = torch.device('cuda:1')

    anchor_generator = AnchorGenerator(
    sizes=((2, 4, 8, 16, 32),),
    aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model_ft = retinanet_resnet50_fpn(
        pretrained=True, 
#         num_classes=3, 
        score_thresh=0.3, nms_thresh=0.3
        # anchor_generator=anchor_generator,
        )

    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # backbone.out_channels = 1280
    # model_ft = RetinaNet(backbone, num_classes=3, score_thresh=0.25, nms_thresh=0.25, anchor_generator=anchor_generator)

    model_ft.to(device)

    for param in model_ft.parameters():
        param.requires_grad = True

    params = [p for p in model_ft.parameters() if p.requires_grad]

#============================================================================
#============================================================================
    loss_hist = Averager()
    itr = 1
    num_epochs = 50
    num_iters  = 50

    # opt_list = [
    #     SGD(params, lr=1, momentum=0.9, weight_decay=0.0005), 
    #     Adadelta(params, lr=1, rho=0.9, eps=1e-06, weight_decay=0.0005), 
    #     Adam(params, lr=1), 
    #     RMSprop(params, lr=1), 
    #     Rprop(params, lr=1), 
    #     SGD(params, lr=1)
    # ]
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adadelta(params, lr=1, rho=0.9, eps=1e-06, weight_decay=0.0005)

#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs*num_iters)
    
    tr_dir =     "/data/jdli/C3train/img"
    tr_tgt_dir = "/data/jdli/C3train/target_mag23"
    tr_img = CSSTimg(tr_dir, tr_tgt_dir)
    
    tr_data_loader = DataLoader(
        tr_img, batch_size=12, shuffle=True, num_workers=16,
        collate_fn=collate_fn
        )

    train(model_ft, tr_data_loader, device=device, num_epochs=num_epochs)
    torch.cuda.empty_cache()
#     torch.save(model_ft.state_dict(), "model/Retina1019_mag23.bin")
    torch.save(model_ft.state_dict(), "/data/jdli/model/Retina1019_mag23.bin")