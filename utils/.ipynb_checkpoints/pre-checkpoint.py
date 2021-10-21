# import os
import glob
import numpy as np
from astropy.table import Table
# from astropy.io import fits
# from astropy.wcs import WCS
from astropy.io.fits import getdata
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
rcParams["font.size"] = 14
from astropy.visualization import ZScaleInterval
import torch
from tqdm import tqdm

"""
Load all multiple Bands Imaging
"""


class CSSTimg():
    """ 
    CSST image instance
    """

    def __init__(self, root_dir, height=9232, width=9216, mag_lmt=21, zscale=None):
        self.root_dir = root_dir
        self.img_names = sorted(glob.glob(root_dir+'/MSC_MS*.fits'))
        self.cat_names = sorted(glob.glob(root_dir+'/*.cat'))
        self.h = height
        self.w = width
        self.mid_h = int(height/2)
        self.mid_w  = int(width/2)
        self.zscale = zscale
        self.mag_lmt = mag_lmt
        
    def __len__(self) -> int:
        num_sets = len(self.img_names)
        return num_sets

    def load_image_disk(self, idx):
        # load from disk -- each set directory contains seperate files for images and masks
        # read images
        self.raw = getdata(self.img_names[idx], memmap=False)
        return self.raw
    
    def crop(self, raw):
        self.img_a = raw[0:self.mid_h, 0:self.mid_w]
        self.img_b = raw[0:self.mid_h, self.mid_w:self.w]
        self.img_c = raw[self.mid_h:self.h, 0:self.mid_w]
        self.img_d = raw[self.mid_h:self.h, self.mid_w:self.w]
        
        return [self.img_a, self.img_b, self.img_c, self.img_d]
    
    def minmaxNormalization(self, raw):
        if self.zscale:
            vmin, vmax = self.zscale.get_limits(self.raw)
        else:
            vmin, vmax = raw.min(), raw.max()
        raw[raw<vmin] = vmin
        raw[raw>vmax] = vmax
        return (raw - vmin) / (vmax - vmin)
    
    def extract_boxes(self, idx):
        cat = Table.read(self.cat_names[idx], format='ascii').to_pandas()
        ind = (cat['mag']<self.mag_lmt)
        cat = cat[ind]
        labels = np.array(cat['flag'].values, dtype=np.uint8)
        
        xc, yc = cat['xImage'].values, cat['yImage'].values

        hlr = abs(10*cat['hlr_disk'].values)
        axis = np.where(hlr<30, 30, hlr)
        x1, x2 = xc-axis, xc+axis
        y1, y2 = yc-axis, yc+axis
        
        ind_a = (xc>=0) & (xc<self.mid_h)     & (yc>=0)         & (yc<self.mid_w)
        ind_b = (xc>self.mid_h) & (xc<self.h) & (yc>=0)         & (yc<self.mid_w)
        ind_c = (xc>=0) & (xc<self.mid_h)     & (yc>self.mid_w) & (yc<=self.w)
        ind_d = (xc>self.mid_h) & (xc<self.h) & (yc>self.mid_w) & (yc<self.w)
        
        box_a = np.c_[x1[ind_a], y1[ind_a], x2[ind_a], y2[ind_a]]
        box_b = np.c_[x1[ind_b], y1[ind_b]-self.mid_w, x2[ind_b], y2[ind_b]-self.mid_w]
        box_c = np.c_[x1[ind_c]-self.mid_h, y1[ind_c], x2[ind_c]-self.mid_h, y2[ind_c]]
        box_d = np.c_[x1[ind_d]-self.mid_h, y1[ind_d]-self.mid_w, x2[ind_d]-self.mid_h, y2[ind_d]-self.mid_w]
        
        self.boxes = [box_a, box_c, box_b , box_d]
        self.labels =[labels[ind_a], labels[ind_b], labels[ind_c], labels[ind_d]]
        return self.boxes, self.labels

    
    def __getitem__(self, idx: int):
        raw = self.load_image_disk(idx)
        raw_crops = self.crop(raw)
        imgs = [self.minmaxNormalization(r) for r in raw_crops]
        boxes, labels = self.extract_boxes(idx)
#         boxes, labels = self.distri_boxes(boxes, labels)
        target = {}
        target["labels"] = labels
        target["boxes"] = boxes
        target["image_id"] = torch.tensor([idx])
        target["image_name"] = self.img_names[idx]
        target["cat_name"] = self.cat_names[idx]
        return imgs, raw, target


if __name__ == "__main__":

    zscl = ZScaleInterval(nsamples=int(1e3), contrast=0.3, 
                        max_reject=0.5, min_npixels=5, krej=2.5, max_iterations=10)
    root_dir = "./test/"
    c3img = CSSTimg(root_dir, zscale=zscl)

    # print(len(c3img))
    img, raw, target = c3img[0]

    for k in tqdm(range(len(c3img))):
        imgs, raw, target = c3img[k]

        for i in range(4):
            img = imgs[i]
            fig, ax = plt.subplots(1,1, figsize=(10,10))
            
            ax.imshow(img, vmin=0, vmax=1, cmap='gray')
            ax.set_axis_off()

            for box in target['boxes'][i]:
                ax.add_patch(
                    patches.Rectangle(
                        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                        edgecolor='red', facecolor='none', lw=1
                        )
                    )
            plt.tight_layout()
            fig.savefig("./test_png/targe_%s_block_%s.png" %(int(target['image_id']), i))