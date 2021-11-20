import glob
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.visualization import ZScaleInterval
import torch
# from tqdm import tqdm


class CSSTimg():
    """ 
    CSST image instance
    """

    def __init__(self, img_dir, tgt_dir, mag_lmt=21):
        
        self.img_names = glob.glob(img_dir+'/*.fits')
        self.img_names.sort()
        self.cat_names = glob.glob(tgt_dir+'/*.npy')
        self.cat_names.sort()

        
    def __len__(self) -> int:
        num_sets = len(self.img_names)
        return num_sets

    def load_image_disk(self, idx):
        # load from disk -- each set directory contains seperate files for images and masks
        raw = getdata(self.img_names[idx], memmap=False)
        height, width = raw.shape[0], raw.shape[1]
        
        raw = raw.reshape(1, height,  width)
        return raw.astype(np.float32)
    
    def load_target_disk(self, idx):
        target = np.load(self.cat_names[idx], allow_pickle=True).item()
        return target
    
    def __getitem__(self, idx: int):
        raw = self.load_image_disk(idx)
        raw = torch.as_tensor(raw, dtype=torch.float32)
        
        target_raw = self.load_target_disk(idx)
        target = {}
        target["boxes"]  = torch.as_tensor(np.array(target_raw["box"]),    dtype=torch.float32)
        target["labels"] = torch.as_tensor(np.array(target_raw["labels"]), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["mag"] = torch.as_tensor(np.array(target_raw["mag"]),    dtype=torch.float32)
#         target["image_name"] =  self.img_names[idx]
#         target["cat_name"] = self.cat_names[idx]
        return raw, target
    
if __name__ == "__main__":
    print("Hello CSST")
    
    c3data = CSSTimg("/data/jdli/C3train/img", "/data/jdli/C3train/target")
    print(len(c3data))