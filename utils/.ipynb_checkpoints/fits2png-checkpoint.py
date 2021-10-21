import os
import glob
import cv2
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
import numpy  as np
import pandas as pd
from tqdm import tqdm


def read_fits(fits_name):
    raw = getdata(fits_name, memmap=False)
    return raw

def minmaxNormalization(raw, zscale):
    vmin, vmax = zscale.get_limits(raw)
    raw[raw<vmin] = vmin
    raw[raw>vmax] = vmax
    return (raw - vmin) / (vmax - vmin)

def extract_boxes(cat_name, mag_deep=21):
    
    cat = Table.read(cat_name, format='ascii').to_pandas()
    ind = (cat['mag']<mag_deep)
    cat = cat[ind]
    labels = np.array(cat['flag'].values, dtype=np.uint8)
    mags   = np.array(cat['mag'].values)
    xc, yc = cat['xImage'].values, cat['yImage'].values
    return {'labels':labels, 'mag':mags, 'x':xc, 'y':yc}

def split_raws(raw_name, cat_name, zscale, mag_deep=23, img_dir='./test_img/', tgt_dir='./test_tgt', xsplit=4, ysplit=4):
    
    raw = read_fits(raw_name)
    cat = extract_boxes(cat_name, mag_deep=23)
    
    height, width = raw.shape[0], raw.shape[1]
    x1, y1 = 0, 0
    M = height//xsplit
    N = width//ysplit
    xc, yc = cat['x'], cat['y']
    
    i = 0
    for y in range(0,height,M):
        for x in range(0,width,N):
            tile = raw[y:y+M, x:x+N]
            img  = minmaxNormalization(tile, zscale)
            ind_sub = (xc>x) & (xc<x+N) & (yc>y) & (yc<y+M)
            ximg, yimg = xc[ind_sub]-x, yc[ind_sub]-y
            
#             cv2.imwrite(save_dir+"chip%d.png" %(i), img)
            out_name = img_dir + "chip%d.fits"%(i)
            hdu = fits.PrimaryHDU(img)
            hdu.writeto(out_name, overwrite=True)
            
            boxsize = 20
            x1 = ximg - boxsize; x1[x1<0] = 0
            x2 = ximg + boxsize; x2[x2>N] = N
            y1 = yimg - boxsize; y1[y1<0] = 0
            y2 = yimg + boxsize; y2[y2>M] = M
            box = np.c_[x1, y1, x2, y2]
            target = {'box':box, 'labels':cat['labels'][ind_sub], 'mag':cat['mag'][ind_sub]}
            np.save(tgt_dir + "chip%d.npy"%(i), target)
            i+=1
    return
            
if __name__ == "__main__":
    
    zscl = ZScaleInterval(
        nsamples=int(1e3), contrast=0.3, max_reject=0.5, 
        min_npixels=5, krej=2.5, max_iterations=10
    )
#     save_dir = './test_img/'
#     raw_name = "/data/jdli/C3/MSC_0000500/MSC_MS_210529060000_100000500_06_raw.fits"
#     cat_name = "/data/jdli/C3/MSC_0000500/MSC_210525120000_0000500_06.cat"
#     split_raws(raw_name, cat_name, zscl)

#     """Make training set"""
#     root_dir = "/data/share/C3mulImg/multipleBandsImaging/CSST_shearOFF/MSC_00005*"

#     raw_lst_1 = glob.glob(root_dir+'/MSC_MS*_08_raw.fits')
#     raw_lst_2 = glob.glob(root_dir+'/MSC_MS*_23_raw.fits')
#     raw_lst   = (raw_lst_1+raw_lst_2); raw_lst.sort()

#     cat_lst_1 = glob.glob(root_dir+'/MSC_*08.cat')
#     cat_lst_2 = glob.glob(root_dir+'/MSC_*23.cat')
#     cat_lst   = (cat_lst_1+cat_lst_2); cat_lst.sort()
    
#     print(len(raw_lst), len(cat_lst))
    
#     for k in tqdm(range(len(raw_lst))):
#         raw_name = raw_lst[k]
#         cat_name = cat_lst[k]
#         split_raws(
#             raw_name, cat_name, zscl, 
#             img_dir='/data/jdli/C3train/img/img%d'%k, 
#             tgt_dir='/data/jdli/C3train/target/tgt%d'%k
#         )
    
    """Make test set"""
    root_dir = "/data/share/C3mulImg/multipleBandsImaging/CSST_shearOFF/MSC_000049*"

    raw_lst_1 = glob.glob(root_dir+'/MSC_MS*_08_raw.fits')
    raw_lst_2 = glob.glob(root_dir+'/MSC_MS*_23_raw.fits')
    raw_lst   = (raw_lst_1+raw_lst_2); raw_lst.sort()

    cat_lst_1 = glob.glob(root_dir+'/MSC_*08.cat')
    cat_lst_2 = glob.glob(root_dir+'/MSC_*23.cat')
    cat_lst   = (cat_lst_1+cat_lst_2); cat_lst.sort()
    
    for k in tqdm(range(len(raw_lst))):
        raw_name = raw_lst[k]
        cat_name = cat_lst[k]
        split_raws(raw_name, cat_name, zscl, img_dir='/data/jdli/C3test/img/img%d'%k, tgt_dir='/data/jdli/C3test/target/tgt%d'%k)
        
        
