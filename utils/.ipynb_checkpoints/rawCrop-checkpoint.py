import os
import glob
import cv2
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
import numpy  as np
import pandas as pd
from tqdm import tqdm


def read_fits(fits_name):
    raw = getdata(fits_name, memmap=False)
    return raw

def read_crs(crs_name):
    return getdata(crs_name)

def get_pixscl(fits_name):
    head = getheader(raw_nme)
    return head['PIXSCAL1'], head['PIXSCAL2']

def scale_box(x, lim):
    x[x<0]   = 0
    x[x>lim] = lim
    return x
    

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
    labels[labels==2] = 1
    mags   = np.array(cat['mag'].values)
    xc, yc = cat['xImage'].values, cat['yImage'].values
    return {'labels':labels, 'mag':mags, 'x':xc, 'y':yc}

def split_raws(raw_name, cat_name, zscale,  img_dir='./test_img/', tgt_dir='./test_tgt', mag_deep=23, xsplit=4, ysplit=4, box_thrsd=10, times_hlr=3, xpixscl=0.075, ypixscl=0.075):
    """
    from C3 raw data to crop image with specify target box
    target boxsize = times_hlr * half-light disk radius
    """
    
    raw = read_fits(raw_name)
    cat = extract_boxes(cat_name, mag_deep=mag_deep)
    
    height, width = raw.shape[0], raw.shape[1]
    x1, y1 = 0, 0
    M = height//xsplit
    N = width//ysplit
    xc, yc = np.array(cat['x']), np.array(cat['y'])
    
    i = 0
    for y in range(0,height,M):
        for x in range(0,width,N):
            tile = raw[y:y+M, x:x+N]
            img  = minmaxNormalization(tile, zscale)
            ind_sub = (xc>x) & (xc<x+N) & (yc>y) & (yc<y+M)
            ximg, yimg = xc[ind_sub]-x, yc[ind_sub]-y
            
            out_name = img_dir + "chip%d.fits"%(i)
            hdu = fits.PrimaryHDU(img)
            hdu.writeto(out_name, overwrite=True)
            
            # Create target box
            xlen = cat[ind_sub]['hlr_disk']*times_hlr/xpixscl
            ylen = cat[ind_sub]['hlr_disk']*times_hlr/ypixscl
            xlen[xlen<box_thrsd]  = box_thrsd
            ylen[ylen<box_thrsd]  = box_thrsd
            
            x1 = scale_box(ximg-xlen,N) ; x2 = scale_box(ximg+xlen,N)
            y1 = scale_box(yimg-ylen,M);  y2 = scale_box(yimg+ylen,M)
            boxes = np.c_[x1, y1, x2, y2]
            
            target = {'box':boxes, 'labels':cat['labels'][ind_sub], 'mag':cat['mag'][ind_sub]}
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

    
    """Make training set"""
    root_dir = "/data/share/C3mulImg/multipleBandsImaging/CSST_shearOFF/MSC_00005*"

    raw_lst_1 = glob.glob(root_dir+'/MSC_MS*_08_raw.fits')
    raw_lst_2 = glob.glob(root_dir+'/MSC_MS*_23_raw.fits')
    raw_lst   = (raw_lst_1+raw_lst_2); raw_lst.sort()

    cat_lst_1 = glob.glob(root_dir+'/MSC_*08.cat')
    cat_lst_2 = glob.glob(root_dir+'/MSC_*23.cat')
    cat_lst   = (cat_lst_1+cat_lst_2); cat_lst.sort()
    
    print(len(raw_lst), len(cat_lst))
    
    for k in tqdm(range(len(raw_lst))):
        raw_name = raw_lst[k]
        cat_name = cat_lst[k]
        
        xpixscl, ypixscl = get_pixscl(raw_name)
        split_raws(
            raw_name, cat_name, zscl, 
            img_dir='/data/jdli/C3train/img/img%d'%k, 
            tgt_dir='/data/jdli/C3train/target/tgt%d'%k,
            mag_deep=24, 
            xsplit=4, ysplit=8, box_thrsd=10, 
            times_hlr=3, xpixscl=0.075, ypixscl=0.075
        )
    
    
    
#     """Make test set"""
#     root_dir = "/data/share/C3mulImg/multipleBandsImaging/CSST_shearOFF/MSC_000049*"

#     raw_lst_1 = glob.glob(root_dir+'/MSC_MS*_08_raw.fits')
#     raw_lst_2 = glob.glob(root_dir+'/MSC_MS*_23_raw.fits')
#     raw_lst   = (raw_lst_1+raw_lst_2); raw_lst.sort()

#     cat_lst_1 = glob.glob(root_dir+'/MSC_*08.cat')
#     cat_lst_2 = glob.glob(root_dir+'/MSC_*23.cat')
#     cat_lst   = (cat_lst_1+cat_lst_2); cat_lst.sort()
    
#     for k in tqdm(range(len(raw_lst))):
#         raw_name = raw_lst[k]
#         cat_name = cat_lst[k]
#         split_raws(raw_name, cat_name, zscl, img_dir='/data/jdli/C3test/img/img%d'%k, tgt_dir='/data/jdli/C3test/target/tgt%d'%k)
        
        
