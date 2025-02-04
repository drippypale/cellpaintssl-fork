import pandas as pd
import torch
from torchvision import transforms

from . import image_ops as imo
from . import utils

def get_img_from_path(path):
    img, metadata = imo.load_tiff_img(path)
    otsuth = float(metadata['otsuth'])
    img = torch.from_numpy(imo.uint16_to_float(img))
    return img, otsuth

class MergedChannelsDataset():
    """
    Dataset classes for merged channels stored in 
    the csv column "FileName_Merged".
    The "otshuth" value is available in 
    the metadata dict as well
    (saved as a python string): 
    th = float(metadata["otsuth"])
    """
    def __init__(self, 
                 data: pd.DataFrame(),
                 transform=None,
                 inference=False,
                 crop_size=224,
                 stride: int=224, 
                 min_area_ratio: float=0.01,
                 meta_columns: list=['batch','plate', 'well', 'field'],
                 otsu_chan: int=4):
        self.data = data
        self.transform = transform
        # inference related params
        self.inference = inference
        self.crop_size = crop_size
        self.resize = transforms.Resize(size=crop_size)
        self.stride = stride
        self.min_area_ratio = min_area_ratio
        self.meta_columns = meta_columns
        self.otsu_chan = otsu_chan
        self.batches = data['batch'].values

    def __len__(self):
        return len(self.data)                   
    def __getitem__(self, ref_idx):
        path = self.data['FileName_Merged'].loc[ref_idx]
        
        if '.tif' in path:
            img, metadata = imo.load_tiff_img(path)
            otsuth = float(metadata['otsuth'])
        else:
            img = imo.load_np_img(path)
            otsuth = float(self.data['Otsu_th'].loc[ref_idx])

        img = torch.from_numpy(imo.uint16_to_float(img))

        if not self.inference:
            img = self.transform(img, otsuth)
            return img, otsuth
        else:
            crops_with_metadata = {}
            croplist, labels = imo.generate_cellcrops(img=img,
                                                  crop_size=self.crop_size,
                                                  min_area_ratio=self.min_area_ratio,
                                                  stride=self.stride,
                                                  otsu_chan=self.otsu_chan,
                                                  otsuth=otsuth)
            crops = torch.stack(croplist)
            if self.transform:
                crops = self.transform(crops)
            crops_with_metadata['crops'] = crops
            crops_with_metadata['labels'] = torch.BoolTensor(labels)
            for c in self.meta_columns:
                crops_with_metadata[c] = self.data.loc[ref_idx, c]
            return crops_with_metadata

class WholeImgDataset:
    """
    Dataset class for reading in a whole microscopy image (1080 x 1080).
    Returns two image views from each field of view.
    """
    def __init__(self, 
                 data: pd.DataFrame(),
                 transform=None,
                 scale_intensity_percentile: float=99.9, 
                 meta_columns: list=['batch','plate', 'well', 'field']):
        self.data = data
        self.scale_intensity_percentile = scale_intensity_percentile
        self.file_cols = utils.get_filename_columns(self.data)
        self.meta_columns = meta_columns
        self.transform = transform
    
    def __len__(self):
        return len(self.data)       
                
    def __getitem__(self, ref_idx):
        metadata = {}
        img_files  = self.data.loc[ref_idx, self.file_cols].values
        img = imo.read_image_channels(img_files)
        img = imo.scale_intensities(img, self.scale_intensity_percentile)
        views = torch.Tensor(img)
        if self.transform:
            views = self.transform(views)
        for c in self.meta_columns:
            metadata[c] = self.data.loc[ref_idx, c]
        return views, metadata