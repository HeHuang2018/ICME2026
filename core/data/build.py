from .datasets.common_corruption import CorruptionDataset
from .datasets.waterbirds import WaterbirdsDataset
from .data_loading import ImageList,get_augmentation,complete_data_dir_path
import torch
from .ttasampler import build_sampler
from torch.utils.data import DataLoader
from ..utils.result_precess import AvgResultProcessor
import getpass
username = getpass.getuser()
import torchvision.transforms as transforms
from loguru import logger as log
import os
import torchvision
import random
from prefetch_generator import BackgroundGenerator
import torch, os
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import json


class CamelyonDataset(Dataset):
    def __init__(self, root_dir, transform, split):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CamelyonDataset, self).__init__()
        self.data_dir = '/gemini/data-1/datasets/camelyon17_v1.0'
        self.original_resolution = (96,96)

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})

        # Get the y values
        self.y_array = torch.LongTensor(self.metadata_df['tumor'].values)
        self.y_size = 1
        self.n_classes = 2

        # Get filenames
        self.input_array = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            for patient, node, x, y in
            self.metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]

        # Extract splits
        # Note that the hospital numbering here is different from what's in the paper,
        # where to avoid confusing readers we used a 1-indexed scheme and just labeled the test hospital as 5.
        # Here, the numbers are 0-indexed.
        test_center = 2
        val_center = 1

        self.split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self.split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        centers = self.metadata_df['center'].values.astype('long')
        num_centers = int(np.max(centers)) + 1
        val_center_mask = (self.metadata_df['center'] == val_center)
        test_center_mask = (self.metadata_df['center'] == test_center)
        self.metadata_df.loc[val_center_mask, 'split'] = self.split_dict['val']
        self.metadata_df.loc[test_center_mask, 'split'] = self.split_dict['test']
        '''
        self._split_scheme = split_scheme
        if self._split_scheme == 'official':
            pass
        elif self._split_scheme == 'in-dist':
            # For the in-distribution oracle,
            # we move slide 23 (corresponding to patient 042, node 3 in the original dataset)
            # from the test set to the training set
            slide_mask = (self._metadata_df['slide'] == 23)
            self._metadata_df.loc[slide_mask, 'split'] = self.split_dict['train']
        else:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        '''
        self.split_array = self.metadata_df['split'].values
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        self.y_array = self.y_array[split_idx]
        print(split_idx)
        tmp = []
        for idx in split_idx:
            tmp.append(self.input_array[idx]) 
        
        self.input_array = tmp #self.input_array[split_idx]
        '''
        self.metadata_array = torch.stack(
            (torch.LongTensor(centers),
             torch.LongTensor(self.metadata_df['slide'].values),
             self.y_array),
            dim=1)
        self.metadata_fields = ['hospital', 'slide', 'y']
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['slide'])
        '''
        self.transform = transform

        print('Total # images:{}, labels:{}'.format(len(self.input_array),len(self.y_array)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        img_filename = os.path.join(
           self.data_dir,
           self.input_array[index])
        x = Image.open(img_filename).convert('RGB')
        y = self.y_array[index]
        #y = F.one_hot(y, num_classes=2)
        if self.transform is not None:
            x = self.transform(x)
        return  x, y

    def __len__(self):
        return len(self.y_array)

class ImageNetDLoader(torch.utils.data.Dataset):
    def __init__ (self,
                  test_base_dir, few_test=None, transform=None, center_crop=False
                  ):
        super().__init__()

        self.test_path = test_base_dir
        self.categories_list = os.listdir(self.test_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test

        self.transforms=transform

        with open('/gemini/code/xhy/CTTA/robustbench/data/imgnet_d2imgnet_id.txt') as f:
            self.dict_imgnet_d2imagenet_id = json.load(f)

        for each in self.categories_list:
            folder_path = os.path.join(self.test_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                self.label_lists.append(self.dict_imgnet_d2imagenet_id[each]+[-1]*(10-len(self.dict_imgnet_d2imagenet_id[each]))) 

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):
        return self.transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
        img = Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]
        return {"images": img_tensor, "labels": labels}

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_domainnet126(cfg, domain_name):
    # domain = cfg.MODEL.CKPT_PATH.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    data_list = [f"/gemini/code/xhy/CTTA/core/data/datasets/domainnet126_lists/{domain_name}_list.txt"]
    transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
    data_dir = f"/gemini/data-1/datasets/DomainNet-126"
    test_dataset = ImageList(
        image_root=data_dir,
        label_files=data_list,
        transform=transform
    )
    test_loader = DataLoaderX(
        test_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False, 
        num_workers=2,
        pin_memory=False,
    )
    return test_loader

def build_imagenet_d(cfg):
    transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
    log.info(f"==>> transform:  {transform}")
    data_dir = complete_data_dir_path(root=cfg.DATA_DIR, dataset_name=cfg.CORRUPTION.DATASET)
    log.info(f"==>> data_dir:  {data_dir}")
    test_dataset = ImageNetDLoader(test_base_dir=data_dir, transform=transform)
    # test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    # random.shuffle(test_dataset.samples)
    test_loader = DataLoaderX(
        test_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True, 
        num_workers=2,
        pin_memory=False,
    )
    return test_loader

def build_imagenet_k_r_v2(cfg):
    transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
    log.info(f"==>> transform:  {transform}")
    data_dir = complete_data_dir_path(root=cfg.DATA_DIR, dataset_name=cfg.CORRUPTION.DATASET)
    log.info(f"==>> data_dir:  {data_dir}")
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    # random.shuffle(test_dataset.samples)
    test_loader = DataLoaderX(
        test_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True, 
        num_workers=2,
        pin_memory=False,
    )
    return test_loader

def build_loader(cfg):
    if cfg.CORRUPTION.DATASET == "cifar10":
        dataset_name = "CIFAR-10-C"
    elif cfg.CORRUPTION.DATASET == "cifar100":
        dataset_name = "CIFAR-100-C"
    elif cfg.CORRUPTION.DATASET == "imagenet":
        dataset_name = "ImageNet-C"
    elif cfg.CORRUPTION.DATASET == "imagenet_3dcc":
        dataset_name = "ImageNet-3DCC"
    elif cfg.CORRUPTION.DATASET == "imagenet_c_bar":
        dataset_name = "ImageNet-C-Bar"
    if cfg.CORRUPTION.DATASET in ["cifar10", "cifar100","imagenet","imagenet_3dcc","imagenet_c_bar"]:
        dataset_class = CorruptionDataset
    else:
        raise NotImplementedError(f"Not Implement for dataset: {cfg.CORRUPTION.DATASET}")

    if cfg.LOADER.SAMPLER.TYPE == "temporal":

        # try:
        #     log.info(f"Load temporal dataset from /gemini/data-1/datasets/{dataset_name}/temporal_{cfg.LOADER.SAMPLER.GAMMA}.pth")
        #     ds = torch.load(f"/gemini/data-1/datasets/{dataset_name}/temporal_{cfg.LOADER.SAMPLER.GAMMA}.pth")
        # except Exception as e:
        file = f"/gemini/data-1/datasets/{dataset_name}/temporal_{cfg.LOADER.SAMPLER.GAMMA}.pth"
        if os.path.exists(file):
            log.info(f"Load temporal dataset from {file}")
            ds = torch.load(file)
        else:
            ds = dataset_class(cfg)
            torch.save(ds, f"/gemini/data-1/datasets/{dataset_name}/temporal_{cfg.LOADER.SAMPLER.GAMMA}.pth")
        sampler = build_sampler(cfg, ds.data_source)
        loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=False)

    elif cfg.LOADER.SAMPLER.TYPE == "mix":

        file = f"/gemini/data-1/datasets/{dataset_name}/mix.pth"
        if os.path.exists(file):
            log.info(f"Load mixed dataset from /gemini/data-1/datasets/{dataset_name}/mix.pth")
            ds = torch.load(f"/gemini/data-1/datasets/{dataset_name}/mix.pth")
        else:
            ds = dataset_class(cfg)
            torch.save(ds, f"/gemini/data-1/datasets/{dataset_name}/mix.pth")
        # sampler = build_sampler(cfg, ds.data_source)
        # loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
        loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, num_workers=0, pin_memory=False, shuffle=True)
    else:
        loader = DataLoader(ds, cfg.TEST.BATCH_SIZE, num_workers=0, pin_memory=False)

    result_processor = AvgResultProcessor(ds.domain_id_to_name)

    return loader, result_processor

def ckpt_path_to_domain_seq(ckpt_path: str):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]

def build_waterbirds(cfg):
    def custom_collate_fn(batch):
        imgs, ys, attrs = zip(*batch)
        
        imgs = torch.stack(imgs, dim=0)
        ys = torch.tensor(ys)
        keys = list(attrs[0].keys())
        attrs = {key: [attr[key] for attr in attrs] for key in keys}
        return imgs, ys, attrs
            
    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_file = f"/gemini/data-1/datasets/waterbirds/waterbirds_dataset.h5py"
    test_dataset = WaterbirdsDataset(dataset_file, 'test', transform)
    loader = DataLoaderX(
        test_dataset, 
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False, 
        num_workers=0,
        pin_memory=False, 
        collate_fn=custom_collate_fn
    )
    return loader