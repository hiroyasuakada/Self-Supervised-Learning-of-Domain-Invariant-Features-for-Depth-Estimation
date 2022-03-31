import os, glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
from dataloader.image_folder import make_dataset
from copy import deepcopy
# from util.FDA import FDA_source_to_target_np 

from util.process_data import read_text_lines, read_file_data, generate_depth_map
import cv2
import scipy
import scipy.io

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_sampler(opt, dataset):
    if opt.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    return sampler


def should_shuffle(opt):
    return (not opt.distributed) and opt.shuffle


def crop_center(pil_img, crop_width, crop_height): 
    img_width, img_height = pil_img.size 
    return pil_img.crop(((img_width - crop_width) // 2, 
                         (img_height - crop_height) // 2, 
                         (img_width + crop_width) // 2, 
                         (img_height + crop_height) // 2))


def dataloader_val(opt, mode='train'):
    dataset = CreateDataset(opt, mode=mode)

    if mode == 'train':
        sampler = get_sampler(opt, dataset)
        dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=opt.batch_size, 
            shuffle=should_shuffle(opt), 
            num_workers=int(opt.num_threads), 
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )  

        return dataset_loader, sampler

    elif mode == 'val' or 'test':
        dataset_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=False,
            drop_last=False,
            sampler=None
        )
   
        return dataset_loader


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode='train'):
        super(CreateDataset, self).__init__()
        self.opt = opt
        self.mode = mode

        if mode == 'train':
            self.img_source_paths, self.img_source_size = make_dataset(opt, opt.img_source_dir, opt.data_list_source)
            self.img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir, opt.data_list_target)

            self.lab_source_paths, self.lab_source_size = make_dataset(opt, opt.lab_source_dir, opt.data_list_source)
            self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir, opt.data_list_target)

        elif mode == 'val':
            self.img_target_val_paths, self.img_target_val_size = make_dataset(opt, opt.img_target_val_dir)

            if opt.dataset == 'nyu':
                self.lab_target_val_paths, self.lab_target_val_size = make_dataset(opt, opt.lab_target_val_dir)
            elif opt.dataset == 'kitti':
                self.lab_target_val_list, self.lab_size_list = get_Kitti_depth_map(opt, opt.lab_target_val_dir)

        elif mode == 'test':
            self.img_target_paths, self.img_target_size = make_dataset(opt, opt.img_target_dir)
            self.lab_target_paths, self.lab_target_size = make_dataset(opt, opt.lab_target_dir)

        self.transform_augment = self._make_transform(opt, True) # for img_source
        self.transform_no_augment = self._make_transform(opt, False)  # for img_target
        self.transform_no_augment_depth = self._make_transform(opt, False, True)  # for depth in both source and target

    # get tensor data
    def __getitem__(self, item):

        # load dataset for training
        if self.mode == 'train':
            # get index
            if self.img_source_size >= self.img_target_size:
                index_source = item
                index_target = random.randint(0, self.img_target_size - 1)
            elif self.img_source_size < self.img_target_size:
                index_source = random.randint(0, self.img_source_size - 1)
                index_target = item           

            # load images and depth
            img_source_path = self.img_source_paths[index_source]
            lab_source_path = self.lab_source_paths[index_source]
            img_source = Image.open(img_source_path).convert('RGB')
            lab_source = Image.open(lab_source_path)

            img_target_path = self.img_target_paths[index_target]
            lab_target_path = self.lab_target_paths[index_target]
            img_target = Image.open(img_target_path).convert('RGB')
            lab_target = Image.open(lab_target_path)          
            
            # for NYU, we crop white edges
            if self.opt.dataset == 'nyu':
                img_source = crop_center(img_source, 624, 468) 
                lab_source = crop_center(lab_source, 624, 468)
                img_target = crop_center(img_target, 624, 468)
                lab_target = crop_center(lab_target, 624, 468)

            # resize images and depth
            img_source = img_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_source = lab_source.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            img_target = img_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

            # apply paired transformation
            img_source, lab_source = self.paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            lab_source = self.transform_no_augment_depth(lab_source)

            img_target, lab_target = self.paired_transform(self.opt, img_target, lab_target)
            img_target = self.transform_no_augment(img_target)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_source': img_source, 'img_source_paths': img_source_path,
                    'lab_source': lab_source, 'lab_source_paths': lab_source_path,
                    'img_target': img_target, 'img_target_paths': img_target_path,
                    'lab_target': lab_target, 'lab_target_paths': lab_target_path,
                    }
        
        # load dataset for validation
        elif self.mode == 'val':
            if self.opt.dataset == 'nyu':
                # load images and depth
                img_target_val_path = self.img_target_val_paths[item]
                lab_target_val_path = self.lab_target_val_paths[item]
                img_target_val = Image.open(img_target_val_path).convert('RGB')
                lab_target_val = Image.open(lab_target_val_path)

                # # for NYU, we crop white edges
                # # if self.opt.dataset == 'nyu':
                # img_target_val = crop_center(img_target_val, 624, 468) 
                # lab_target_val = crop_center(lab_target_val, 624, 468)

                # resize images and depth
                img_target_val = img_target_val.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
                lab_target_val = lab_target_val.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

                # apply general transformation
                img_target_val = self.transform_no_augment(img_target_val)
                lab_target_val = self.transform_no_augment_depth(lab_target_val)

                return {'img_target_val': img_target_val, 'img_target_val_paths': img_target_val_path,
                        'lab_target_val': lab_target_val, 'lab_target_val_paths': lab_target_val_path,
                        'lab_size': None
                        }
            
            elif self.opt.dataset == 'kitti':
                # load images and depth
                img_target_val_path = self.img_target_val_paths[item]
                img_target_val = Image.open(img_target_val_path).convert('RGB')

                # resize images and depth
                img_target_val = img_target_val.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

                # apply general transformation
                img_target_val = self.transform_no_augment(img_target_val)
                
                # load ground truth depth maps of Kitti
                lab_target_val = self.lab_target_val_list[item]
                lab_size = self.lab_size_list[item]

                return {'img_target_val': img_target_val, 'img_target_val_paths': img_target_val_path,
                        'lab_target_val': lab_target_val, 
                        'lab_size': lab_size,
                        }


        # load dataset for testing
        elif self.mode == 'test':
            # get paths
            img_target_path = self.img_target_paths[item]
            lab_target_path = self.lab_target_paths[item]

            # load images and depth
            img_target = Image.open(img_target_path).convert('RGB')
            lab_target = Image.open(lab_target_path)

            # # for NYU, we crop white edges
            # if self.opt.dataset == 'nyu':
            #     img_target = crop_center(img_target, 624, 468) 
            #     lab_target = crop_center(lab_target, 624, 468)

            # resize images and depth
            img_target = img_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)
            lab_target = lab_target.resize([self.opt.load_size[0], self.opt.load_size[1]], Image.BICUBIC)

            # apply general transformation
            img_target = self.transform_no_augment(img_target)
            lab_target = self.transform_no_augment_depth(lab_target)

            return {'img_target': img_target, 'img_target_paths': img_target_path,
                    'lab_target': lab_target, 'lab_target_paths': lab_target_path,
                    }

    def __len__(self):
        if self.mode == 'train':
            return max(self.img_source_size, self.img_target_size)

        elif self.mode == 'val':
            return self.img_target_val_size
        
        elif self.mode == 'test':
            return self.img_target_size

    def _make_transform(self, opt, augment, depth=False):
        transforms_list = []

        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8))
        
        transforms_list.append(transforms.ToTensor())
        
        if depth:
            pass
        else:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transforms_list)

    def paired_transform(self, opt, image, depth):

        if opt.flip:
            n_flip = random.random()
            if n_flip > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if opt.rotation:
            n_rotation = random.random()
            if n_rotation > 0.5:
                degree = random.randrange(-500, 500)/100
                image = F.rotate(image, degree, Image.BICUBIC)
                depth = F.rotate(depth, degree, Image.BILINEAR)

        return image, depth


def get_Kitti_depth_map(opt, filename='./datasplit/eigen_test_files.txt'): # './datasplit/eigen_val_files.txt'
    print('checking availability of kitti gt depth maps...')
    test_files = read_text_lines(filename)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, opt.txt_data_path)

    num_samples = len(im_files)
    ground_truths = []

    print('loading kitti gt depth maps...')
    for t_id in range(num_samples):
        camera_id = cams[t_id]

        depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
        ground_truths.append(depth.astype(np.float32))         

    return ground_truths, im_sizes