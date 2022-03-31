import numpy as np
import torch
import os
import imageio

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, depth=False):
    if image_tensor.dim() == 3:  # (C, H, W)
        image_tensor = image_tensor.cpu().float()
    else:
        image_tensor = image_tensor[0].cpu().float()
    
    if depth:
        image_tensor = image_tensor * bytes
    else:
        image_tensor = (image_tensor + 1.0) / 2.0 * bytes

    image_numpy = (image_tensor.permute(1, 2, 0)).numpy().astype(imtype)
    return image_numpy
 
def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

########################################################################################################

def get_feat_patches(x, kc, kh, kw, dc, dh, dw):
    x = x.unsqueeze(0)  # (256, 24, 32) to (1, 256, 24, 32)
    patches = x.unfold(1, int(kc), int(dc)).unfold(2, int(kh), int(dh)).unfold(3, int(kw), int(dw))
    patches = patches.contiguous().view(-1, int(kc), int(kh), int(kw))

    return patches

    # tensor_vec = get_feat_patches( # (256, 24, 32) to (1, 256, 24, 32) to (4, 256, 12, 16)
    #     tensor, 
    #     kc=self.z_numchnls, kh=self.z_height/self.split_scale, kw=self.z_width/self.split_scale, 
    #     dc=1, dh=self.z_height/self.split_scale, dw=self.z_width/self.split_scale
    #     )
    # tensor_vec = tensor_vec.view(-1,int(self.z_numchnls*self.z_height*self.z_width/self.split_scale/self.split_scale))  # z tensor to a vector (4, 256*12*16)

    # size = patches.size()[0]
    # imgid_new = []
    # for i in range(len(imgid)): # batch size
    #     for t in range(int(size/len(imgid))):
    #         imgid_new.append(imgid[i] + '_{}'.format(t))

    # assert size == len(imgid_new)


########################################################################################################