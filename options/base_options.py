import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # basic define
        self.parser.add_argument('--project_name', type=str, default='project_name',
                                 help='name of the project. This is used for wnadb')
        self.parser.add_argument('--experiment_name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='models are save here')
        self.parser.add_argument('--txt_data_path', type=str, default='kitti_data', 
                                 help='If use .txt to load images, please indicate path to image dataset')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='wsupervised',
                                 help='choose which model to use, [supervised] | [wsupervised]')


        ### dataset parameters ###
        self.parser.add_argument('--img_source_dir', type=str, help='training and testing dataset for source domain')
        self.parser.add_argument('--img_target_dir', type=str, help='training and testing dataser for target domain')
        self.parser.add_argument('--lab_source_dir', type=str, help='training label for source domain')
        self.parser.add_argument('--lab_target_dir', type=str, help='training label for target domain')
        self.parser.add_argument('--img_target_val_dir', type=str, help='validation dataset for target domain')
        self.parser.add_argument('--lab_target_val_dir', type=str, help='validation label for target domain')
        self.parser.add_argument('--dataset_mode', type=str, default='unpaired', help='chooses how datasets are loaded. [paired| unpaired]')
        self.parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', nargs='+', type=int, default=[256, 192], help='scale images to this size, [256, 192] or [640, 192]')
        self.parser.add_argument('--flip', action='store_true', help='if specified, do flip the image for data augmentation')
        self.parser.add_argument('--rotation', action='store_true', help='if specified, rotate the images for data augmentation')
        self.parser.add_argument('--crop', action='store_true', help='if specified, crop the images for data augmentation')
        self.parser.add_argument('--shuffle', action='store_true', help='if true, takes images randomly')

        
        # network structure define
        self.parser.add_argument('--image_nc', type=int, default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--label_nc', type=int, default=1,
                                 help='# of output label channels')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='# of discriminator filter in first conv layer')
        self.parser.add_argument('--image_feature', type=int, default=512,
                                 help='the max channels for image features')
        self.parser.add_argument('--num_D', type=int, default=1,
                                 help='# of number of the discriminator')
        self.parser.add_argument('--transform_layers', type=int, default=9,
                                 help='# of number of the down sample layers for transform network')
        self.parser.add_argument('--task_layers', type=int, default=4,
                                 help='# of number of the down sample layers for task network')
        self.parser.add_argument('--image_D_layers', type=int, default=3,
                                 help='# of number of the down layers for image discriminator')
        self.parser.add_argument('--feature_D_layers', type=int, default=2,
                                 help='# of number of the layers for features discriminator')
        self.parser.add_argument('--task_model_type', type=str, default='UNet',
                                 help='select model for task network [UNet] |[ResNet]')
        self.parser.add_argument('--trans_model_type', type=str, default='ResNet',
                                 help='select model for transform network [UNet] |[ResNet]')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='batch normalization or instance normalization')
        self.parser.add_argument('--activation', type=str, default='PReLU',
                                 help='ReLu, LeakyReLU, PReLU, or SELU')
        self.parser.add_argument('--init_type', type=str, default='kaiming',
                                 help='network initialization [normal|xavier|kaiming]')
        self.parser.add_argument('--drop_rate', type=float, default=0,
                                 help='# of drop rate')
        self.parser.add_argument('--U_weight', type=float, default=0.1,
                                 help='weight for Unet')
        self.parser.add_argument('--simsiam_mode', type=str, default='normal',
                                 help='simsiam mode [normal|channel|patch]')                                 
        self.parser.add_argument('--unet_feature_id', type=int, default=-1,
                                 help='specify id to return features from Unet:\
                                      -1 for deconv2, -2 for deconv3, -3 for deconv4, -4 for deconv5, -5 for center_out')         
        self.parser.add_argument('--scale_simsiam_hidden', type=int, default=3,
                                 help='specify value to divide input dimension to get hidden dimension: 1,2,3,4, etc. please see cl_dec_model for detail')    

        # display parameter define
        self.parser.add_argument('--display_winsize', type=int, default=256,
                                 help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1,
                                 help='display id of the web')
        self.parser.add_argument('--display_port', type=int, default=8097,
                                 help='visidom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visidom web panel')

        ### others
        self.parser.add_argument('--pre_trained_on_ImageNet', action='store_true', help='ENB5 or not')
        self.parser.add_argument('--init_enc', action='store_true', help='initialize encoder')
        self.parser.add_argument('--path_to_pre_trained_Task', type=str, default=None, help='path to pre_trained_Task')
        self.parser.add_argument('--path_to_pre_trained_G_s2t', type=str, default=None, help='path to pre_trained_G_s2t')
        self.parser.add_argument('--path_to_pre_trained_G_t2s', type=str, default=None, help='path to pre_trained_G_t2s')
        self.parser.add_argument('--path_to_pre_trained_D_t', type=str, default=None, help='path to pre_trained_D_t')
        self.parser.add_argument('--path_to_pre_trained_D_s', type=str, default=None, help='path to pre_trained_D_s')
        self.parser.add_argument('--data_list_source', type=str, default=None, help='path to data list for source domain')
        self.parser.add_argument('--data_list_target', type=str, default=None, help='path to data list for target domain')
        self.parser.add_argument('--experiment', action='store_true')
        self.parser.add_argument('--dataset', type=str, default='nyu', help='nyu or kitti' )
        self.parser.add_argument('--distributed', help='Use DDP', action='store_true')
        self.parser.add_argument('--fp16', help='Use FP16 training', action='store_true')


    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt=self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >=0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids):
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('--------------Options--------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.experiment_name)
        util.mkdirs(expr_dir)
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')

        return self.opt