import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from util.GP import GPStruct
from .base_model import BaseModel
from . import network
from util.loss import ssim


class FullFineModel(BaseModel):
    def name(self):
        return 'Full Fine model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = [
            'task_s2t', 'task_s', 'smooth_t', 'smooth_t2s', 'crdoco'
            ]
        
        self.visual_names = [
            'img_s', 'img_t', 'img_s2t', 'img_t2s',
            'depth_s2t', 'depth_t', 'depth_t2s', 'depth_s', 'lab_s', 'lab_t'
            ]
       
        if self.isTrain:
            self.model_names = ['Task', 'G_s2t', 'G_t2s']
        else:
            self.model_names = ['Task', 'G_s2t']

        # define the transform network
        self.netTask = network.define_Task(
            opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
            opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
            False, opt.gpu_ids, opt.U_weight, opt.pre_trained_on_ImageNet, opt
            )

        self.netG_s2t = network.define_G(
            opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
            opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
            False, opt.gpu_ids, opt.U_weight
            )

        self.netG_t2s = network.define_G(
            opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
            opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
            False, opt.gpu_ids, opt.U_weight
            )
    
        if opt.path_to_pre_trained_Task is not None:
            self.load_networks(
                net=self.netTask, 
                path_to_pre_trained_weights=opt.path_to_pre_trained_Task
                )

        if opt.path_to_pre_trained_G_s2t is not None:
            self.load_networks(
                net=self.netG_s2t, 
                path_to_pre_trained_weights=opt.path_to_pre_trained_G_s2t
                )

        if opt.path_to_pre_trained_G_t2s is not None:
            self.load_networks(
                net=self.netG_t2s, 
                path_to_pre_trained_weights=opt.path_to_pre_trained_G_t2s
                )
        

        # unfreeze and freeze layers
        network._unfreeze(self.netTask)
        # # # # self.netTask.module.encoder.apply(network.freeze_bn)

        if opt.path_to_pre_trained_Task:
            self.netTask.apply(network.freeze_bn)

            # # unfreeze batch norm for some layers
            # self.netTask.module.decoder.deconv2.apply(network.unfreeze_bn)  # -2
            # print('unfreeze deconv2')

            # self.netTask.module.decoder.deconv3.apply(network.unfreeze_bn)  # -3
            # print('unfreeze deconv3')

            # self.netTask.module.decoder.deconv4.apply(network.unfreeze_bn)  # -3
            # print('unfreeze deconv4')


        network._freeze(self.netG_s2t, self.netG_t2s)
        self.netG_s2t.apply(network.freeze_bn)
        self.netG_t2s.apply(network.freeze_bn)

        if self.isTrain:

            # create image pool
            self.fake_img_t_pool = ImagePool(opt.pool_size)
            self.fake_img_s_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.l1loss = torch.nn.L1Loss()
            self.criterionCrDoCo = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            self.nonlinearity = torch.nn.ReLU()

            # initialize optimizers
            if opt.pre_trained_on_ImageNet is False:
                self.optimizer_Task = torch.optim.Adam(
                    self.netTask.parameters(), lr=opt.lr_task, betas=(0.95, 0.999))
            else:
                Task_encoder_params = [param for param in self.netTask.module.encoder.parameters()]

                self.optimizer_Task = torch.optim.Adam(
                    [
                        {'params': self.netTask.module.decoder.parameters()},
                        {'params': Task_encoder_params[:163], 'lr': opt.lr_task / 1000},
                        {'params': Task_encoder_params[163:345], 'lr': opt.lr_task / 100},
                        {'params': Task_encoder_params[345:], 'lr': opt.lr_task / 10},
                        # {'params': Task_encoder_params[504:], 'lr': opt.lr_task},
                    ], 
                    lr=opt.lr_task, betas=(0.95, 0.999)
                )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Task)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source'].cuda(self.gpu_ids[0])
        self.img_target = input['img_target'].cuda(self.gpu_ids[0])
        self.lab_source = input['lab_source'].cuda(self.gpu_ids[0])
        self.lab_target = input['lab_target'].cuda(self.gpu_ids[0])

    def forward(self):
        self.img_s = Variable(self.img_source)
        self.img_t = Variable(self.img_target)
        self.lab_s = Variable(self.lab_source)
        self.lab_t = Variable(self.lab_target)

        outputs_s2t = self.netG_s2t(self.img_s)
        self.img_s2t = outputs_s2t[-1]
        outputs_t2s = self.netG_t2s(self.img_t)
        self.img_t2s =outputs_t2s[-1]

    def backward_T_stylizedsynthesis2depth(self):

        # task network

        outputs_s2t = self.netTask.forward(self.img_s2t)

        size=len(outputs_s2t)
        self.feat_s2t_task = outputs_s2t[0]
        self.depth_s2t = outputs_s2t[1:]

        # task loss
        lab_real = task.scale_pyramid(self.lab_s, size-1)
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.depth_s2t, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_task_s2t = task_loss * self.opt.lambda_task

        total_loss = self.loss_task_s2t
        total_loss.backward()

    def backward_T_real2depth(self):

        # image2depth

        outputs_t = self.netTask.forward(self.img_t)

        size = len(outputs_t)
        self.feat_t_task = outputs_t[0]
        self.depth_t = outputs_t[1:]

        img_real = task.scale_pyramid(self.img_t, size - 1)
        self.loss_smooth_t = task.get_smooth_weight(self.depth_t, img_real, size-1) * self.opt.lambda_smooth

        total_loss = self.loss_smooth_t
        total_loss.backward(retain_graph=True)

    def backward_S_synthesis2depth(self):

        # task network

        outputs_s = self.netTask.forward(self.img_s)

        size=len(outputs_s)
        self.feat_s_task = outputs_s[0]
        self.depth_s = outputs_s[1:]

        # task loss
        lab_real = task.scale_pyramid(self.lab_s, size-1)
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.depth_s, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_task_s = task_loss * self.opt.lambda_task

        total_loss = self.loss_task_s
        total_loss.backward()
        del total_loss

    def backward_S_stylizedreal2depth(self):

        # task network

        outputs_t2s = self.netTask.forward(self.img_t2s)

        size = len(outputs_t2s)
        self.feat_t2s_task = outputs_t2s[0]
        self.depth_t2s = outputs_t2s[1:]

        img_real = task.scale_pyramid(self.img_t2s.detach(), size - 1)
        self.loss_smooth_t2s = task.get_smooth_weight(self.depth_t2s, img_real, size-1) * self.opt.lambda_smooth

        total_loss = self.loss_smooth_t2s
        total_loss.backward(retain_graph=True)

################################################
##### Source and Target Domain for CrDoCo ######
################################################

    def backward_S_T_crdoco(self):
        
        loss_crdoco = self.criterionCrDoCo(self.depth_t2s[-1], self.depth_t[-1])
        self.loss_crdoco = loss_crdoco * self.opt.lambda_crdoco

        total_loss = self.loss_crdoco
        total_loss.backward()
        del total_loss

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # T2Net
        self.optimizer_Task.zero_grad()

        self.backward_T_stylizedsynthesis2depth()
        self.backward_T_real2depth()

        self.backward_S_stylizedreal2depth()
        self.backward_S_synthesis2depth()

        self.backward_S_T_crdoco()

        self.optimizer_Task.step()


