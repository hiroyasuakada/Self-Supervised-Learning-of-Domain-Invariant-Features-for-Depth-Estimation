import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from util.GP import GPStruct
from .base_model import BaseModel
from . import network
from util.loss import ssim

class FullModel(BaseModel):
    def name(self):
        return 'Full model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = [
            'G_t', 'G_s', 'D_t', 'D_s', 'idt_t', 'idt_s', 
            'cycle_s2t2s', 'cycle_t2s2t', 'crdoco',
            'task_s2t', 'task_s', 'smooth_t', 'smooth_t2s', 
            ]
        
        self.visual_names = [
            'img_s', 'img_t', 'img_s2t', 'img_t2t', 'img_t2s', 'img_s2s', 
            'depth_s2t', 'depth_t', 'depth_t2s', 'depth_s', 'lab_s', 'lab_t'
            ]
       
        if self.isTrain:
            self.model_names = ['Task', 'G_s2t', 'G_t2s', 'D_t', 'D_s']
        else:
            self.model_names = ['Task', 'G_s2t']

        # define the transform network
        self.netTask = network.define_Task(
            opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
            opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
            False, opt.gpu_ids, opt.U_weight, opt.pre_trained_on_ImageNet
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
    
        self.netD_t = network.define_D(
            opt.image_nc, opt.ndf, opt.image_D_layers, opt.num_D, opt.norm,
            opt.activation, opt.init_type, opt.gpu_ids
            )

        self.netD_s = network.define_D(
            opt.image_nc, opt.ndf, opt.image_D_layers, opt.num_D, opt.norm,
            opt.activation, opt.init_type, opt.gpu_ids
            )

        # self.net_f_D = network.define_featureD(
        #     opt.image_feature, opt.feature_D_layers, opt.norm,
        #     opt.activation, opt.init_type, opt.gpu_ids
        #     )


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
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(filter(lambda p: p.requires_grad, self.netG_s2t.parameters()),
                                filter(lambda p: p.requires_grad, self.netG_t2s.parameters())),
                                lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(filter(lambda p: p.requires_grad, self.netD_t.parameters()),
                                filter(lambda p: p.requires_grad, self.netD_s.parameters())),
                                lr=opt.lr_trans, betas=(0.5, 0.9))

            if opt.pre_trained_on_ImageNet is False:
                self.optimizer_Task = torch.optim.Adam(
                    self.netTask.parameters(), lr=opt.lr_task, betas=(0.95, 0.999))
            else:
                # Task_encoder_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0], self.netTask.module.encoder.named_parameters()))))
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
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
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

    def backward_D_basic(self, netD, real, fake):

        loss_D = 0
        for (real_i, fake_i) in zip(real, fake):
            # Real
            D_real = netD(real_i.detach())
            # fake
            D_fake = netD(fake_i.detach())

            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                loss_D += (torch.mean((D_real_i-1.0)**2) + torch.mean((D_fake_i -0.0)**2))*0.5

        return loss_D

    def backward_T_D(self):
        network._freeze(self.netG_s2t, self.netTask)
        network._unfreeze(self.netD_t)
        size = len(self.img_s2t)
        fake = []
        for i in range(size):
            fake.append(self.fake_img_t_pool.query(self.img_s2t[i]))
        real = task.scale_pyramid(self.img_t, size)
        self.loss_D_t = self.backward_D_basic(self.netD_t, real, fake)
        
        total_loss = self.loss_D_t
        total_loss.backward()
        del total_loss

    def backward_S_D(self):
        network._freeze(self.netG_t2s, self.netTask)
        network._unfreeze(self.netD_s)
        size = len(self.img_t2s)
        fake = []
        for i in range(size):
            fake.append(self.fake_img_s_pool.query(self.img_t2s[i]))
        real = task.scale_pyramid(self.img_s, size)
        self.loss_D_s = self.backward_D_basic(self.netD_s, real, fake)

        total_loss = self.loss_D_s
        total_loss.backward()
        del total_loss

    def foreward_G_basic(self, net_G, img_s, img_t):

        img = torch.cat([img_s, img_t], 0)
        fake = net_G(img)

        size = len(fake)

        f_s, f_t = fake[0].chunk(2)
        img_fake = fake[1:]

        img_s_fake = []
        img_t_fake = []

        for img_fake_i in img_fake:
            img_s, img_t = img_fake_i.chunk(2)
            img_s_fake.append(img_s)
            img_t_fake.append(img_t)

        return img_s_fake, img_t_fake, f_s, f_t, size


##########################
##### Target Domain ######
##########################

    def backward_T_synthesis2real(self):

        # image to image transform
        network._freeze(self.netTask, self.netD_t, self.netD_s)
        network._unfreeze(self.netG_s2t, self.netG_t2s)
        self.img_s2t, self.img_t2t, _, _, size = \
            self.foreward_G_basic(self.netG_s2t, self.img_s, self.img_t)

        # cycle consistency
        self.img_s2t2s = self.netG_t2s(self.img_s2t[-1])
        
        loss_cycle_s2t2s = self.criterionCycle(self.img_s2t2s[-1], self.img_s)
        self.loss_cycle_s2t2s = loss_cycle_s2t2s * self.opt.lambda_cycle

        # image GAN loss and reconstruction loss
        img_real = task.scale_pyramid(self.img_t, size - 1)
        loss_G_t = 0
        loss_idt_t = 0
        for i in range(size - 1):
            loss_idt_t += self.l1loss(self.img_t2t[i], img_real[i])
            D_fake = self.netD_t(self.img_s2t[i])
            for D_fake_i in D_fake:
                loss_G_t += torch.mean((D_fake_i - 1.0) ** 2)

        self.loss_G_t = loss_G_t * self.opt.lambda_gan_img
        self.loss_idt_t = loss_idt_t * self.opt.lambda_identity

        total_loss = self.loss_G_t + self.loss_idt_t + self.loss_cycle_s2t2s
        total_loss.backward(retain_graph=True)

    def backward_T_stylizedsynthesis2depth(self):

        # task network
        network._freeze(self.netD_t)
        network._unfreeze(self.netG_s2t, self.netTask)
        outputs_s2t = self.netTask.forward(self.img_s2t[-1])

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
        network._freeze(self.netG_s2t, self.netD_t)
        network._unfreeze(self.netTask)
        outputs_t = self.netTask.forward(self.img_t)

        size = len(outputs_t)
        self.feat_t_task = outputs_t[0]
        self.depth_t = outputs_t[1:]

        img_real = task.scale_pyramid(self.img_t, size - 1)
        self.loss_smooth_t = task.get_smooth_weight(self.depth_t, img_real, size-1) * self.opt.lambda_smooth

        total_loss = self.loss_smooth_t
        total_loss.backward(retain_graph=True)

##########################
##### Source Domain ######
##########################

    def backward_S_real2synthesis(self):

        # image to image transform
        network._freeze(self.netTask, self.netD_s, self.netD_t)
        network._unfreeze(self.netG_t2s, self.netG_s2t)
        self.img_t2s, self.img_s2s, _, _, size = \
            self.foreward_G_basic(self.netG_t2s, self.img_t, self.img_s)

        # cycle consistency
        self.img_t2s2t = self.netG_s2t(self.img_t2s[-1])

        loss_cycle_t2s2t = self.criterionCycle(self.img_t2s2t[-1], self.img_t)
        self.loss_cycle_t2s2t = loss_cycle_t2s2t * self.opt.lambda_cycle

        # image GAN loss and reconstruction loss
        img_real = task.scale_pyramid(self.img_s, size - 1)
        loss_G_s = 0
        loss_idt_s = 0
        for i in range(size - 1):
            loss_idt_s += self.l1loss(self.img_s2s[i], img_real[i])
            D_fake = self.netD_s(self.img_t2s[i])
            for D_fake_i in D_fake:
                loss_G_s += torch.mean((D_fake_i - 1.0) ** 2)

        self.loss_G_s = loss_G_s * self.opt.lambda_gan_img
        self.loss_idt_s = loss_idt_s * self.opt.lambda_identity

        total_loss = self.loss_G_s + self.loss_idt_s + self.loss_cycle_t2s2t
        total_loss.backward(retain_graph=True)

    def backward_S_synthesis2depth(self):

        # task network
        network._freeze(self.netG_t2s, self.netD_s)
        network._unfreeze(self.netTask)
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
        network._freeze(self.netD_s, self.netG_t2s)
        network._unfreeze(self.netTask)
        outputs_t2s = self.netTask.forward(self.img_t2s[-1])

        size = len(outputs_t2s)
        self.feat_t2s_task = outputs_t2s[0]
        self.depth_t2s = outputs_t2s[1:]

        img_real = task.scale_pyramid(self.img_t2s[-1].detach(), size - 1)
        self.loss_smooth_t2s = task.get_smooth_weight(self.depth_t2s, img_real, size-1) * self.opt.lambda_smooth

        total_loss = self.loss_smooth_t2s
        total_loss.backward(retain_graph=True)

################################################
##### Source and Target Domain for CrDoCo ######
################################################

    def backward_S_T_crdoco(self):
        
        network._freeze(self.netD_s, self.netD_t)
        network._unfreeze(self.netG_t2s, self.netTask)

        loss_crdoco = self.criterionCrDoCo(self.depth_t2s[-1], self.depth_t[-1])
        self.loss_crdoco = loss_crdoco * self.opt.lambda_crdoco

        total_loss = self.loss_crdoco
        total_loss.backward()
        del total_loss

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # T2Net
        self.optimizer_Task.zero_grad()
        self.optimizer_G.zero_grad()

        self.backward_T_synthesis2real()
        self.backward_T_stylizedsynthesis2depth()
        self.backward_T_real2depth()

        self.backward_S_real2synthesis()
        self.backward_S_stylizedreal2depth()
        self.backward_S_synthesis2depth()

        self.backward_S_T_crdoco()

        self.optimizer_Task.step()
        self.optimizer_G.step()

        # Discriminator
        self.optimizer_D.zero_grad()

        self.backward_T_D()
        self.backward_S_D()

        self.optimizer_D.step()

