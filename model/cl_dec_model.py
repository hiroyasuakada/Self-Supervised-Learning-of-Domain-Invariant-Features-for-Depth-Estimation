import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
import util.task as task
from util.GP import GPStruct
from .base_model import BaseModel
from . import network
from util.loss import ssim

class CLDecModel(BaseModel):
    def name(self):
        return 'CL Dec model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = [
            'SimSiam_dec_s', 
            'SimSiam_dec_t',
            ]
        
        self.visual_names = [
            'img_s', 'img_t', 
            'img_s2t', 'img_t2s',
            ]
       
        self.model_names = [
            'Task', 
            'SimSiam_dec'
            ]

        # define the transform network
        self.netTask = network.define_Task(
            opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
            opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
            False, opt.gpu_ids, opt.U_weight, opt.pre_trained_on_ImageNet
            )

        # self.netSimSiam_dec = network.define_SimSiamHead(
        #     in_dim=12, projector_hidden_dim=12, projector_out_dim=12, 
        #     predictor_hidden_dim=4, num_layer=3, init_type=opt.init_type, 
        #     gpu_ids=opt.gpu_ids, mode=opt.simsiam_mode
        #     )

        # specify id to return features from Unet: '-1' for deconv2, '-2' for deconv3, '-3' for deconv4, '-4' for deconv5, '-5' for center_out
        self.unet_feature_id = opt.unet_feature_id

        if self.opt.unet_feature_id == -1:
            in_dim=12 
            projector_hidden_dim=12
            projector_out_dim=12
            predictor_hidden_dim=12//opt.scale_simsiam_hidden # best result with opt.scale_simsiam_hidden = 3

        elif self.opt.unet_feature_id == -2:
            in_dim=24
            projector_hidden_dim=24
            projector_out_dim=24
            predictor_hidden_dim=24//opt.scale_simsiam_hidden
        
        elif self.opt.unet_feature_id == -3:
            in_dim=40
            projector_hidden_dim=40
            projector_out_dim=40
            predictor_hidden_dim=40//opt.scale_simsiam_hidden
        
        elif self.opt.unet_feature_id == -4:
            in_dim=64
            projector_hidden_dim=64
            projector_out_dim=64
            predictor_hidden_dim=64//opt.scale_simsiam_hidden
        
        elif self.opt.unet_feature_id == -5:
            in_dim=176
            projector_hidden_dim=176
            projector_out_dim=176
            predictor_hidden_dim=176//opt.scale_simsiam_hidden

        self.netSimSiam_dec = network.define_SimSiamHead(
            in_dim=in_dim, projector_hidden_dim=projector_hidden_dim, projector_out_dim=projector_out_dim, 
            predictor_hidden_dim=predictor_hidden_dim, num_layer=3, init_type=opt.init_type, 
            gpu_ids=opt.gpu_ids, mode=opt.simsiam_mode
            )

            # self.netSimSiam_dec = network.define_SimSiamHead(
            #     in_dim=32, projector_hidden_dim=32, projector_out_dim=32, 
            #     predictor_hidden_dim=8, num_layer=2, init_type=opt.init_type, 
            #     gpu_ids=opt.gpu_ids, mode=opt.simsiam_mode
            #     )

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
        network._unfreeze(self.netTask.module.decoder, self.netSimSiam_dec)

        network._freeze(self.netTask.module.encoder)
        self.netTask.module.encoder.apply(network.freeze_bn)

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

                self.optimizer_Task = torch.optim.Adam(
                    [
                        {'params': self.netTask.module.decoder.parameters()},
                    ], 
                    lr=opt.lr_task, betas=(0.95, 0.999)
                )

            self.optimizer_SimSiam_dec = torch.optim.Adam(
                self.netSimSiam_dec.parameters(), lr=opt.lr_task, betas=(0.95, 0.999)
                )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Task)
            self.optimizers.append(self.optimizer_SimSiam_dec)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source'].cuda(self.gpu_ids[0])
        self.img_target = input['img_target'].cuda(self.gpu_ids[0])

    def forward(self):
        self.img_s = Variable(self.img_source)
        self.img_t = Variable(self.img_target)

        outputs_s2t = self.netG_s2t(self.img_s)
        self.img_s2t = outputs_s2t[-1]
        outputs_t2s = self.netG_t2s(self.img_t)
        self.img_t2s =outputs_t2s[-1]

    def backward_cl_dec(self):
        # process with synthetic images       
        latent_dec_s = self.netTask(self.img_s, return_features=True)[self.unet_feature_id]
        latent_dec_s2t = self.netTask(self.img_s2t.detach(), return_features=True)[self.unet_feature_id]

        loss_SimSiam_dec_s = (self.netSimSiam_dec(latent_dec_s, latent_dec_s2t))
        self.loss_SimSiam_dec_s = loss_SimSiam_dec_s.mean() * self.opt.lambda_simsiam

        total_loss_dec_s = self.loss_SimSiam_dec_s
        total_loss_dec_s.backward()
        del total_loss_dec_s

        # process with real images
        latent_dec_t = self.netTask(self.img_t, return_features=True)[self.unet_feature_id]
        latent_dec_t2s = self.netTask(self.img_t2s.detach(), return_features=True)[self.unet_feature_id]

        loss_SimSiam_dec_t = (self.netSimSiam_dec(latent_dec_t, latent_dec_t2s))
        self.loss_SimSiam_dec_t = loss_SimSiam_dec_t.mean() * self.opt.lambda_simsiam

        total_loss_dec_t = self.loss_SimSiam_dec_t
        total_loss_dec_t.backward()
        del total_loss_dec_t

    def optimize_parameters(self, epoch):

        self.forward()

        self.optimizer_Task.zero_grad()
        self.optimizer_SimSiam_dec.zero_grad()

        self.backward_cl_dec()
        
        self.optimizer_Task.step()
        self.optimizer_SimSiam_dec.step()

