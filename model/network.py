import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler
import efficientnet_pytorch


######################################################################################
# Functions
######################################################################################
def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch+1+1+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            lr_l = 1.0 - max(0, epoch+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, gain, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('total number of parameters of {}: {:.3f} M'.format(name, num_params / 1e6))


def init_net(net, init_type='normal', gpu_ids=[], pre_trained_enc=False, init_enc=False):

    # print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()

    if pre_trained_enc is False:
        init_weights(net, init_type)
    elif init_enc is True:
        init_weights(net, init_type)
    else:
        if isinstance(net, torch.nn.DataParallel):
            init_weights(net.module.decoder, init_type)
        else:
            init_weights(net.decoder, init_type)
        print('initialized only decoder')

    return net


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True

def freeze_bn_affine(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False


# define the generator(transform) network
def define_G(input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', model_type='UNet',
                    init_type='xavier', drop_rate=0, add_noise=False, gpu_ids=[], weight=0.1):

    if model_type == 'ResNet':
        net = _ResGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids)
    elif model_type == 'UNet':
        net = _UNetGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)
    else:
        raise NotImplementedError('model type [%s] is not implemented', model_type)

    print_network(net, name='G')

    return init_net(net, init_type, gpu_ids)


# define the discriminator network
def define_D(input_nc, ndf = 64, n_layers = 3, num_D = 1, norm = 'batch', activation = 'PReLU', init_type='xavier', gpu_ids = []):

    net = _MultiscaleDiscriminator(input_nc, ndf, n_layers, num_D, norm, activation, gpu_ids)

    print_network(net, name='D_img')

    return init_net(net, init_type, gpu_ids)


# define the feature discriminator network
def define_featureD(input_nc, n_layers=2, norm='batch', activation='PReLU', init_type='xavier', gpu_ids=[]):

    net = _FeatureDiscriminator(input_nc, n_layers, norm, activation, gpu_ids)

    print_network(net, name='D_feat')

    return init_net(net, init_type, gpu_ids)


# define the task network
def define_Task(input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', model_type='UNet',
                    init_type='xavier', drop_rate=0, add_noise=False, gpu_ids=[], weight=0.1, pre_trained_enc=False, opt=None):

    if model_type == 'ResNet':
        net = _ResGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids)
    elif model_type == 'UNet':
        if pre_trained_enc is False:
            net = _UNetGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)
        else:
            net = UNetEB5Model(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)
    
    else:
        raise NotImplementedError('model type [%s] is not implemented', model_type)

    print_network(net, name='Task')

    return init_net(net, init_type, gpu_ids, pre_trained_enc, opt.init_enc)


# define the simsiam network
def define_SimSiamHead(in_dim=2048, projector_hidden_dim=2048, projector_out_dim=2048, 
            predictor_hidden_dim=512, num_layer=3, init_type='xavier', gpu_ids=[], mode='normal', opt=None):

    if mode == 'normal':
        net = SimSiamHead(in_dim, projector_hidden_dim, projector_out_dim, predictor_hidden_dim, num_layer)
        print_network(net, name='SimSiam Head: normal mode')

    elif mode == 'channel':
        net = SimSiamHead_channel(in_dim, projector_hidden_dim, projector_out_dim, predictor_hidden_dim, num_layer)
        print_network(net, name='SimSiam Head: channel wise mode')

    elif mode == 'patch':
        net = SimSiamHead_patch(in_dim, projector_hidden_dim, projector_out_dim, predictor_hidden_dim, num_layer)
        print_network(net, name='SimSiam Head: patch mode')

    elif mode == 'patch_middle':
        net = SimSiamHead_patch_middle(in_dim, projector_hidden_dim, projector_out_dim, predictor_hidden_dim, num_layer)
        print_network(net, name='SimSiam Head: patch middle mode')

    return init_net(net, init_type, gpu_ids)

######################################################################################
# Basic Operation
######################################################################################


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(x.data.get_device()) - 0.5) / 10.0)
        return x+noise


class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
            )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc * width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc * width, output_nc, kernel_size=3, padding=0, bias=use_bias)
        )

    def forward(self, x):
        result = []
        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output+x)


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DownBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity,
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _ShuffleUpBlock(nn.Module):
    def __init__(self, input_nc, up_scale, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_ShuffleUpBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, input_nc*up_scale**2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.PixelShuffle(up_scale),
            nonlinearity,
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


######################################################################################
# Network structure
######################################################################################

class _ResGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[]):
        super(_ResGenerator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        ]

        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (i+1), 2)
            encoder += [
                _EncoderBlock(ngf * mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_bias),
                nn.AvgPool2d(kernel_size=2, stride=2)
            ]

        mult = min(2 ** n_downsampling, 2)
        for i in range(n_blocks-n_downsampling):
            encoder +=[
                _InceptionBlock(ngf*mult, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, width=1,
                                drop_rate=drop_rate, use_bias=use_bias)
            ]

        decoder = []
        if add_noise:
            decoder += [GaussianNoiseLayer()]

        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (n_downsampling - i -1), 2)
            decoder +=[
                _DecoderUpBlock(ngf*mult_prev, ngf*mult_prev, ngf*mult, norm_layer, nonlinearity, use_bias),
            ]

        decoder +=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        feature = self.encoder(input)
        result = [feature]
        output = self.decoder(feature)
        result.append(output)
        return result


class _UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetGenerator, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        for i in range(layers-4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center=[]
        for i in range(7-layers):
            center +=[
                _InceptionBlock(ngf*8, ngf*8, norm_layer, nonlinearity, 7-layers, drop_rate, use_bias)
            ]

        center += [
        _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        ]
        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input, gp=False):
        conv1 = self.pool(self.conv1(input))
        conv2 = self.pool(self.conv2.forward(conv1))
        conv3 = self.pool(self.conv3.forward(conv2))
        center_in = self.pool(self.conv4.forward(conv3))

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)

        center_out = self.center.forward(center_in)
        result = [center_in]

        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        if not gp:
            deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
            output4 = self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))
            result.append(output4)
            deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
            output3 = self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
            result.append(output3)
            deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
            output2 = self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
            result.append(output2)
            output1 = self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
            result.append(output1)
            return result
        else:
            return center_out


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
# 
# add simsiam head
#
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################


class SimSiamHead(nn.Module):
    def __init__(self, in_dim=2048, projector_hidden_dim=2048, projector_out_dim=2048, predictor_hidden_dim=512, num_layer=3):
        super().__init__()
        predictor_out_dim = projector_out_dim

        if num_layer == 3:
            self.projector = nn.Sequential(
                nn.Conv2d(in_dim, projector_hidden_dim, 1, 1),
                nn.BatchNorm2d(projector_hidden_dim),
                nn.ReLU(),

                nn.Conv2d(projector_hidden_dim, projector_hidden_dim, 1, 1),
                nn.BatchNorm2d(projector_hidden_dim),
                nn.ReLU(),

                nn.Conv2d(projector_hidden_dim, projector_out_dim, 1, 1),
                nn.BatchNorm2d(projector_out_dim),
            )

        elif num_layer == 2:
            self.projector = nn.Sequential(
                    nn.Conv2d(in_dim, projector_hidden_dim, 1, 1),
                    nn.BatchNorm2d(projector_hidden_dim),
                    nn.ReLU(),

                    nn.Conv2d(projector_hidden_dim, projector_out_dim, 1, 1),
                    nn.BatchNorm2d(projector_out_dim),
                ) 

        self.predictor = nn.Sequential(
            nn.Conv2d(projector_out_dim, predictor_hidden_dim, 1, 1),
            nn.BatchNorm2d(predictor_hidden_dim),
            nn.ReLU(),

            nn.Conv2d(predictor_hidden_dim, predictor_out_dim, 1, 1)
        )

    def forward(self, latent1, latent2):
        z1, z2 = self.projector(latent1), self.projector(latent2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = 0.5 * (self.sim(p1, z2) + self.sim(p2, z1))
        return L


    def sim(self, p, z):
        return -nn.functional.cosine_similarity(p, z.detach(), dim=1).mean()


class SimSiamHead_channel(nn.Module):
    def __init__(self, in_dim=12, projector_hidden_dim=12, projector_out_dim=12, predictor_hidden_dim=4, num_layer=3):
        super().__init__()

        self.scale_factor_to_predictor_hidden_dim = 4  # predictor_hidden_dim = 4

        self.in_dim = int(in_dim / self.scale_factor_to_predictor_hidden_dim)  # 3
        self.projector_hidden_dim = int(projector_hidden_dim / self.scale_factor_to_predictor_hidden_dim)  # 3
        self.projector_out_dim = int(projector_out_dim / self.scale_factor_to_predictor_hidden_dim)  # 3
        self.predictor_hidden_dim = int(predictor_hidden_dim / self.scale_factor_to_predictor_hidden_dim)  # 1

        self.id_list = [i for i in range(12)]
        for id in self.id_list:
            setattr(self, 'head_' + str(id), SimSiamHead(in_dim=self.in_dim, projector_hidden_dim=self.projector_hidden_dim, projector_out_dim=self.projector_out_dim, predictor_hidden_dim=self.predictor_hidden_dim, num_layer=num_layer))

    def forward(self, latent1, latent2):
        B, C, H, W = latent1.size()
        L = torch.tensor(0.0).cuda(latent1.device)

        for id in self.id_list:
            head_id = getattr(self, 'head_' + str(id))
            latent1_id = latent1[:, id, :, :].view(B, self.in_dim, H, int(W/self.in_dim))
            latent2_id = latent2[:, id, :, :].view(B, self.in_dim, H, int(W/self.in_dim))
            L += head_id(latent1_id, latent2_id)

        return L


def get_feat_patches(x, kc, kh, kw, dc, dh, dw):
    # x = x.unsqueeze(0)  # (256, 24, 32) to (1, 256, 24, 32)
    B, C, H, W = x.size()
    patches = x.unfold(1, int(kc), int(dc)).unfold(2, int(kh), int(dh)).unfold(3, int(kw), int(dw))
    patches = patches.contiguous().view(B, -1, int(kc), int(kh), int(kw))

    return patches

    # tensor_vec = get_feat_patches( # (256, 24, 32) to (1, 256, 24, 32) to (4, 256, 12, 16)
    #     tensor, 
    #     kc=self.z_numchnls, kh=self.z_height/self.split_scale, kw=self.z_width/self.split_scale, 
    #     dc=1, dh=self.z_height/self.split_scale, dw=self.z_width/self.split_scale
    #     )
    # tensor_vec = tensor_vec.view(-1,int(self.z_numchnls*self.z_height*self.z_width/self.split_scale/self.split_scale))  # z tensor to a vector (4, 256*12*16)


class SimSiamHead_patch(nn.Module):
    def __init__(self, in_dim=12, projector_hidden_dim=12, projector_out_dim=12, predictor_hidden_dim=4, num_layer=3):
        super().__init__()
        self.in_dim = in_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim
        self.predictor_hidden_dim = predictor_hidden_dim

        self.id_list = [i for i in range(3*10)]
        for id in self.id_list:
            setattr(self, 'head_' + str(id), SimSiamHead(in_dim=self.in_dim, projector_hidden_dim=self.projector_hidden_dim, projector_out_dim=self.projector_out_dim, predictor_hidden_dim=self.predictor_hidden_dim, num_layer=num_layer))

    def forward(self, latent1, latent2):
        # (B, 12, 288, 960) to (B, 30, 12, 96, 96): 30 patches
        latent1 = get_feat_patches(latent1, 12, 96, 96, 1, 96, 96)
        latent2 = get_feat_patches(latent2, 12, 96, 96, 1, 96, 96)

        L = torch.tensor(0.0).cuda(latent1.device)

        for id in self.id_list:
            head_id = getattr(self, 'head_' + str(id))
            latent1_id = latent1[:, id, :, :, :]
            latent2_id = latent2[:, id, :, :, :]
            L += head_id(latent1_id, latent2_id) / 30

        return L


class SimSiamHead_patch_middle(nn.Module):
    def __init__(self, in_dim=2048, projector_hidden_dim=2048, projector_out_dim=2048, predictor_hidden_dim=512, num_layer=3):
        super().__init__()
        predictor_out_dim = projector_out_dim

        self.in_dim = in_dim
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim
        self.predictor_hidden_dim = predictor_hidden_dim

        if num_layer == 3:
            self.projector = nn.Sequential(
                nn.Conv2d(in_dim, projector_hidden_dim, 1, 1),
                nn.BatchNorm2d(projector_hidden_dim),
                nn.ReLU(),

                nn.Conv2d(projector_hidden_dim, projector_hidden_dim, 1, 1),
                nn.BatchNorm2d(projector_hidden_dim),
                nn.ReLU(),

                nn.Conv2d(projector_hidden_dim, projector_out_dim, 1, 1),
                nn.BatchNorm2d(projector_out_dim),
            )

        elif num_layer == 2:
            self.projector = nn.Sequential(
                    nn.Conv2d(in_dim, projector_hidden_dim, 1, 1),
                    nn.BatchNorm2d(projector_hidden_dim),
                    nn.ReLU(),

                    nn.Conv2d(projector_hidden_dim, projector_out_dim, 1, 1),
                    nn.BatchNorm2d(projector_out_dim),
                ) 


        self.id_list = [i for i in range(3*10)]
        for id in self.id_list:
            setattr(self, 'head_' + str(id), Projector_id(in_dim=self.in_dim, projector_hidden_dim=self.projector_hidden_dim, projector_out_dim=self.projector_out_dim, predictor_hidden_dim=self.predictor_hidden_dim, num_layer=num_layer))

        self.predictor = nn.Sequential(
            nn.Conv2d(projector_out_dim, predictor_hidden_dim, 1, 1),
            nn.BatchNorm2d(predictor_hidden_dim),
            nn.ReLU(),

            nn.Conv2d(predictor_hidden_dim, predictor_out_dim, 1, 1)
        )

    def forward(self, latent1, latent2):
        z1, z2 = self.projector(latent1), self.projector(latent2)

        # (B, 12, 288, 960) to (B, 30, 12, 96, 96): 30 patches
        z1 = get_feat_patches(z1, 12, 96, 96, 1, 96, 96)
        z2 = get_feat_patches(z2, 12, 96, 96, 1, 96, 96)

        L = torch.tensor(0.0).cuda(z1.device)

        for id in self.id_list:
            head_id = getattr(self, 'head_' + str(id))
            z1_id = z1[:, id, :, :, :]
            z2_id = z2[:, id, :, :, :]
            L += head_id(z1_id, z2_id) / 30

        return L


class Projector_id(nn.Module):
    def __init__(self, in_dim=2048, projector_hidden_dim=2048, projector_out_dim=2048, predictor_hidden_dim=512, num_layer=3):
        super().__init__()
        predictor_out_dim = projector_out_dim

        self.predictor = nn.Sequential(
            nn.Conv2d(projector_out_dim, predictor_hidden_dim, 1, 1),
            nn.BatchNorm2d(predictor_hidden_dim),
            nn.ReLU(),

            nn.Conv2d(predictor_hidden_dim, predictor_out_dim, 1, 1)
        )

    def forward(self, z1, z2):
        p1, p2 = self.predictor(z1), self.predictor(z2)
        L = 0.5 * (self.sim(p1, z2) + self.sim(p2, z1))

        return L

    def sim(self, p, z):
        return -nn.functional.cosine_similarity(p, z.detach(), dim=1).mean()


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
# 
# add ENB5 version task network
#
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################


class UNetEB5Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(UNetEB5Model, self).__init__()
        
        self.encoder = _UNetEB5Encoder(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)
        self.decoder = _UNetEB5Decoder(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)

    def forward(self, input, return_features=False, return_full=False):
        features = self.encoder(input)
        output = self.decoder(features, return_features=return_features, return_full=return_full)
        return output


class _UNetEB5Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetEB5Encoder, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        from efficientnet_pytorch import EfficientNet
        self.encoder_pre_trained = EfficientNet.from_pretrained('efficientnet-b5', advprop=True)
        # self.encoder_pre_trained = EfficientNet.from_name('efficientnet-b5')
        print('Removing last dense layer')
        self.encoder_pre_trained._fc = nn.Identity()

    def forward(self, input):
        features = self.encoder_pre_trained.extract_endpoints(input) 

        return features


class _UNetEB5Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetEB5Decoder, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        center=[]
        center += [_DecoderUpBlock(2048, 2048, 176, norm_layer, nonlinearity, use_bias)]
        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        self.deconv5 = _DecoderUpBlock(352, 352, 64, norm_layer, nonlinearity, use_bias)
        self.deconv4 = _DecoderUpBlock(128, 128, 40, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(80, 80, 24, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(48, 48, 12, norm_layer, nonlinearity, use_bias)

        self.output1 = _OutputBlock(12, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, features, return_features=False, return_full=False):
        conv1 = features['reduction_1']  
        conv2 = features['reduction_2']  
        conv3 = features['reduction_3']  
        conv4 = features['reduction_4']     
        conv5 = features['reduction_6']

        result = [conv5]
        center_out = self.center.forward(conv5)

        deconv5 = self.deconv5.forward(torch.cat([center_out, conv4], 1))
        deconv4 = self.deconv4.forward(torch.cat([deconv5, conv3], 1))
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2], 1))
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1], 1))

        if return_features:
            return [center_out, deconv5, deconv4, deconv3, deconv2]

        output1 = self.output1.forward(deconv2)
        result.append(output1)

        if return_full:
            return result, [center_out, deconv5, deconv4, deconv3, deconv2]

        return result


### Scalar Self Attention
class ScalarSelfAttention(nn.Module):
    """ Self-Attention Layer"""

    def __init__(self, in_dim):
        super(ScalarSelfAttention, self).__init__()

        # pointwise convolution
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-2)

        # output = x + gamma * o
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        X = x

        # B,C',W,H → B,C',N
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])  # B,C',N
        proj_query = proj_query.permute(0, 2, 1)  
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])  # B,C',N

        S = torch.bmm(proj_query, proj_key) 

        attention_map_T = self.softmax(S)  # row-dimentional softmax
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])  # B,C,N
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        
        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x + self.gamma * o

        return out, attention_map


class _UNetSAGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetSAGenerator, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        for i in range(layers-4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center=[]
        for i in range(7-layers):
            center +=[
                _InceptionBlock(ngf*8, ngf*8, norm_layer, nonlinearity, 7-layers, drop_rate, use_bias)
            ]

        center += [
        _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        ]

        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        # self attention layers
        self.sa1 = ScalarSelfAttention(ngf*4)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input, gp=False):
        conv1 = self.pool(self.conv1(input))
        conv2 = self.pool(self.conv2.forward(conv1))
        conv3 = self.pool(self.conv3.forward(conv2))
        center_in = self.pool(self.conv4.forward(conv3))

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)

        # self attention layers
        center_out, amap1 = self.sa1(center_out)

        result = [center_in]

        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        output4 = self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        result.append(output4)
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        result.append(output3)
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        result.append(output2)
        output1 = self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)

        if not gp:
            return result
        else:
            return result, center_out


class _MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=1, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.gpu_ids = gpu_ids

        for i in range(num_D):
            netD = _Discriminator(input_nc, ndf, n_layers, norm, activation, gpu_ids)
            setattr(self, 'scale'+str(i), netD)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, 'scale'+str(i))
            output = netD.forward(input)
            result.append(output)
            if i != (self.num_D-1):
                input = self.downsample(input)
        return result


class _Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_Discriminator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nonlinearity,
        ]

        nf_mult=1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nonlinearity,
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nonlinearity,
            nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class _FeatureDiscriminator(nn.Module):
    def __init__(self, input_nc, n_layers=2, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_FeatureDiscriminator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Linear(input_nc * 40 * 12, input_nc),
            # nn.Linear(input_nc * 16 * 12, input_nc),
            nonlinearity,
        ]

        for i in range(1, n_layers):
            model +=[
                nn.Linear(input_nc, input_nc),
                nonlinearity
            ]

        model +=[nn.Linear(input_nc, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        result = []
        input = input.view(-1, 512 * 40 * 12)
        # input = input.view(-1, 512 * 16 * 12)
        output = self.model(input)
        result.append(output)
        return result


if __name__ == "__main__":
    
    model = UNetEB5Model(3, 1)
    print_network(model)


    # model = UNetEB5Model(3, 1).cuda()
    # inputs = torch.rand(1, 3, 640, 480).cuda()
    
    # endpoints = model.encoder.encoder_pre_trained.extract_endpoints(inputs)
    # print(endpoints['reduction_1'].shape)  # torch.Size([1, 24, 320, 240])
    # print(endpoints['reduction_2'].shape)  # torch.Size([1, 40, 160, 120])
    # print(endpoints['reduction_3'].shape)  # torch.Size([1, 64, 80, 60])
    # print(endpoints['reduction_4'].shape)  # torch.Size([1, 176, 40, 30])
    # print(endpoints['reduction_5'].shape)  # torch.Size([1, 2048, 20, 15])

    # print(model.encoder)
    # print('================================================================')

    # i = 0
    # list_parameters = [name for name, _ in model.encoder.named_parameters()]
    # # for name, param in model.encoder.named_parameters():
    # #     list_parameters.append(name)
    # #     i += 1
    # # print(i)
    
    # for t in list_parameters[:163]:
    #     print(t)
    # print('================================================================')
    # for s in list_parameters[163:345]:
    #     print(s)
    # print('================================================================')
    # for p in list_parameters[345:504]:
    #     print(p)
    # print('================================================================')
    # for q in list_parameters[504:]:
    #     print(q)

    # list_parameters = [param for _, param in model.encoder.named_parameters()]

    # optimizer_Task = torch.optim.Adam(
    #     [
    #         {'params': model.decoder.parameters()},
    #         {'params': list_parameters[:163], 'lr': 0.5 / 1000},
    #         {'params': list_parameters[163:345], 'lr': 0.5 / 100},
    #         {'params': list_parameters[345:504], 'lr': 0.5 / 10},
    #         {'params': list_parameters[504:], 'lr': 0.5},
    #     ], 
    #     lr=0.5, betas=(0.95, 0.999)
    # )

    # print(optimizer_Task.param_groups[0]["lr"])
    # print(optimizer_Task.param_groups[1]["lr"])
    # print(optimizer_Task.param_groups[2]["lr"])
    # print(optimizer_Task.param_groups[3]["lr"])
    # print(optimizer_Task.param_groups[4]["lr"])

    # print(model.encoder.state_dict()['_blocks.38._bn2.weight'])

