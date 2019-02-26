import torch as t
import torch.nn as nn
from torch.nn import init
import functools

# helper functions


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer, adopted from CycleGan

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    batch norm 记录平均值和标准差

    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    instance norm 不记录数据
    """
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d,
                                       affine=False,
                                       track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            f'normalization layer [{norm_type}] not found')
    return norm_layer


def init_weights(net, ini_type='normal', init_gain=0.02):
    """
    Initialize network weights, adopted from CycleGan
    :net (network): the network to be initialized
    :ini_type (str): name of the initialization method: normal/xavier/kaiming/orthogonal
    :init_gain (float): scaling for normal, xavier, and orthogonal
    :return: None, apply initialization to the network
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if ini_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)
        elif ini_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif ini_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif ini_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(
                f'initialization method [{ini_type}] is not implemented')

    print(f'initialize network with {ini_type}')
    net.apply(init_func)  # apply the initialization function <init_func>


class ResnetGenerator(nn.Module):
    def __init__(self, opt):
        # input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """ Construct a Resnet-based generator, adopt from CycleGan """

        input_nc = opt.input_nc
        output_nc = opt.output_nc
        n_blocks = opt.n_blocks
        ngf = opt.ngf
        use_dropout = opt.use_dropout

        assert (n_blocks > 0)  # resnet block 数目需要大于一

        super(ResnetGenerator, self).__init__()

        model = [nn.ReflectionPad2d(padding=3),
                 nn.Conv2d(in_channels=input_nc,
                           out_channels=ngf,
                           kernel_size=7,
                           padding=0,
                           bias=True),
                 nn.InstanceNorm2d(num_features=ngf),
                 nn.ReLU(inplace=True)]
        # [1, 3, 262, 262]  ->  [1, 64, 256, 256]

        """ add 2 downsampling layers """
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(in_channels=ngf * mult,
                                out_channels=ngf * mult * 2,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=True),
                      nn.InstanceNorm2d(num_features=ngf * mult * 2),
                      nn.ReLU(inplace=True)]
        # [1, 64, 256, 256]  ->  [1, 64 * 2, 128, 128]
        # [1, 64 * 2, 256, 256]  ->  [1, 64 * 4, 128, 128]

        """ add n ResNet blocks """
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(dim=ngf * mult, use_dropout=use_dropout)]

        # TODO: ask Ekta, why is there no dimension change between blocks here?

        """ add 2 upsampling layers """
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(in_channels=ngf * mult,
                                         out_channels=int(ngf * mult / 2),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(num_features=int(ngf * mult / 2)),
                      nn.ReLU(inplace=True)]
            # 64 * 4  -> 64 * 2
            # 128x128 -> 256x256
            # 64 * 2  -> 64
            # 256x256 -> 512x512

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        # 64      -> 3
        # 512x512 -> 512x512

        # 将 list model 转化为 nn.Sequential
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """ standard forward """
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)

    def build_conv_block(self, dim, use_dropout):

        conv_block = [nn.ReflectionPad2d(padding=1)]
        conv_block += [nn.Conv2d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=3,
                                 padding=0,
                                 bias=True),
                       nn.InstanceNorm2d(num_features=dim),
                       nn.ReLU(inplace=True)]

        if use_dropout:
            conv_block += [nn.Dropout(p=0.5)]

        conv_block += [nn.ReflectionPad2d(padding=1)]
        conv_block += [nn.Conv2d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=3,
                                 padding=0,
                                 bias=True),
                       nn.InstanceNorm2d(num_features=dim)]

        return nn.Sequential(*conv_block)

    def forward(self, input):
        """ add skip connections """
        out = input + self.conv_block(input)
        return out


class NLayerDiscriminator(nn.Module):
    """ defines a PatchGAN discriminator, adopt from CycleGAN """

    def __init__(self, opt):
        super(NLayerDiscriminator, self).__init__()
        use_bias = True

        input_nc = opt.input_nc
        ndf = opt.ndf
        n_layers = 3

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels=input_nc,
                              out_channels=ndf,
                              kernel_size=kw,
                              stride=2,
                              padding=padw),
                    nn.LeakyReLU(negative_slope=0.2,
                                 inplace=True)]
        # 3       -> 64
        # 512x512 -> 256x256

        nf_mult = 1
        for n in range(1, n_layers):
            # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(in_channels=ndf * nf_mult_prev,
                                   out_channels=ndf * nf_mult,
                                   kernel_size=kw,
                                   stride=2,
                                   padding=padw,
                                   bias=True),
                         nn.InstanceNorm2d(num_features=ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]
            # 64      -> 64 * 2
            # 256x256 -> 128x128
            # 64 * 2  -> 64 * 4
            # 128x128 -> 64x64

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(in_channels=ndf * nf_mult_prev,
                               out_channels=ndf * nf_mult,
                               kernel_size=kw,
                               stride=1,
                               padding=padw,
                               bias=True),
                     nn.InstanceNorm2d(num_features=ndf * nf_mult),
                     nn.LeakyReLU(negative_slope=0.2,
                                  inplace=True)]
        # 64 * 4 -> 64 * 8
        # 64x64  -> 63x63

        """ output 1 channel prediction map """
        sequence += [nn.Conv2d(in_channels=ndf * nf_mult,
                               out_channels=1,
                               kernel_size=kw,
                               stride=1,
                               padding=padw)]
        # 64 * 8 -> 1
        # 63x63  -> 62x62

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """ standard forward """
        return self.model(input)


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
        self.register_buffer('real_label', t.tensor(target_real_label))
        self.register_buffer('fake_label', t.tensor(target_fake_label))
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
        # return target_tensor.expand_as(prediction)
        return target_tensor.expand_as(prediction).to(t.device('cuda'))

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # prediction is 2D prediction map vector from Discriminator
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # loss = self.loss(prediction, target_tensor)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
