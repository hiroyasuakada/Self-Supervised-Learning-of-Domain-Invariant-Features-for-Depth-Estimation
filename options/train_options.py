from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # training epoch
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count')
        self.parser.add_argument('--niter', type=int, default=10,
                                 help='# of iter with initial learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10,
                                 help='# of iter to decay learning rate to zero')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--transform_epoch', type=int, default=0,
                                 help='# of epoch for transform learning')
        self.parser.add_argument('--task_epoch', type=int, default=0,
                                 help='# of epoch for task learning')


        # learning rate and loss weight
        self.parser.add_argument('--optimizer_type', type=str, default='Adam',
                                 help='optimizer type[Adam|SGD]')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy[lambda|step|plateau]')
        self.parser.add_argument('--lr_task', type=float, default=1e-4,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--lr_trans', type=float, default=5e-5,
                                 help='initial learning rate for discriminator')

        self.parser.add_argument('--lambda_gan_img', type=float, default=1.0,
                                 help='weight for image GAN loss')
        self.parser.add_argument('--lambda_gan_feature', type=float, default=0.1,
                                 help='weight for feature GAN loss')
        self.parser.add_argument('--lambda_cycle', type=float, default=10.0,
                                 help='weight for cycle consistency loss')
        self.parser.add_argument('--lambda_identity', type=float, default=100.0,
                                 help='weight for identity loss')

        self.parser.add_argument('--lambda_task', type=float, default=100.0,
                                 help='weight for task loss')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.1,
                                 help='weight for depth smooth loss')
        self.parser.add_argument('--lambda_crdoco', type=float, default=1.0,
                                 help='weight for crdoco loss')

        self.parser.add_argument('--lambda_simsiam', type=float, default=10.0,
                                 help='weight for simsiam loss')
        self.parser.add_argument('--lambda_local_contrastive', type=float, default=0.0,
                                 help='weight for local contrastive loss')


        # display the results
        self.parser.add_argument('--display_freq', type=int, default=40,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=40,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=80,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--val_freq', type=int, default=80,
                                 help='frequency of validation')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results')
        # others
        self.parser.add_argument('--separate', action='store_true',
                                 help='transform and task network training end-to-end or separate')
        self.parser.add_argument('--pool_size', type=int, default=20,
                                 help='the size of image buffer that stores previously generated images')

        ### training parameters for Gaussian Process
        self.parser.add_argument('--garg_crop', help='Use garg crop for validation', action='store_true')
        self.parser.add_argument('--eigen_crop', help='Use eigen crop for validation', action='store_true')
        self.parser.add_argument('--no_online_eval', help='do online evaluation ?', action='store_true')


        self.isTrain = True
