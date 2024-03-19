import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from skimage.measure import compare_psnr
import warnings
warnings.filterwarnings('ignore')
from util.util import calc_psnr
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.restoration import estimate_sigma   
from torch.nn.functional import l1_loss
class L1_TVLoss_Charbonnier(nn.Module):
    def __init__(self):
        super(L1_TVLoss_Charbonnier, self).__init__()
        self.e = 0.000001 ** 2
    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))
        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))
        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))
        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))
        return h_tv + w_tv


class GammablindModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['MSE','f','sigma','zeta_s']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['lr','hr','recon']
        if self.isTrain:
            self.model_names = ['f']
        else:  # during test time, only load G
            self.model_names = ['f']
            self.visual_names = ['lr','hr','recon']
        # define networks (both generator and discriminator)
        self.netf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionKL = torch.nn.KLDivLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f)
                        
        self.TVL1loss = L1_TVLoss_Charbonnier()
        self.batch = opt.batch_size
        self.zeta = opt.parameter
        self.sigma_min = 0.001
        self.sigma_max = 0.05
        self.sigma_annealing = 2*11000
     
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.hr = input['B' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  
        self.lr = (self.hr.cpu().numpy())*np.random.gamma(self.zeta_s,1/self.zeta_s,self.hr.shape) 
        self.lr = torch.from_numpy(self.lr).to(self.device,dtype = torch.float32)
        
    def set_input_val(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.lr = input['A' if AtoB else 'B'].to(self.device,dtype = torch.float32)
        self.hr = input['B' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']   
        self.k = self.lr.shape[2]*self.lr.shape[3]*self.lr.shape[0]
        
    def set_param_s(self, iter):       
        self.min_log = np.log([120])
        self.zeta_now = 40
        self.zeta_s = self.min_log + np.random.rand(1) * (np.log([self.zeta_now]) - self.min_log)
        self.zeta_s = np.exp(self.zeta_s)                
        self.zeta_s = self.zeta_s[0]
        self.loss_zeta_s = self.zeta_s
        
    def set_sigma(self, iter):
        perc = min((iter+1)/float(self.sigma_annealing), 1.0)
        self.sigma = self.sigma_max * (1-perc) + self.sigma_min * perc
        self.loss_sigma = self.sigma
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.denom = (self.zeta_s*self.lr)
        self.nom = (self.zeta_s-1) - self.lr*self.netf(self.lr,0)[0]
        self.recon = self.denom/self.nom 
    def forward_test(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.denom = (self.zeta*self.lr)
        self.nom = (self.zeta-1) - self.lr*self.netf(self.lr,0)[0]
        self.recon = self.denom/self.nom         
    def forward_tv(self,i):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        sigma_max = 120
        sigma_min = 40
        best_loss = 100
        best_sigma = 0
        tvlosses = []
        best_sigmas = 0
        for iter in range(80):
            perc = min((iter+1)/float(80), 1.0)
            sigma = sigma_max* (perc) + sigma_min *(1-perc)
            zeta = sigma
            denom = (zeta*self.lr)
            nom = (zeta-1) - self.lr*self.netf(self.lr,0)[0]
            recon = denom/nom      
            idx = [recon>0]
            tvloss = self.TVL1loss(recon) + 1*(torch.mean(torch.log(recon[idx])) + torch.mean(self.lr[idx]/recon[idx]) + torch.mean((self.lr[idx]/recon[idx])**2))
            if tvloss < best_loss:
                best_loss = tvloss
                best_sigma = zeta
                self.recon = recon
            print('Current zeta %f TVL1loss: %.4f Best_TVloss(Best zeta %f) : %.4f' % (zeta,tvloss,best_sigma,best_loss))
        
    def forward_psnr(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.denom = (self.zeta*self.lr)
        self.nom = (self.zeta -1) - self.lr*self.netf(self.lr,0)[0]
        self.recon = self.denom/self.nom
        self.hr = self.hr.detach().cpu()
        psnr = calc_psnr(self.recon.detach().cpu(),self.hr)             
        _,loss = self.netf(self.lr,self.sigma_min)        
        return  psnr,loss 
    
    def backward_f(self):
        """Calculate GAN and L1 loss for the generator"""            
        self.output,self.loss_f = self.netf(self.lr,self.sigma)     
        self.loss_MSE = self.criterionL2(self.recon,self.hr)
        self.loss_f.backward()
        
    def optimize_parameters(self):
        self.forward()                  # compute fake images: G(A)     
        self.optimizer_f.zero_grad()        # set G's gradients to zero
        self.backward_f()                   # calculate graidents for G
        self.optimizer_f.step()              # udpate G's weights              