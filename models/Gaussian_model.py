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

class GaussianModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['MSE','f','sigma']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['lr','hr','output_f','recon']
        else:
            self.visual_names = ['lr','hr','recon']
        if self.isTrain:
            self.model_names = ['f']
        else:  # during test time, only load G
            self.model_names = ['f']
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
            
        self.variance = (opt.parameter/255)**2
        self.batch = opt.batch_size
        self.sigma_min = 0.001
        self.sigma_max = 0.05
        self.sigma_annealing = 2*11000
        
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.lr = input['A' if AtoB else 'B'].to(self.device,dtype = torch.float32)
        self.hr = input['B' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.n = np.random.normal(scale = 1.0, size = self.hr.shape)
        self.n = torch.from_numpy(self.n).to(self.device,dtype = torch.float32)      
        self.k = self.lr.shape[2]*self.lr.shape[3]*self.lr.shape[0]
        
    def set_sigma(self, iter):
        perc = min((iter+1)/float(self.sigma_annealing), 1.0)
        self.sigma = self.sigma_max * (1-perc) + self.sigma_min * perc
        self.loss_sigma = self.sigma
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.output_f = (self.variance)*self.netf(self.lr,0)[0]
        self.recon = (self.variance)*self.netf(self.lr,0)[0]+self.lr
  
    def forward_psnr(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recon = (self.variance)*self.netf(self.lr,0)[0]+self.lr
        self.recon = torch.clamp(self.recon.detach().cpu(), 0, 1)
        self.hr = self.hr.detach().cpu()
        psnr = calc_psnr(self.recon,self.hr)        
        return  psnr 
        
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