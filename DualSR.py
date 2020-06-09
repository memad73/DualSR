import torch
import loss
import networks
import util
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.io import loadmat
import os
from torch_sobel import Sobel

class DualSR:
  
    def __init__(self, conf):
        # Fix random seed
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True # slightly reduces throughput
    
        # Acquire configuration
        self.conf = conf
        
        # Define the networks
        self.G_DN = networks.Generator_DN().cuda()
        self.D_DN = networks.Discriminator_DN().cuda()
        self.G_UP = networks.Generator_UP().cuda()

        # Losses
        self.criterion_gan = loss.GANLoss().cuda()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_interp = torch.nn.L1Loss()
        self.regularization = loss.DownsamplerRegularization(conf.scale_factor_downsampler, self.G_DN.G_kernel_size)

        # Initialize networks weights
        self.G_DN.apply(networks.weights_init_G_DN)
        self.D_DN.apply(networks.weights_init_D_DN)
        self.G_UP.apply(networks.weights_init_G_UP)
        
        # Optimizers
        self.optimizer_G_DN = torch.optim.Adam(self.G_DN.parameters(), lr=conf.lr_G_DN, betas=(conf.beta1, 0.999))
        self.optimizer_D_DN = torch.optim.Adam(self.D_DN.parameters(), lr=conf.lr_D_DN, betas=(conf.beta1, 0.999))
        self.optimizer_G_UP = torch.optim.Adam(self.G_UP.parameters(), lr=conf.lr_G_UP, betas=(conf.beta1, 0.999))
        
        # Read input image
        self.in_img = util.read_image(conf.input_image_path)
        self.in_img_t= util.im2tensor(self.in_img)
        b_x = self.in_img_t.shape[2] % conf.scale_factor
        b_y = self.in_img_t.shape[3] % conf.scale_factor
        self.in_img_cropped_t = self.in_img_t[..., b_x:, b_y:]
        
        self.gt_img = util.read_image(conf.gt_path) if conf.gt_path is not None else None
        self.gt_kernel = loadmat(conf.kernel_path)['Kernel'] if conf.kernel_path is not None else None
        
        if self.gt_kernel is not None:
            self.gt_kernel = np.pad(self.gt_kernel, 1, 'constant')
            self.gt_kernel = util.kernel_shift(self.gt_kernel, sf=conf.scale_factor)
            self.gt_kernel_t = torch.FloatTensor(self.gt_kernel).cuda()
        
            self.gt_downsampled_img_t = util.downscale_with_kernel(self.in_img_cropped_t, self.gt_kernel_t)
            self.gt_downsampled_img = util.tensor2im(self.gt_downsampled_img_t)
        
        # Debug variables
        self.debug_steps = []
        self.UP_psnrs = [] if self.gt_img is not None else None
        self.DN_psnrs = [] if self.gt_kernel is not None else None
        
        if self.conf.debug:
            self.loss_GANs = []
            self.loss_cycle_forwards = []
            self.loss_cycle_backwards = []
            self.loss_interps = []
            self.loss_Discriminators = []
        
        self.iter = 0
        
    def train(self, data):
        self.set_input(data)
        self.train_G()
        self.train_D()
        if self.conf.debug:
            if self.iter % self.conf.eval_iters == 0:
                self.quick_eval()
            if self.iter % self.conf.plot_iters == 0:
                self.plot()
        self.iter = self.iter + 1
    
    
    def set_input(self, data):
        self.real_HR = data['HR']
        self.real_LR = data['LR']
        self.real_LR_bicubic = data['LR_bicubic']
    
    
    def train_G(self):
        # Turn off gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], False)
        
        # Rese gradient valus
        self.optimizer_G_UP.zero_grad()
        self.optimizer_G_DN.zero_grad()
        
        # Forward path
        self.fake_HR = self.G_UP(self.real_LR)
        self.rec_LR = self.G_DN(self.fake_HR)
        # Backward path
        self.fake_LR = self.G_DN(self.real_HR)
        self.rec_HR = self.G_UP(self.fake_LR)
        
        # Losses
        self.loss_GAN = self.criterion_gan(self.D_DN(self.fake_LR), True)
        self.loss_cycle_forward = self.criterion_cycle(self.rec_LR, util.shave_a2b(self.real_LR, self.rec_LR)) * self.conf.lambda_cycle
        self.loss_cycle_backward = self.criterion_cycle(self.rec_HR, util.shave_a2b(self.real_HR, self.rec_HR)) * self.conf.lambda_cycle
        
        sobel_A = Sobel()(self.real_LR_bicubic.detach())
        loss_map_A = 1 - torch.clamp(sobel_A, 0, 1)
        self.loss_interp = self.criterion_interp(self.fake_HR * loss_map_A, self.real_LR_bicubic * loss_map_A) * self.conf.lambda_interp
        
        self.curr_k = util.calc_curr_k(self.G_DN.parameters())
        self.loss_regularization = self.regularization(self.curr_k, self.real_HR, self.fake_LR) * self.conf.lambda_regularization
              
        self.total_loss = self.loss_GAN + self.loss_cycle_forward + self.loss_cycle_backward + self.loss_interp + self.loss_regularization
        self.total_loss.backward()
        
        self.optimizer_G_UP.step()
        self.optimizer_G_DN.step()
        
        
    def train_D(self):
        # Turn on gradient calculation for discriminator
        util.set_requires_grad([self.D_DN], True)
        
        # Rese gradient valus
        self.optimizer_D_DN.zero_grad()
        
        # Fake
        pred_fake = self.D_DN(self.fake_LR.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Real
        pred_real = self.D_DN(util.shave_a2b(self.real_LR, self.fake_LR))
        loss_D_real = self.criterion_gan(pred_real, True)
        # Combined loss and calculate gradients
        self.loss_Discriminator = (loss_D_real + loss_D_fake) * 0.5
        self.loss_Discriminator.backward()

        # Update weights
        self.optimizer_D_DN.step()
        
               
    def eval(self):
        self.quick_eval()
        if self.conf.debug:
            self.plot()
            
        util.save_final_kernel(util.move2cpu(self.curr_k), self.conf)
        plt.imsave(os.path.join(self.conf.output_dir, '%s.png' % self.conf.abs_img_name), self.upsampled_img)
        
        if self.gt_img is not None:
            print('Upsampler PSNR = ', self.UP_psnrs[-1])
        if self.gt_kernel is not None:
            print("Downsampler PSNR = ", self.DN_psnrs[-1])
        print('*' * 60 + '\nOutput is saved in \'%s\' folder\n' % self.conf.output_dir)
        plt.close('all')
    
    def quick_eval(self):
        # Evaluate trained upsampler and downsampler on input data
        with torch.no_grad():
            downsampled_img_t = self.G_DN(self.in_img_cropped_t)
            upsampled_img_t = self.G_UP(self.in_img_t)
        
        self.downsampled_img = util.tensor2im(downsampled_img_t)
        self.upsampled_img = util.tensor2im(upsampled_img_t)
        
        if self.gt_kernel is not None:
            self.DN_psnrs += [util.cal_y_psnr(self.downsampled_img, self.gt_downsampled_img, border=self.conf.scale_factor)]
        if self.gt_img is not None:
            self.UP_psnrs += [util.cal_y_psnr(self.upsampled_img, self.gt_img, border=self.conf.scale_factor)]
        self.debug_steps += [self.iter]

        if self.conf.debug:
            # Save loss values for visualization
            self.loss_GANs += [util.move2cpu(self.loss_GAN)]
            self.loss_cycle_forwards += [util.move2cpu(self.loss_cycle_forward)]
            self.loss_cycle_backwards += [util.move2cpu(self.loss_cycle_backward)]
            self.loss_interps += [util.move2cpu(self.loss_interp)]
            self.loss_Discriminators += [util.move2cpu(self.loss_Discriminator)]
        
            
    def plot(self):
        loss_names = ['loss_GANs', 'loss_cycle_forwards', 'loss_cycle_backwards', 'loss_interps', 'loss_Discriminators']
        
        if self.gt_img is not None:
            plots_data, labels = zip(*[(np.array(x), l) for (x, l)
                               in zip([self.UP_psnrs, self.DN_psnrs],
                                      ['Upsampler PSNR', 'Downsampler PSNR']) if x is not None])
        else:
            plots_data, labels = [0.0], 'None'
        
        plots_data2, labels2 = zip(*[(np.array(x), l) for (x, l)
                                   in zip([getattr(self, name) for name in loss_names],
                                          loss_names) if x is not None])
        # For the first iteration create the figure
        if not self.iter:
            # Create figure and split it using GridSpec. Name each region as needed
            self.fig = plt.figure(figsize=(9, 8))
            #self.fig = plt.figure()
            grid = GridSpec(4, 4)
            self.psnr_plot_space = plt.subplot(grid[0:2, 0:2])
            self.loss_plot_space = plt.subplot(grid[0:2, 2:4])
            
            self.real_LR_space = plt.subplot(grid[2, 0])
            self.fake_HR_space = plt.subplot(grid[2, 1])
            self.rec_LR_space = plt.subplot(grid[2, 2])
            self.real_HR_space = plt.subplot(grid[3, 0])
            self.fake_LR_space = plt.subplot(grid[3, 1])
            self.rec_HR_space = plt.subplot(grid[3, 2])
            self.curr_ker_space = plt.subplot(grid[2, 3])
            self.ideal_ker_space = plt.subplot(grid[3, 3])

            # Activate interactive mode for live plot updating
            plt.ion()

            # Set some parameters for the plots
            self.psnr_plot_space.set_ylabel('db')
            self.psnr_plot_space.grid(True)
            self.psnr_plot_space.legend(labels)
            
            self.loss_plot_space.grid(True)
            self.loss_plot_space.legend(labels2)
            
            self.curr_ker_space.title.set_text('estimated kernel')
            self.ideal_ker_space.title.set_text('gt kernel')
            self.real_LR_space.title.set_text('$x$')
            self.real_HR_space.title.set_text('$y$')
            self.fake_HR_space.title.set_text('$G_{UP}(x)$')
            self.fake_LR_space.title.set_text('$G_{DN}(y)$')
            self.rec_LR_space.title.set_text('$G_{DN}(G_{UP}(x))$')
            self.rec_HR_space.title.set_text('$G_{UP}(G_{DN}(y))$')
            
            # loop over all needed plot types. if some data is none than skip, if some data is one value tile it
            self.plots = self.psnr_plot_space.plot(*[[0]] * 2 * len(plots_data))
            self.plots2 = self.loss_plot_space.plot(*[[0]] * 2 * len(plots_data2))
            
            # These line are needed in order to see the graphics at real time
            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.01)
            return

        # Update plots
        for plot, plot_data in zip(self.plots, plots_data):
            plot.set_data(self.debug_steps, plot_data)
            
        for plot, plot_data in zip(self.plots2, plots_data2):
            plot.set_data(self.debug_steps, plot_data)
        
        self.psnr_plot_space.set_xlim([0, self.iter + 1])
        all_losses = np.array(plots_data)
        self.psnr_plot_space.set_ylim([np.min(all_losses)*0.9, np.max(all_losses)*1.1])
        
        self.loss_plot_space.set_xlim([0, self.iter + 1])
        all_losses2 = np.array(plots_data2)
        self.loss_plot_space.set_ylim([np.min(all_losses2)*0.9, np.max(all_losses2)*1.1])

        self.psnr_plot_space.legend(labels)
        self.loss_plot_space.legend(labels2)

        # Show current images
        self.curr_ker_space.imshow(util.move2cpu(self.curr_k))
        if self.gt_kernel is not None:
            self.ideal_ker_space.imshow(self.gt_kernel)
        self.real_LR_space.imshow(util.tensor2im(self.real_LR))
        self.real_HR_space.imshow(util.tensor2im(self.real_HR))
        self.fake_HR_space.imshow(util.tensor2im(self.fake_HR))
        self.fake_LR_space.imshow(util.tensor2im(self.fake_LR))
        self.rec_LR_space.imshow(util.tensor2im(self.rec_LR))
        self.rec_HR_space.imshow(util.tensor2im(self.rec_HR))
        
        self.curr_ker_space.axis('off')
        self.ideal_ker_space.axis('off')
        self.real_LR_space.axis('off')
        self.real_HR_space.axis('off')
        self.fake_HR_space.axis('off')
        self.fake_LR_space.axis('off')
        self.rec_LR_space.axis('off')
        self.rec_HR_space.axis('off')
        
            
        # These line are needed in order to see the graphics at real time
        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.01)
        
        