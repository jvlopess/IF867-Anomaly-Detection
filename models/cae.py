from .base_model import BaseModel
from . import networks
import torch
from utils import utils
from models import init_net
import os
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from skimage.metrics import peak_signal_noise_ratio
class CAE(BaseModel):
    """This class implements the Convolutional AutoEncoder for normal image generation
    CAE is processed in encoder and decoder that is composed CNN layers
    """

    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        """Initialize the CAE model"""
        BaseModel.__init__(self, opt)
        self.opt = opt
        img_size = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent
        self.encoder = init_net(networks.Encoder(latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize encoder networks doing data parallel and init_weights
        self.decoder = init_net(networks.Decoder(img_size, latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize decoder networks doing data parallel and init_weights
        self.networks = ['encoder', 'decoder']
        self.criterion = torch.nn.MSELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['loss']

        if self.opt.mode == 'train':# if mode is train, we have to set optimizer and requires grad is true
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.set_requires_grad(self.decoder, self.encoder, requires_grad=True)

    def forward(self):
        features = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(features)

    def backward(self):
        self.loss = self.criterion(10*self.real_imgs, 10*self.generated_imgs)
        self.loss.backward()

    def set_input(self, input):
        self.real_imgs = input['img'].to(self.device)

    def train(self):
        self.forward()
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        self.backward()
        self.optimizer_d.step()
        self.optimizer_e.step()

    def test(self):
        with torch.no_grad():
            self.forward()
            mse, psnr= self.calculate_metrics()    
            print(f"Test Metrics - MSE: {mse:.4f}, PSNR: {psnr:.2f} dB") 
        return mse,psnr

    def save_images(self, data):
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object)
        paths = os.path.join(paths, "result")
        anomaly_img_compared = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img_compared, paths, data)

    def calculate_metrics(self):
        """Calcula MSE, PSNR """

        real_imgs = self.real_imgs.cpu().numpy()
        generated_imgs = self.generated_imgs.cpu().numpy()

        real_imgs = real_imgs.transpose(0, 2, 3, 1)  # (batch, H, W, C)
        generated_imgs = generated_imgs.transpose(0, 2, 3, 1)

        mse_list = []
        psnr_list = []
        y_scores = []
        y_true = []

        for real, gen in zip(real_imgs, generated_imgs):
            mse = mean_squared_error(real.flatten(), gen.flatten())
            psnr = peak_signal_noise_ratio(real, gen, data_range=real.max() - real.min())

            mse_list.append(mse)
            psnr_list.append(psnr)
            y_scores.append(mse)

        return np.mean(mse_list), np.mean(psnr_list)



