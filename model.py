import os
import functools
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim import lr_scheduler

# ---------------------------------------
# 1. Funções e Utilitários de Inicialização
# ---------------------------------------

import importlib

def load_model(model_name):
    """Importa dinamicamente o modelo baseado no nome"""
    import model  # Importa ele mesmo, já que está tudo em model.py
    model_module = model
    model_class = getattr(model_module, model_name.upper())   
    if model_class is None:
        print(f"Não há modelo com nome {model_name}")
        exit(0)
    return model_class

def create_model(opt):
    """Cria o modelo baseado no nome passado em opt.model"""
    model_class = load_model(opt.model)
    model = model_class(opt)
    print(f"Modelo [{type(model).__name__}] foi criado com sucesso")
    return model

def init_weights(net, init_type='normal', init_gain=0.02):
    """Inicializa os pesos da rede de acordo com o método especificado."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Inicialização [{init_type}] não implementada')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print(f'Inicializando a rede com {init_type}')
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu=[], mode='train'):
    """Configura a rede, move para GPU se disponível e aplica a inicialização de pesos se em modo treino."""
    if len(gpu) > 0 and torch.cuda.is_available():
        net.to(gpu[0])
        net = nn.DataParallel(net, gpu)
    else:
        net.to('cpu')
    if mode == 'train':
        init_weights(net, init_type, init_gain=init_gain)
    return net

def get_scheduler(optimizer, opt):
    """Retorna o scheduler de aprendizado de acordo com a política definida."""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(f'Política de LR [{opt.lr_policy}] não implementada')
    return scheduler

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Retorna uma camada de normalização de acordo com o tipo especificado."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError(f'Layer de normalização [{norm_type}] não encontrada')
    return norm_layer

# ---------------------------------------
# 2. Definição das Redes: Encoder e Decoder (para o CAE)
# ---------------------------------------

class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.model(input)
        return img.view(img.shape[0], *self.img_shape)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=0)
        )

        self.last_layer = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(img.shape[0], -1)
        features = self.last_layer(features)
        features = features.view(features.shape[0], -1, 1, 1)
        return features

# ---------------------------------------
# 3. Base do Modelo e Implementação do CAE
# ---------------------------------------

class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.gpu = opt.gpu
        if len(self.gpu) > 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu[0]}')
        else:
            self.device = torch.device('cpu')
        self.optimizers = []
        self.networks = []
        self.save_dir = os.path.join(opt.save_dir, opt.object)
        self.isTrain = (opt.mode.lower() == 'train')
    
    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def setup(self, opt):
        if opt.mode == 'train':
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        elif opt.mode == 'test':
            self.load_networks()
        self.print_networks(opt.verbose)

    def set_requires_grad(self, *nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_generated_imgs(self):
        visual_imgs = None
        for name in self.visual_names:
            if isinstance(name, str):
                visual_imgs = getattr(self, name)
        return visual_imgs

    def eval(self):
        for name in self.networks:
            net = getattr(self, name)
            net.eval()

    def update_learning_rate(self, epoch):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'{epoch} : learning rate {old_lr:.7f} -> {lr:.7f}')

    def print_networks(self, verbose):
        print('---------- Redes Inicializadas -------------')
        for name in self.networks:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = sum(param.numel() for param in net.parameters())
                if verbose:
                    print(net)
                print(f'[Rede {name}] Número total de parâmetros: {num_params / 1e6:.3f} M')
        print('-----------------------------------------------')

    def save_networks(self):
        os.makedirs(self.save_dir, exist_ok=True)
        save_encoder_path = os.path.join(self.save_dir, f'{self.model_name}_e.pth')
        save_decoder_path = os.path.join(self.save_dir, f'{self.model_name}_d.pth')
        net_d = getattr(self, 'decoder')
        net_e = getattr(self, 'encoder')

        if len(self.gpu) > 0 and torch.cuda.is_available():
            torch.save(net_d.module.cpu().state_dict(), save_decoder_path)
            net_d.cuda(self.gpu[0])
            torch.save(net_e.module.cpu().state_dict(), save_encoder_path)
            net_e.cuda(self.gpu[0])
        else:
            torch.save(net_d.cpu().state_dict(), save_decoder_path)
            torch.save(net_e.cpu().state_dict(), save_encoder_path)

    def load_networks(self):
        load_encoder_path = os.path.join(self.save_dir, f'{self.model_name}_e.pth')
        load_decoder_path = os.path.join(self.save_dir, f'{self.model_name}_d.pth')
        net_e = getattr(self, 'encoder')
        net_d = getattr(self, 'decoder')
        if isinstance(net_d, nn.DataParallel):
            net_d = net_d.module
        if isinstance(net_e, nn.DataParallel):
            net_e = net_e.module
        print('Carregando encoder de', load_encoder_path)
        print('Carregando decoder de', load_decoder_path)
        encoder_state = torch.load(load_encoder_path)
        decoder_state = torch.load(load_decoder_path)
        net_e.load_state_dict(encoder_state)
        net_d.load_state_dict(decoder_state)

    def get_current_losses(self, *loss_names):
        losses = {}
        for name in loss_names:
            losses[name] = float(getattr(self, name))
        return losses

class CAE(BaseModel):
    """
    Implementação do Convolutional AutoEncoder (CAE).
    O CAE utiliza um encoder e um decoder para reconstruir imagens,
    permitindo medir o erro de reconstrução como métrica de anomalia.
    """
    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        img_shape = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent
        self.encoder = init_net(Encoder(latent), gpu=opt.gpu, mode=opt.mode)
        self.decoder = init_net(Decoder(img_shape, latent), gpu=opt.gpu, mode=opt.mode)
        self.networks = ['encoder', 'decoder']
        self.criterion = nn.MSELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['loss']

        if self.opt.mode == 'train':
            self.optimizer_e = optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                          betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                          betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.set_requires_grad(self.decoder, self.encoder, requires_grad=True)

    def set_input(self, input):
        self.real_imgs = input['img'].to(self.device)

    def forward(self):
        features = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(features)

    def backward(self):
        self.loss = self.criterion(10 * self.real_imgs, 10 * self.generated_imgs)
        self.loss.backward()

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

    def save_images(self, data):
        # Supondo que 'utils.compare_images' e 'utils.save_images' estejam implementados externamente.
        import utils
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object, "result")
        anomaly_img = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img, paths, data)
