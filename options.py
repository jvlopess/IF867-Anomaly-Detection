import argparse
import torch
import os
import utils  # Certifique-se de que o módulo utils esteja disponível no seu projeto

class BaseOptions():
    """
    Define os argumentos comuns para treinamento e teste, além de funções
    para imprimir e salvar as configurações.
    """
    def initialize(self, parser):
        parser.add_argument('--data_dir', type=str, default='dataset/', help='Caminho para o dataset')
        parser.add_argument('--gpu', type=str, default='0', help='Número do GPU, ex.: 0 ou 0,1,2')
        parser.add_argument('--save_dir', type=str, default='dataset/', help='Diretório para salvar os modelos')
        parser.add_argument('--model', type=str, default='cae', help='Modelo a ser utilizado')  # Alterado para "cae"
        parser.add_argument('--channels', type=int, default=3, help='Número de canais das imagens: 3 para RGB, 1 para grayscale')
        parser.add_argument('--img_size', type=int, default=256, help='Tamanho das imagens de entrada e saída')
        parser.add_argument('--latent', type=int, default=100, help='Tamanho do vetor latente')
        parser.add_argument('--init_type', type=str, default='normal', help='Tipo de inicialização [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='Fator de escala para inicialização')
        parser.add_argument('--dataset', type=str, default='mvtec', help='Método de carregamento do dataset')
        parser.add_argument('--verbose', action='store_true', help='Exibe informações adicionais para debug')
        parser.add_argument('--cropsize', type=int, default=256, help='Tamanho do crop das imagens')
        parser.add_argument('--object', type=str, default='bottle', help='Objeto para treinamento')
        return parser

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        # Se nenhum argumento for passado, os valores padrão serão usados.
        opt = parser.parse_args()

        self.print_options(opt)

        return opt

    def print_options(self, opt):
        message = '----------------------Arguments-------------------------\n'
        for k, v in sorted(vars(opt).items()):
            message += f'{k:>25}: {v:<30}\n'
        message += '---------------------End--------------------------------\n'
        print(message)

        # Salvando as opções em um arquivo
        result_dir = os.path.join(opt.save_dir, opt.model)
        utils.mkdirs(result_dir)
        opt_file_name = os.path.join(result_dir, f'{opt.mode}opt.txt')
        with open(opt_file_name, 'wt') as f:
            f.write(message)


class TrainOptions(BaseOptions):
    """
    Define os argumentos específicos para o treinamento.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_epoch_freq', type=int, default=1, help='Frequência de impressão no console')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='Frequência para salvar checkpoints')
        parser.add_argument('--epoch_count', type=int, default=0, help='Ponto de partida caso utilize um modelo pré-treinado')
        parser.add_argument('--n_epochs', type=int, default=10, help='Número total de épocas')
        parser.add_argument('--n_epochs_decay', type=int, default=50, help='Número de épocas para decaimento linear da taxa de aprendizado')
        parser.add_argument('--beta1', type=float, default=0.5, help='Momento do Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='Segundo momento do Adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='Taxa de aprendizado')
        parser.add_argument('--lr_policy', type=str, default='linear', help='Política de taxa de aprendizado')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='Decaimento da taxa de aprendizado a cada x iterações')
        parser.add_argument('--batch_size', type=int, default=8, help='Tamanho do batch')
        parser.add_argument('--mode', type=str, default='train', help='Modo de operação: train ou pretrained')
        parser.add_argument('--num_threads', default=0, type=int, help='Número de threads para carregar dados')
        parser.add_argument('--no_dropout', action='store_true', help='Desativa o dropout')
        parser.add_argument('--rotate', action='store_false', help='Ativa rotação das imagens nos transforms')
        parser.add_argument('--brightness', default=0.1, type=float, help='Ajuste de brilho nos transforms')
        return parser


class TestOptions(BaseOptions):
    """
    Define os argumentos específicos para o teste.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--mode', type=str, default='test', help='Modo de teste')
        parser.add_argument('--batch_size', type=int, default=1, help='Tamanho do batch para teste')
        parser.add_argument('--num_threads', default=0, type=int, help='Número de threads para carregar dados')
        parser.add_argument('--threshold', type=float, default=0.2, help='Threshold para comparar imagens reais e geradas')
        parser.add_argument('--rotate', action='store_false', help='Ativa rotação das imagens nos transforms')
        parser.add_argument('--brightness', default=0., type=float, help='Ajuste de brilho nos transforms')
        return parser
