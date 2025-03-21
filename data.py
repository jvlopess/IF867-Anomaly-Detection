import os
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

# ------------------------------
# 1. Funções Auxiliares para Imagens
# ------------------------------

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_imgfile(filename):
    """Verifica se o arquivo possui extensão de imagem."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_datapaths(root_dir):
    """Retorna uma lista com os caminhos de todas as imagens em root_dir."""
    image_paths = []
    assert os.path.isdir(root_dir), f"{root_dir} não existe"
    for root, _, names in os.walk(root_dir):
        for name in names:
            if is_imgfile(name):
                path = os.path.join(root, name)
                image_paths.append(path)
    return image_paths

def get_transform(opt):
    """Cria um pipeline de transformações baseado nos parâmetros de 'opt'."""
    transform_list = []
    if hasattr(opt, "rotate") and opt.rotate:
        transform_list.append(transforms.RandomRotation(0.5))
    transform_list.append(transforms.ColorJitter(brightness=getattr(opt, "brightness", 0)))
    transform_list.append(transforms.Resize((getattr(opt, "cropsize", 256), getattr(opt, "cropsize", 256)), interpolation=Image.BILINEAR))
    transform_list.append(transforms.ToTensor())
    if getattr(opt, "channels", 3) == 3:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif getattr(opt, "channels", 3) == 1:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)

# ------------------------------
# 2. Definição da Classe BaseDataset
# ------------------------------

class BaseDataset(data.Dataset):
    """Classe abstrata para datasets."""
    def __init__(self, opt):
        self.opt = opt
        self.root = getattr(opt, "data_dir", "")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

# ------------------------------
# 3. Implementação do Dataset Mvtec
# ------------------------------

class MvtecDataset(BaseDataset):
    """
    Dataset para o MVtec.
    Estrutura esperada:
      train/good/[imagens]  ou  test/[categoria]/[imagens]
    """
    def __init__(self, opt):
        super(MvtecDataset, self).__init__(opt)
        self.img_size = getattr(opt, "img_size", 256)
        self.mode = getattr(opt, "mode", "train")  # 'train' ou 'test'
        self.object = getattr(opt, "object", "")    # Categoria, ex: 'bottle'
        self.data_dir = os.path.join(self.root, self.object, self.mode)
        self.img_paths = get_datapaths(self.data_dir)
        self.dataset_size = len(self.img_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        img_path = self.img_paths[index % self.dataset_size]
        # Extrai o label a partir do nome da pasta (ex.: 'good' ou o nome da categoria)
        label = os.path.basename(os.path.dirname(img_path))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {'label': label, 'img': img, 'path': img_path}

    def __len__(self):
        return self.dataset_size

# ------------------------------
# 4. Implementação do Custom DataLoader
# ------------------------------

class CustomDatasetDataLoader():
    """
    Wrapper para carregar dados com DataLoader do PyTorch.
    """
    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=getattr(opt, "batch_size", 1),
            num_workers=int(getattr(opt, "num_threads", 1)),
            drop_last=True
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

# ------------------------------
# 5. Função Principal para Criar o Dataset
# ------------------------------

def create_dataset(opt):
    """
    Função que cria o dataset e retorna o DataLoader.
    """
    dataset = MvtecDataset(opt)
    data_loader = CustomDatasetDataLoader(opt, dataset).load_data()
    print(f"Dataset [{type(dataset).__name__}] criado com {len(dataset)} imagens.")
    return data_loader
