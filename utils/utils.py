"""This module has some useful functions"""

import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    elif not os.path.exists(paths):
        os.makedirs(paths)

def save_images(images, image_paths, data):
    images = Image.fromarray(images)
    label = data["label"][0]
    file_name = data["path"][0].split("/")[-1]
    if not os.path.exists(image_paths):
        os.mkdir(image_paths)
    image_paths = os.path.join(image_paths, label + file_name)
    images.save(image_paths)

def convert2img(image, imtype=np.uint8):
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
        assert len(image.squeeze().shape) < 4
    if image.dtype != np.uint8:
        image = (np.transpose(image.squeeze(), (1, 2, 0)) * 0.5 + 0.5) * 255
    return image.astype(imtype)

def plt_show(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img = img.numpy()
    if img.dtype != "uint8":
        img_numpy = img * 0.5 + 0.5
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()

def plot_losses(losses_list, num_epochs):
    # Extrair valores das perdas
    all_losses = {key: [] for key in losses_list[0].keys()}  # Criar listas para cada tipo de loss

    # Preencher as listas de perdas para cada chave (losses)
    for losses in losses_list:
        for key, value in losses.items():
            all_losses[key].append(value)

    # Plotar as perdas para cada chave
    plt.figure(figsize=(8, 5))

    for key, values in all_losses.items():
        plt.plot(range(1, num_epochs + 1), values, marker='o', linestyle='-', label=key)  # Plot para cada loss

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Evolution of Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_images(real_img, generated_img, threshold=0.4):
    generated_img = generated_img.type_as(real_img)
    diff_img = np.abs(generated_img - real_img)
    real_img = convert2img(real_img)
    generated_img = convert2img(generated_img)
    diff_img = convert2img(diff_img)

    threshold = (threshold*0.5+0.5)*255
    diff_img[diff_img <= threshold] = 0

    anomaly_img = np.zeros_like(real_img)
    anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img>0)[0], np.where(diff_img>0)[1]] = [200, 0, 0]
    
    anomaly_img=convert2img(anomaly_img)
    # Criar uma única imagem combinando as 4 imagens
    img_height, img_width, _ = real_img.shape
    combined_width = img_width * 4  # Como são 4 imagens lado a lado
    combined_img = Image.new('RGB', (combined_width, img_height))

    # Converter imagens para PIL
    real_pil = Image.fromarray(real_img)
    gen_pil = Image.fromarray(generated_img)
    diff_pil = Image.fromarray(diff_img)
    anomaly_pil = Image.fromarray(anomaly_img)

    # Adicionar imagens na imagem combinada
    combined_img.paste(real_pil, (0, 0))
    combined_img.paste(gen_pil, (img_width, 0))
    combined_img.paste(diff_pil, (img_width * 2, 0))
    combined_img.paste(anomaly_pil, (img_width * 3, 0))

    return np.array(combined_img)  # Retorna a imagem combinada como um array numpy
