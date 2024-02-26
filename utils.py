import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import yaml
from torchvision.transforms import ToTensor


def train(model, data_yaml, criterion, optimizer, epochs):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])

    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)

    img_path = 'dataset/train/images/'
    label_path = 'dataset/train/labels/'

    img_files = os.listdir(img_path)
    num_img_files = len(img_files)

    label_files = os.listdir(label_path)
    num_label_files = len(label_files)

    print(f'Количество файлов в папке с метками: {num_label_files}')
    print(f'Количество файлов в папке с изображениями: {num_img_files}')

    for epoch in range(epochs):
        running_loss = 0.0

        for img_file, label_file in zip(img_files, label_files):
            image = transform(Image.open(img_path + img_file)).to(device)
            with open(label_path + label_file, 'r') as labelfile:
                label_content = labelfile.read()

                optimizer.zero_grad()

                outputs = model(image.unsqueeze(0))
                outputs = torch.nn.functional.interpolate(outputs, size=label_content[-2:], mode='bilinear',
                                                          align_corners=False)

                loss = criterion(outputs, label_content)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}')


def train_custom_model(model, data_yaml, criterion, optimizer, epochs, batch_size=4, learning_rate=0.001):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])

    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)

    img_path = 'dataset/train/images/'
    label_path = 'dataset/train/labels/'

    img_files = os.listdir(img_path)
    num_img_files = len(img_files)

    label_files = os.listdir(label_path)
    num_label_files = len(label_files)

    print(f'Количество файлов в папке с метками: {num_label_files}')
    print(f'Количество файлов в папке с изображениями: {num_img_files}')

    model.to(device)
    model.train()

    for epoch in range(epochs):
        print("start")
        running_loss = 0.0

        for img_file, label_file in zip(img_files, label_files):
            image = transform(Image.open(img_path + img_file)).to(device)
            with open(label_path + label_file, 'r') as labelfile:
                label_content = labelfile.read()

                optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label_content)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(epochs)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}')

    print('Training complete.')