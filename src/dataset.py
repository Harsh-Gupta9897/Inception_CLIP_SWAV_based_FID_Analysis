import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader,Dataset
import numpy as np
# import random
import glob
from PIL import Image
# from utils import plot_images
import timm
# from tqdm import tqdm
class CelebADataset(Dataset):
    def __init__(self, transform=None,train_paths=None,augment=False):
        super().__init__()
        self.train_paths = train_paths
        self.transform = transform
        self.augment = augment
    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, index):
        image_path = self.train_paths[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return [image,0]


def random_crop_percentage(img, crop_percentage):
    width, height = img.size
    crop_size = int(min(width, height) * crop_percentage)
    crop_transform = transforms.RandomCrop(crop_size)
    return crop_transform(img)

# def mixup_transform(image1, image2, alpha):
#     print(image1.size)
#     print(image2.size)
#     assert image1.size == image2.size, "The two images must have the same size."

#     lam = random.betavariate(alpha, alpha)

#     mixed_image = transforms.functional.blend(image1, image2, lam)

#     return mixed_image


def get_loader(transform,dataset="CIFAR100", augment_type="Rot20deg"):
    
    # Define the augmentations
    # Define the augmentations
    augmentations = [
        transforms.RandomRotation((20,20)),
        transforms.RandomRotation((50,50)),
        transforms.RandomRotation((90,90)),
        transforms.RandomRotation((-30, 30)),
        transforms.RandomHorizontalFlip(1.0),
        transforms.RandomVerticalFlip(1.0),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Circular Shift
        transforms.Lambda(lambda x: random_crop_percentage(x, 0.5)),# ZoomIN
        transforms.RandomAffine(0, scale=(1.2, 1.2)),  # ZoomOut
        transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=1.0),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.ElasticTransform(alpha=250.0),
        transforms.RandomPosterize(bits=3,p=1.0),
        transforms.RandAugment()
    ]

    # Define the keys for the augmentations
    augmentation_keys = [
        "Rot20deg",
        "Rot50deg",
        "Rot90deg",
        "RandomRot30deg",
        "HorizontalFlip",
        "VerticalFlip",
        "CirShift",
        "ZoomIN",
        "ZoomOut",
        "ColorJitter",
        "Gaussian_Blur",
        "RandomPrespective",
        "RandomRotation_Scale",
        "ElasticTransform",
        "RandomPosterize",
        "RandomAugmentation",
    ]

    # Convert the augmentations list to a dictionary with specific keys
    augmentation_dict = {key: augmentation for key, augmentation in zip(augmentation_keys, augmentations)}
    combined_transform = transforms.Compose(
        [   augmentation_dict[augment_type],transform] if augment_type!='MixUp' else [transform,]
        )
    print(combined_transform)

    if dataset=='CIFAR100':
        train_set = torchvision.datasets.CIFAR100(root='./data/',train=True,download=True,transform=transform) 
        augmented_set = torchvision.datasets.CIFAR100(root='./data/',train=True,transform=combined_transform) 
    elif dataset=='CelebA':
        data_url ='./data/Celeba_512/'
        train_paths = glob.glob(data_url+'*')

        train_set = CelebADataset(transform=transform,train_paths=train_paths)
        augmented_set = CelebADataset(transform=combined_transform,train_paths=train_paths) 
    
    train_loader= DataLoader(train_set, batch_size=100,num_workers=10)
    augmented_loader= DataLoader(augmented_set, batch_size=100,num_workers=10)
    
    
    return train_loader,augmented_loader


def mixup_transformed_inputs(inputs, classes,num_classes=1000):
    mixup_args = {
            'mixup_alpha': 0.5,
            'cutmix_alpha': 0.0,
            'cutmix_minmax': None,
            'prob': 1.0,
            'switch_prob': 0.,
            'mode': 'batch',
            'label_smoothing': 0,
            'num_classes': 1000}
    mixup_transform = timm.data.mixup.Mixup(**mixup_args)
    inputs,_= mixup_transform(inputs,classes)
    return inputs

# if __name__=='__main__':
#     transform= transforms.Compose([
#                 transforms.Resize((512, 512)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                
#             ])
#     train_loader, augment_loader  = get_loader(transform,"CIFAR100",augment_type="MixUp")
#     if augment_type=="MixUp":
        
#         for inputs,classes in tqdm(augment_loader):
#             inputs= mixup_transformed_inputs(inputs,classes,1000)
#     plot_images(inputs,"MixUp_Demo")

