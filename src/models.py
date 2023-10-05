from torchvision import transforms
import clip
from torchvision import models
import torch
from torchvision.models import Inception_V3_Weights

def get_model(model="IneptionV3",device='cpu'):
    transform= transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
        ])
    if model=="InceptionV3":
        model = models.inception_v3(pretrained=True).to(device)
        model.aux_logits=False
        transform = Inception_V3_Weights.IMAGENET1K_V1.transforms()

    elif model=="CLIP":
        model, transform = clip.load('ViT-B/32', device=device)
    elif model=="SWAV":
        model=torch.hub.load('facebookresearch/swav','resnet50',pretrained=True).to(device)
    elif model=="LLR":
        # model=torch.jit.load('../FID_LLR/model_cifar_finetuned2.pth').to(device)
        model=torch.jit.load('../FID_LLR/model_cifar_lora_0.9.pth').to(device)

    elif model=="LLR1":
        model = torch.jit.load('../FID_LLR/model_cifar1.pth').to(device)
        # model=torch.load('../FID_LLR/model_cifar1.pth').to(device)
    model.eval()
    return model,transform