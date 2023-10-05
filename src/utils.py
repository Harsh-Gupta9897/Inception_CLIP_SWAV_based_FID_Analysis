import torch ,numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from dataset import mixup_transformed_inputs
def get_features(loader,model,fid_model_type="CLIP",dataset="CIFAR100", device='cpu',augment_type=None):
    all_features = []
    num_classes = 1000 if dataset=="CIFAR100" else 1
    flag=True
    with torch.no_grad():
        for images, labels in tqdm(loader):
            if augment_type=="MixUp":
                images = mixup_transformed_inputs(images,labels,num_classes)
                
            if fid_model_type=="CLIP":
                features = model.encode_image(images.to(device))
            elif fid_model_type=="InceptionV3":
                features = model(images.to(device)).data
            elif fid_model_type=="SWAV":
                features = model(images.to(device)).data
            elif fid_model_type=="LLR1":
                features = model(images.to(device)).data
            elif fid_model_type=="LLR":
                features = model(images.to(device)).data
            all_features.append(features)

            if augment_type!=None and flag:
                plot_images(images,f'{fid_model_type}_{dataset}_{augment_type}.png')
                flag=False
            elif flag:
                plot_images(images,f't_{fid_model_type}_{dataset}.png')
                flag=False

    return torch.cat(all_features).cpu().numpy()



def plot_images(batch_images,fig_name="CIFAR100"):
    batch_images = batch_images.clone().detach()
    batch_images = batch_images.mul(0.5).add(0.5)  # Unnormalize the images if necessary
    grid_images = vutils.make_grid(batch_images, nrow=8, padding=2, normalize=False)

    # Convert tensor to numpy array and transpose dimensions
    grid_images = F.to_pil_image(grid_images)
    
    grid_images = grid_images.convert("RGB")
    grid_images = np.array(grid_images)
#     grid_images = np.transpose(grid_images, (x))

    # Display the grid of images
    # plt.figure()
    plt.imshow(grid_images)
    plt.axis('off')
    
    plt.savefig(f'./Images/{fig_name}')




  