import clip,torch

from dataset import get_loader
from utils import get_features
from FID import compute_fid
from models import get_model
import time
import pandas as pd

if __name__=='__main__':

    datasets = [ "CIFAR100"] #,"CelebA",]
    fid_models = ["LLR"] # ,  ["InceptionV3","CLIP","SWAV"] 
    results_time = {}  # Dictionary to store results for execution time
    results_fid = {}
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    augment_types = [
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
        "MixUp",
    ]
    for fid_model in fid_models:
        model, transform = get_model(fid_model, device)
        for dataset in datasets:
            for augment_type in augment_types:
                train_loader, augment_loader = get_loader(transform, dataset,augment_type)
                start_time = time.time()
                augmented_features = get_features(augment_loader, model, fid_model, dataset, device,augment_type)
                train_features = get_features(train_loader, model, fid_model,  dataset,device)
                fid_value = compute_fid(train_features, augmented_features)
                execution_time = time.time() - start_time

                if "FID_"+fid_model not in results_time:
                    results_time["FID_"+fid_model] = {}
                    results_fid["FID_"+fid_model] = {}
            

                # Save execution time in the results_time dictionary
                results_time["FID_"+fid_model][dataset + '_' + augment_type] = round(execution_time, 3)

                # Save FID value in the results_fid dictionary
                results_fid["FID_"+fid_model][dataset + '_' + augment_type] = round(fid_value, 3)
                
                # Create dataframes from the results dictionaries
                time_df = pd.DataFrame(results_time)
                fid_df = pd.DataFrame(results_fid)

                time_df.to_csv('experiments/execution_time_llr_adaptor0.9_cifar.csv', index_label='Evaluation Metric')
                fid_df.to_csv('experiments/fid_value_llr_adaptor0.9_cifar.csv', index_label='Evaluation Metric')
                
                # print("Results saved to execution_time.csv and fid_value.csv")
                print(f"Model: {fid_model}\t Dataset:{dataset}\t Augmentation: {augment_type}")

    