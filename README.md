# Inception_CLIP_SWAV_based_FID_Analysis


Inception ,CLIP, SWAV based_FID_Analysis for CelebA and CIFAR dataset

### Results:

| Evaluation Metric          | FID_InceptionV3 | FID_CLIP | FID_SWAV |
|---------------------------|-----------------|----------|----------|
| CIFAR100_Rot20deg         | 408.862         | 19.73    | 0.825    |
| CIFAR100_Rot50deg         | 537.452         | 19.485   | 1.247    |
| CIFAR100_Rot90deg         | 92.25           | 4.453    | 0.051    |
| CIFAR100_RandomRot30deg   | 175.363         | 10.393   | 0.42     |
| CIFAR100_HorizontalFlip   | 0.417           | 0.009    | 0.0      |
| CIFAR100_VerticalFlip     | 53.073          | 4.346    | 0.022    |
| CIFAR100_CirShift         | 31.154          | 4.328    | 0.084    |
| CIFAR100_ZoomIN           | 300.113         | 19.969   | 1.672    |
| CIFAR100_ZoomOut          | 51.932          | 8.428    | 0.507    |
| CIFAR100_ColorJitter      | 10.477          | 4.933    | 0.054    |
| CIFAR100_Gaussian_Blur    | 83.258          | 5.281    | 0.882    |
| CIFAR100_RandomPrespective| 470.906         | 16.584   | 1.294    |
| CIFAR100_RandomRotation_Scale| 746.468     | 27.214   | 2.463    |
| CIFAR100_ElasticTransform | 633.684         | 28.787   | 0.844    |
| CIFAR100_RandomPosterize  | 142.22          | 13.901   | 1.356    |
| CIFAR100_RandomAugmentation| 30.751        | 3.892    | 0.122    |
| CIFAR100_MixUp            | 18.337          | 1.118    | 0.053    |
| CelebA_Rot20deg           | 272.614         | 6.39     | 0.378    |
| CelebA_Rot50deg           | 354.683         | 14.841   | 0.802    |
| CelebA_Rot90deg           | 562.472         | 16.725   | 0.506    |
| CelebA_RandomRot30deg     | 106.523         | 5.349    | 0.266    |
| CelebA_HorizontalFlip     | 0.4             | 0.022    | 0.0      |
| CelebA_VerticalFlip       | 374.537         | 39.596   | 0.287    |
| CelebA_CirShift           | 26.983          | 4.701    | 0.082    |
| CelebA_ZoomIN             | 241.703         | 15.485   | 0.82     |
| CelebA_ZoomOut            | 29.727          | 2.584    | 0.14     |
| CelebA_ColorJitter        | 6.22            | 8.718    | 0.03     |
| CelebA_Gaussian_Blur      | 6.198           | 0.631    | 0.015    |
| CelebA_RandomPrespective  | 1093.524        | 12.348   | 0.784    |
| CelebA_RandomRotation_Scale| 893.771        | 22.479   | 1.199    |
| CelebA_ElasticTransform   | 829.366         | 35.086   | 5.471    |
| CelebA_RandomPosterize    | 181.956         | 10.02    | 0.465    |
| CelebA_RandomAugmentation | 58.123          | 5.13     | 0.148    |
| CelebA_MixUp              | 22.531          | 3.011    | 0.044    |
