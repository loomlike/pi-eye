https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

train/
  masks/
    202409130000_mask.png
    202409130001_mask.png
    202409130002_mask.png
    202409130003_mask.png
    ...
  images/
    202409130000.png
    202409130001.png
    202409130002.png
    202409130003.png
    ...


## Labeling

https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2

1. Create **Data Labeling** project
1. Select **Image Classification Multi-label** task
1. Create data asset, select the type as **Folder** and then select the source **from Azure storage**.
