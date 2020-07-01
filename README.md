A deep convolutional neural network was trained on a dataset of lung X-ray images from healthy patients as well as
those with COVID-19 and pneumonia in order to classify each condition given each class of image. As the dataset was balanced
due to a limited number of available COVID-19 images, two cycle consistent adversarial networks (CycleGAN) models were
implemented and trained to translate samples from the normal and pneumonia classes into COVID-19 samples. The dataset was
upsampled by generating images from these models. Accuracy, losses, and other classification metrics from the deep learning
model before and after training were compared. We found that upsampling COVID-19 samples with synthetic images using our
approach improved overall classification accuracy for the test dataset to 96.2% and achieved a faster convergence than the
baseline model.

This work was inspired by [this paper](https://pubmed.ncbi.nlm.nih.gov/32252036/). 
