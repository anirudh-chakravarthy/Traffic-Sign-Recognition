# Traffic-Sign-Recognition
Traffic sign classification on the German Traffic Sign Recognition Dataset.

## Requirements
To install the necessary packages, run:
`pip install -r requirements.txt`

The GTSRB dataset can be downloaded from [this link.](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)

## Usage 

### Training
To train a model, run the following command.
``` 
python train.py [-h] [-d BASE_DIR] [-m MODEL] [-e EPOCHS] [-b BATCH_SIZE]
                [--lr LR] [--num_classes NUM_CLASSES] [-c CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -d BASE_DIR, --base_dir BASE_DIR
                        path to dataset (default: None)
  -m MODEL, --model MODEL
                        Model to use. Choose between LeNet_baseline,
                        LeNet_modified, AlexNet, VGG16, VGG19 and ResNet18
                        (default: LeNet_baseline)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to run (default: 25)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (default: 1)
  --lr LR               learning rate (default: 1e-3)
  --num_classes NUM_CLASSES
                        Number of classes (default: 43 for GTSRB)
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        path to checkpoint directory
```

Architectures currently supported are: LeNet-5, AlexNet, VGG16, VGG19, 
ResNet18.

### Testing
To evaluate performance of a trained model, run the following command.
```
python test.py [-h] [-m MODEL] [--pretrained_model PRETRAINED_MODEL]
               [-d BASE_DIR] [--num_classes NUM_CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model to use (default: None)
  --pretrained_model PRETRAINED_MODEL
                        path to pretrained model (default: None)
  -d BASE_DIR, --base_dir BASE_DIR
                        path to dataset (default: None)
  --num_classes NUM_CLASSES
                        Number of classes (43 for GTSRB)
```


## Performance
Due to computational constraints, only LeNet could be trained. Other architectures were tested for correctness by training for 1 epoch.

Model | Val accuracy | Test accuracy
--- | --- | ---
LeNet baseline | 98.113% | 91.37%
LeNet modified | 98.636% | 94.36%

The following hyperparameters were used:
1. Optimizer: Adam with 1e-3 lr
2. Epochs: 25
3. Batch size: 64 