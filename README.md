 # **ML Model with Training Pipeline**
 This project implements a complete CI/CD pipeline for training and testing a CNN model on the MNIST dataset.
 The pipeline includes model training, testing, and automated validation through GitHub Actions.

 ## Project Structure
 Assignment_5/<br>
├── model.py <br>
├── train.py <br>
├── test_model.py <br>
├── requirements.txt<br>
└── .github/<br>
    └── workflows/<br>
        └── ml-pipeline.yml<br>

## Model Architecture
The project uses a lightweight CNN architecture designed for MNIST digit classification:

2 convolutional layers (8 and 16 filters) 

2 max pooling layers

2 Batch normalization layers

2 fully connected layers (24 neurons and 10 output classes)

Total parameters: 20386

## Key Features
* Automated training and testing pipeline
* Dataset size limited to 25,000 samples for faster training
* Model parameter count kept under 25,000 for efficiency
* Automated accuracy testing (>95% required)
* Model artifacts saved with timestamps
 


  
