
# Basic Information

We used two additional descriptors along with the original patches extracted from WSIs which help our model to learn and classify the input samples more accurately with substantial information.
    
**A. HOG parameters**
1) Cells per block
2) Orientations
4) Pixels per cell.

**B. LBP parameters**
1) Method 
2) Number of patterns
3) Radius of circle

# Model Information

Our model comprises two modules.

**A. Autoencoder parameters (Unsupervised Learning)**
1) Number of hidden layers
2) Number of neurons per layer
3) Type of activation function
4) Type of reconstruction loss

**B. Autoencoder parameters (Fused Features Classification)**
1) Number of layers
2) Number of epochs 
3) Learning rate
4) Batch size
5) Optimizer
6) Batch normalization.


***!! NOTE: !!*** 

Each Jupyter notebook includes 

1) Our experimental strategy to classify the binary-class and multi-class samples for each WSI dataset.
2) We also added the confusion matrices and ROC curve analysis for each strategy.


