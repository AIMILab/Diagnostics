# Paper Information

The repository of our following paper **An Unsupervised Learning based on Multiple Descriptors For WSIs Diagnosis** (_Diagnostics_): [Paper](http://empty.com)

1. ***Datasets***
 
- [ICIAR2018](https://iciar2018-challenge.grand-challenge.org/Dataset)
- [Dartmouth](https://bmirds.github.io/LungCancer)

2. ***Environment Setup***
 
Setup python environment 3.6.x ++ and install necessary packages and dependencies using dockerfile:
 - Numpy
 - Tensorflow 
 - Keras
 - Scikit-learn
 - Pandas
 - Scipy
 - Six

***!! NOTE: !!*** To evaluate the performance results faster we recommend to use docker files. We used Quadro RTX 5000 16GB. RAM 128GB

# Directory Structure:

The directory tree look like as follows:

- README.md
- Images
- DockerFiles (Docker.cpu, Docker.gpu)
- WSI Pre-processing
	- README.md (WSIs Used and Parameters Information for processing each dataset WSIs)
	- Create_WSIs_Patches_ICIAR
	- Create_WSIs_Patches_Dartmouth
	- WSI-Patches-Read-Write Pickle
	
- Our Model Scripts
	- Dataset-ICIAR
	  - Balanced_ICIAR_Binary+TSNE.ipynb
	  - Balanced_ICIAR_Multi+TSNE.ipynb
	  - UnBalanced_ICIAR_Binary+TSNE.ipynb
	  - UnBalanced_ICIAR_Multi+TSNE.ipynb
	  
	- Dataset-Dartmouth
	  - Balanced_Dartmouth_Multi+TSNE.ipynb

- All Other Model Scripts (8 State-of-the-art models in each file)
	- Dataset-ICIAR
	  - Balanced_ICIAR_Binary+TSNE.ipynb
	  - Balanced_ICIAR_Multi+TSNE.ipynb
	  - UnBalanced_ICIAR_Binary+TSNE.ipynb
	  - UnBalanced_ICIAR_Multi+TSNE.ipynb
	  
	- Dataset-Dartmouth
	  - Balanced_Dartmouth_Multi+TSNE.ipynb

# Commands To Use Docker Files:

To run a script from scratch you can use the following docker files.

  git clone https://github.com/AIMILab/Diagnostics.git
    
***A. Without CUDA Support:***

- Built the image from scratch (Docker.cpu file)

  sudo docker build -t diagnostics_2022:diagnostics_2022-cpu -f DockerFiles/Dockerfile_cpu .

  sudo docker run -it -v "$PWD"/Scripts:/root diagnostics_2022:diagnostics_2022-cpu

***B. With CUDA Support:***

- Built the image from scratch (Docker.gpu file)

  sudo docker build -t diagnostics_2022:diagnostics_2022-gpu -f DockerFiles/Dockerfile_gpu .

  sudo nvidia-docker run -it -v "$PWD"/Scripts:/root diagnostics_2022:diagnostics_2022-gpu

# Results:

**Confusion Matrices**

**A. Dataset-ICIAR**

<!-- -*Our Model* ![Dataset-2](/images/Confusion_Matrix_D1.png) -->
-*Our Model* <img src="/Images/Confusion_Matrix_D1.png" width="600" height="500">

**B. Dataset-Dartmouth**

<!-- -*Our Model* ![Dataset-2](/images/Confusion_Matrix_D2.png) -->
-*Our Model* <img src="/Images/Confusion_Matrix_D2.png" width="600" height="500">

**AUC(ROC) Curves**

**A. Dataset-ICIAR**

<!-- -*Our Model* ![Dataset-1](/images/ROC_D1.png) -->
-*Our Model* <img src="/Images/ROC_D1.png" width="600" height="300">

**B. Dataset-Dartmouth**

<!-- -*Our Model* ![Dataset-1](/images/ROC_D2.png) -->
-*Our Model* <img src="/Images/ROC_D2.png" width="600" height="300">

# Citation:

If you find this code useful in your research, please consider citing:

@article{Sheikh2022,
  title={An Unsupervised Learning based on Multiple Descriptors For WSIs Diagnosis},
  author={Taimoor Shakeel Sheikh, Jee Yeon Kim, Jaesool Shim, Migyung Cho},
  journal={arXiv preprint arXiv:Diagnostics},
  year={2022}
}
