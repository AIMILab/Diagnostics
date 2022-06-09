<!-- # Paper Information

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

- Scripts (Main File to run script)
- Dataset-1
  - Balanced_ICIAR_Binary+TSNE.ipynb
  - Balanced_ICIAR_Multi+TSNE.ipynb
  - UnBalanced_ICIAR_Binary+TSNE.ipynb
  - UnBalanced_ICIAR_Multi+TSNE.ipynb
  
- Dataset-2
  - Balanced_Dartmouth_Multi+TSNE.ipynb

- Other Models
  - Balanced_ICIAR_Binary+TSNE.ipynb
  - Balanced_ICIAR_Multi+TSNE.ipynb
  - UnBalanced_ICIAR_Binary+TSNE.ipynb
  - UnBalanced_ICIAR_Multi+TSNE.ipynb
  - Balanced_Dartmouth_Multi+TSNE.ipynb

# Commands To Run Script:

To run script you have two options either pull the build image or build from the scratch using docker files.

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

<!-- -*Our Model* ![Dataset-2](/images/Confusion_Matrix_D1.png) --
-*Our Model* <img src="/images/Confusion_Matrix_D1.png" width="600" height="500">

**B. Dataset-Dartmouth**

<!-- -*Our Model* ![Dataset-2](/images/Confusion_Matrix_D2.png) --
-*Our Model* <img src="/images/Confusion_Matrix_D2.png" width="600" height="500">

**AUC(ROC) Curves**

**A. Dataset-ICIAR**

<!-- -*Our Model* ![Dataset-1](/images/ROC_D1.png) --
-*Our Model* <img src="/images/ROC_D1.png" width="600" height="300">

**B. Dataset-Dartmouth**

<!-- -*Our Model* ![Dataset-1](/images/ROC_D2.png) --
-*Our Model* <img src="/images/ROC_D2.png" width="600" height="300">

# Citation:

If you find this code useful in your research, please consider citing:

@article{Sheikh2022,
  title={An Extended Unsupervised Deep Learning Model based on Multiple Descriptors For WSIs Diagnosis},
  author={Taimoor Shakeel Sheikh, Jee Yeon Kim, Jaesool Shim, Migyung Cho},
  journal={arXiv preprint arXiv:Diagnostics},
  year={2022}
}
 -->
 
 



Select different models

1) ResNet 
2) DensNet
3) Inception
4) MobileNet
5) RuneCNN
6) BreastNet
7) LiverNet
8) HCNNet



# Commands To Run Script:

	- The user-defined set of parameters depends on the dataset, as reported in the scripts (Create_Patches_ICIAR.py and Create_Patches_Dartmouth.py) for segmentation and patching from WSIs.
    
		A. **List of segmentation parameters**
			1) Level (seg_level): Downsample level on which to segment the WSI.
			2) Threshold (sthresh):  Positive integer, using a higher threshold leads to less foreground and more background detection.
			3) Median filter Threshold (mthresh):  Must be a positive and odd integer.
			4) Otsu's Method (use_otsu): default: True(otsu) or False(Binary thresholding)
			5) Morphological Operation (close): Positive integer or -1.

		B. **List of contour filtering parameters**
			1) Tissues Area filter Threshold (a_t): Positive integer, the minimum size of detected foreground contours to consider relative to a reference patch size at level N.
			2) Holes Area filter Threshold (a_h): Positive integer, the minimum size of detected holes/cavities in foreground contours to avoid.
			3) Maximum Holes detected per foreground contours (max_n_holes): Positive integer, higher maximum leads to more accurate patching but increases computational cost.

		C. **List of patching parameters**
			1) Padding(use_padding): True(to pad) or False (to not pad)
			2) Contour Function (contour_fn): Decide whether a patch should be considered foreground or background.
				a) four_pt: if all four points in a small, grid around the center of the patch are inside the contour.
				b) center: if the center of the patch is inside the contour. 
				c) basic: if the top-left corner of the patch is inside the contour.

		D. **List of visualization parameters**
			1) Level (vis_level): Downsample level to visualize the segmentation results.
			2) Visualize line (line_thickness): to visualize the segmentation results, Positive integer, in terms of the number of pixels occupied by the drawn line at level N.

		
	***A. ICIAR Dataset:***
		
		python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY --sourceXML DATA_DIRECTORY_XML --save_dir RESULTS_DIRECTORY --patch --patch_size 128	--seg --sthresh 8 --mthresh 9, --close 4 --use_otsu True --stitch

		Example:

		1) For Training WSIs

		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/Train/WSI/SVS
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/Train/WSI/XML

		2) For Testing WSIs

		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/Test/WSI/SVS
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/Test/WSI/XML

		3) For Saving Patched Images

		RESULTS_DIRECTORY = RESULTS_DIRECTORY_ICIAR

		- With XML
			python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --sourceXML DATA_DIRECTORY/ICIAR_2018/Train/WSI/XML --save_dir RESULTS_DIRECTORY_ICIAR --seg --patch --patch_size=256														  

		- Without XML
			python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --save_dir RESULTS_DIRECTORY_ICIAR --seg --patch --patch_size 256														  


	***B. Dartmouth Dataset:***
		
		python Create_WSIs_Patches_Dartmouth.py --source DATA_DIRECTORY --sourceXML DATA_DIRECTORY_XML --save_dir RESULTS_DIRECTORY --patch --patch_size 128	--seg --sthresh 8 --mthresh 9, --close 4 --use_otsu True --stitch

		Example:

		1) For WSIs

		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/WSI/SVS/Solid/
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/WSI/SVS/Acinar/
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/WSI/SVS/Lepidic/
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/WSI/SVS/Acinar/
		DATA_DIRECTORY = DATA_DIRECTORY/Dartmouth/WSI/SVS/Papillary/

		2) For Saving Patched Images

		RESULTS_DIRECTORY = RESULTS_DIRECTORY_Dartmouth

		- Without XML
			python Create_WSIs_Patches_Dartmouth.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --save_dir RESULTS_DIRECTORY_ICIAR --seg --patch --patch_size 256														  


		- Following 31 WSIs were used for pre-processing and to generate image patches for each WSI and was categorized into five classes.
			1) Acinar (DHMC_15, DHMC_27, DHMC_38, DHMC_83, DHMC_110, DHMC_121, DHMC_130, DHMC_133, DHMC_138)
			2) Lepidic (DHMC_18, DHMC_35, DHMC_54, DHMC_84, DHMC_109)
			3) Micropapillary (DHMC_33, DHMC_51, DHMC_55, DHMC_137, DHMC_139)
			4) Papillary (DHMC_24, DHMC_53, DHMC_98, DHMC_135) 
			5) Solid (DHMC_17, DHMC_39, DHMC_43, DHMC_45, DHMC_47, DHMC_49, DHMC_67, DHMC_128)



