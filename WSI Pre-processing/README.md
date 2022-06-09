# Pre-processing Information

The user-defined set of parameters depends on the dataset, as reported in the scripts (Create_WSIs_Patches_ICIAR.py and Create_WSIs_Patches_Dartmouth.py) for segmentation and patching from WSIs.
    
**A. List of segmentation parameters**
1) Level (seg_level): Downsample level on which to segment the WSI.
2) Threshold (sthresh):  Positive integer, using a higher threshold leads to less foreground and more background detection.
3) Median filter Threshold (mthresh):  Must be a positive and odd integer.
4) Otsu's Method (use_otsu): True(Otsu's) or False(Binary thresholding)
5) Morphological Operation (close): Positive integer or -1.

**B. List of contour filtering parameters**
1) Tissues Area filter Threshold (a_t): Positive integer, the minimum size of detected foreground contours to consider relative to a reference patch size at level N.
3) Holes Area filter Threshold (a_h): Positive integer, the minimum size of detected holes/cavities in foreground contours to avoid.
4) Maximum Holes detected per foreground contours (max_n_holes): Positive integer, higher maximum leads to more accurate patching but increases computational cost.

**C. List of patching parameters**
1) Padding(use_padding): True(to pad) or False (to not pad).
2) Contour Function (contour_fn): Decide whether a patch should be considered foreground or background.
	a) four_pt: if all four points in a small, grid around the center of the patch are inside the contour.
	b) center: if the center of the patch is inside the contour. 
	c) basic: if the top-left corner of the patch is inside the contour.

**D. List of visualization parameters**
1) Level (vis_level): Downsample level to visualize the segmentation results.
2) Visualize line (line_thickness): Positive integer, in terms of the number of pixels occupied by the drawn line at level N to display the segmentated results.

# Commands To Run Script

***A. ICIAR Dataset:*** (https://iciar2018-challenge.grand-challenge.org/Dataset)

python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY --sourceXML DATA_DIRECTORY_XML --save_dir RESULTS_DIRECTORY --patch --patch_size 256 --seg --stitch

Example:

1) For Training WSIs

	source = DATA_DIRECTORY/Dartmouth/Train/WSI/SVS
	
	sourceXML = DATA_DIRECTORY/Dartmouth/Train/WSI/XML

2) For Testing WSIs

	source = DATA_DIRECTORY/Dartmouth/Test/WSI/SVS
	
	sourceXML = DATA_DIRECTORY/Dartmouth/Test/WSI/XML

3) For Saving Patched Images

	save_dir = RESULTS_DIRECTORY_ICIAR

- **With XML**

	python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --sourceXML DATA_DIRECTORY/ICIAR_2018/Train/WSI/XML --save_dir RESULTS_DIRECTORY_ICIAR --patch --patch_size 256 --seg --stitch

- **Without XML**

	python Create_WSIs_Patches_ICIAR.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --save_dir RESULTS_DIRECTORY_ICIAR --patch --patch_size 256 --seg --stitch

***B. Dartmouth Dataset:*** (https://bmirds.github.io/LungCancer)

python Create_WSIs_Patches_Dartmouth.py --source DATA_DIRECTORY --sourceXML DATA_DIRECTORY_XML --save_dir RESULTS_DIRECTORY --patch --patch_size 256 --seg --stitch

Example:

1) For All WSIs

	source = DATA_DIRECTORY/Dartmouth/WSI/SVS/Solid/
	source = DATA_DIRECTORY/Dartmouth/WSI/SVS/Acinar/
	source = DATA_DIRECTORY/Dartmouth/WSI/SVS/Lepidic/
	source = DATA_DIRECTORY/Dartmouth/WSI/SVS/Acinar/
	source = DATA_DIRECTORY/Dartmouth/WSI/SVS/Papillary/

2) For Saving Patched Images

	save_dir = RESULTS_DIRECTORY_Dartmouth

- **Without XML**

 	python Create_WSIs_Patches_Dartmouth.py --source DATA_DIRECTORY/ICIAR_2018/Train/WSI/SVS --save_dir RESULTS_DIRECTORY_ICIAR --patch --patch_size 256 --seg														  

- **Following 31 WSIs were used for pre-processing and to generate image patches for each WSI and was categorized into five classes.**
	1) Acinar (DHMC_15, DHMC_27, DHMC_38, DHMC_83, DHMC_110, DHMC_121, DHMC_130, DHMC_133, DHMC_138)
	2) Lepidic (DHMC_18, DHMC_35, DHMC_54, DHMC_84, DHMC_109)
	3) Micropapillary (DHMC_33, DHMC_51, DHMC_55, DHMC_137, DHMC_139)
	4) Papillary (DHMC_24, DHMC_53, DHMC_98, DHMC_135) 
	5) Solid (DHMC_17, DHMC_39, DHMC_43, DHMC_45, DHMC_47, DHMC_49, DHMC_67, DHMC_128)
