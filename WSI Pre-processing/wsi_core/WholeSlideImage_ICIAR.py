
#from openslide import open_slide # http://openslide.org/api/python/

#### Uncomment this for Windows #### 
# import os
# os.environ['PATH'] = "C:/Users/Administrator/Downloads/openslide-win64-20160612/openslide-win64-20160612/bin" + ";" + os.environ['PATH']
# import openslide

import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image
import pdb
import h5py
import math
from wsi_core.wsi_utils_ICIAR import savePatchIter_bag_hdf5, initialize_hdf5_bag

def DrawGrid(img, coord, shape, thickness=2, color=(0,0,0,255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord-thickness//2)), tuple(coord - thickness//2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img

def DrawMap(canvas, patch_dset, coords, patch_size, indices=None, verbose=1, draw_grid=True):
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)
        print('start stitching {}'.format(patch_dset.attrs['wsi_name']))
    
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))
        
        patch_id = indices[idx]
        patch = patch_dset[patch_id]
        
        
        try:
            patch = cv2.resize(patch (64, 64), interpolation=cv2.INTER_AREA)
            print(img.shape)
        except:
            break
        
        height, width , layers = patch.shape
        size=(width,height)
        print(size)
        patch_size = img.shape

        coord = coords[patch_id]
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(0,0,0), bg_color_=(255,255,255), alpha=-1):
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['imgs']
    coords = file['coords'][:]
    if 'downsampled_level_dim' in dset.attrs.keys():
        w, h = dset.attrs['downsampled_level_dim']
    else:
        w, h = dset.attrs['level_dim']
    print('original size: {} x {}'.format(w, h))
    w = w // downscale
    h = h //downscale
    coords = (coords / downscale).astype(np.int32)
    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(dset)))
    img_shape = dset[0].shape
    print('patch shape: {}'.format(img_shape))
    downscaled_shape = (img_shape[1] // downscale, img_shape[0] // downscale)

    if w*h > Image.MAX_IMAGE_PIXELS: 
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)
    
    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w,h), mode="RGB", color=bg_color)
        heatmap_ = Image.new(size=(w,h), mode="RGB", color=bg_color_)
    else:
        heatmap = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
        heatmap_ = Image.new(size=(w,h), mode="RGBA", color=bg_color_ + (int(255 * alpha),))
    
    heatmap = np.array(heatmap)
    heatmap = DrawMap(heatmap, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)

    heatmap_ = np.array(heatmap_)
    heatmap_ = DrawMap(heatmap_, dset, coords, downscaled_shape, indices=None, draw_grid=draw_grid)
    
    file.close()
    return heatmap, heatmap_

class WholeSlideImage(object):
    def __init__(self, path, full_path_, hdf5_file=None):
        self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.wsi = openslide.open_slide(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
        self.xml = ET.parse(full_path_)
        print(full_path_)

        self.check = 2
        self.contours_tissue = None
        self.contours_disease1 = None    
        self.contours_disease2 = None    
        self.contours_disease3 = None    
        self.contours_disease4 = None    
        self.hdf5_file1 = hdf5_file
        self.hdf5_file2 = hdf5_file
        self.hdf5_file3 = hdf5_file
        self.hdf5_file4 = hdf5_file
        self.hdf5_file5 = hdf5_file
        # self.seg_level = None
        print(self.name)
        self.scale = None
        
    def getOpenSlide(self):
        return self.wsi
        
    def getOpenSlideXML(self):
        return self.xml

    def initXML_ICIAR(self, **seg_params):
                            
        def _createContour(coord_list):
            coord = []
            for v in coord_list:
                x = int(v.get('X'))
                y = int(v.get('Y'))
                coord += [[x,y]]
        
            return np.array([coord], dtype = 'int32')
        
        root = self.xml.getroot()
        annotations = root.findall('Annotation')
        print(annotations)
        print("\nNo. of Annotations: ", len(annotations))
        
        for a in annotations:
            print("\nAnnotation ID: ",int(a.get('Id')))
            regions = root[int(a.get('Id'))-1][1].findall('Region')
            print("No. of Regions: ", len(regions))
            
            pixel_spacing = float(root.get('MicronsPerPixel'))
            
            labels = []
            coords_disease1,coords_disease2,coords_disease3,coords_disease4 = [],[],[],[]
            length = []
            coords = []
            area = []
            id = []
                        
            for r in regions:
                print("Region ID: ",int(r.get('Id')))
                area += [float(r.get('AreaMicrons'))]
                length += [float(r.get('LengthMicrons'))]
                try:
                    label = r[0][0].get('Value')
                except:
                    label = r.get('Text')
                if 'benign' in label.lower():
                    label = 1
                elif 'in situ' in label.lower():
                    label = 2
                elif 'invasive' in label.lower():
                    label = 3
                else:
                    label = 4
                    
                labels += [label]
                vertices = r[1]
                print("Label: ",label)
                if label==1:
                    coords_disease1 += [_createContour(vertices)]
                elif label==2:
                    coords_disease2 += [_createContour(vertices)]
                elif label==3:
                    coords_disease3 += [_createContour(vertices)]
                elif label==4:
                    coords_disease4 += [_createContour(vertices)]
        
            self.contours_disease1  = coords_disease1
            self.contours_disease2  = coords_disease2
            self.contours_disease3  = coords_disease3
            self.contours_disease4  = coords_disease4
        
        print("\nNo. of Regions Disease 1 (Pathologist): ", len(self.contours_disease1))
        print("\nNo. of Regions Disease 2 (Pathologist): ", len(self.contours_disease2))
        print("\nNo. of Regions Disease 3 (Pathologist): ", len(self.contours_disease3))
        print("\nNo. of Regions Disease 4 (Pathologist): ", len(self.contours_disease4))
    
        self.contours_disease1 = sorted(self.contours_disease1, key=cv2.contourArea, reverse=True)
        self.contours_disease2 = sorted(self.contours_disease2, key=cv2.contourArea, reverse=True)
        self.contours_disease3 = sorted(self.contours_disease3, key=cv2.contourArea, reverse=True)
        self.contours_disease4 = sorted(self.contours_disease4, key=cv2.contourArea, reverse=True)

    def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, mthresh=7, close = 0, use_otsu=False, 
                            filter_params={'a':100}, ref_patch_size=512):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)

            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                a = cv2.contourArea(cont)
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)

            all_holes = []
            for parent in filtered:
                all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
        
        img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
                
        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

        scale = self.level_downsamples[seg_level]
        self.scale = scale
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        print(len(contours))
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
        print(hierarchy.shape)
        if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
                                         
        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

    def visWSI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
                    line_thickness=12, max_size=None, crop_window=None, annot_color_=(255,125,255)):
        img = np.array(self.wsi.read_region((0,0), vis_level, self.level_dim[vis_level]).convert("RGB"))
        img_ = np.array(self.wsi.read_region((0,0), vis_level, self.level_dim[vis_level]).convert("RGB"))
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if self.contours_tissue is not None:
            cv2.drawContours(img_, self.scaleContourDim(self.contours_tissue, scale), 
                             -1, color, line_thickness, lineType=cv2.LINE_8)

            for holes in self.holes_tissue:
                cv2.drawContours(img_, self.scaleContourDim(holes, scale), 
                                 -1, hole_color, line_thickness, lineType=cv2.LINE_8)

        if self.check==2:
            if self.contours_disease1 is not None:
                annot_color=(125,100,250)
                cv2.drawContours(img_, self.scaleContourDim(self.contours_disease1, scale), 
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8)

            if self.contours_disease2 is not None:
                annot_color=(36,125,12)
                cv2.drawContours(img_, self.scaleContourDim(self.contours_disease2, scale), 
                                 -1, annot_color_, line_thickness, lineType=cv2.LINE_8)
                                 
            if self.contours_disease3 is not None:
                annot_color_=(225,255,15)
                cv2.drawContours(img_, self.scaleContourDim(self.contours_disease3, scale), 
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8)

            if self.contours_disease4 is not None:
                annot_color_=(125,255,125)
                cv2.drawContours(img_, self.scaleContourDim(self.contours_disease4, scale), 
                                 -1, annot_color_, line_thickness, lineType=cv2.LINE_8)
        
        img = Image.fromarray(img)
        img_ = Image.fromarray(img_)
        if crop_window is not None:
            top, left, bot, right = crop_window
            left = int(left * scale[0])
            right = int(right * scale[0])
            top =  int(top * scale[1])
            bot = int(bot * scale[1])
            crop_window = (top, left, bot, right)
            img = img.crop(crop_window)
            img_ = img_.crop(crop_window)
        w, h = img_.size
        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
            img_ = img_.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img , img_

    def createPatches_bag_hdf5(self, save_path, patch_level=0, patch_size=256, step_size=256, save_coord=True, **kwargs):

        print("\nCreating patches for: ", self.name, "...\n")
        elapsed = time.time()
        countL=0
                
        if self.check==2:
            if self.contours_disease1:
                countL = countL +1
            if self.contours_disease2:
                countL = countL +1
            if self.contours_disease3: 
                countL = countL +1
            if self.contours_disease4:
                countL = countL +1

            countL = countL + 5
            for ib in range(0,countL):
                if ib==0 and len(self.contours_tissue)>0:
                    contours = self.contours_tissue
                    contour_holes = self.holes_tissue
                    name_ = self.name+'_Original'
                    Type_='Original'
                    m = 1

                if ib==1 and len(self.contours_disease1)>0:
                    contours = self.contours_disease1
                    contour_holes = None
                    name_ = self.name+'_D1'
                    Type_='D1'
                    m = 1

                if ib==2 and len(self.contours_disease2)>0:
                    contours = self.contours_disease2
                    contour_holes = None
                    name_ = self.name+'_D2'
                    Type_='D2'
                    m = 1

                if ib==3 and len(self.contours_disease3)>0:
                    contours = self.contours_disease3
                    contour_holes = None
                    name_ = self.name+'_D3'
                    Type_='D3'
                    m = 1

                if ib==4 and len(self.contours_disease4)>0:
                    contours = self.contours_disease4
                    contour_holes = None
                    name_ = self.name+'_D4'
                    Type_='D4'
                    m = 1
                    
                try:
                    print(len(contours))
                    for idx, cont in enumerate(contours):
                        print(name_)
                        patch_gen = self._getPatchGenerator(name_, cont, idx, patch_level, save_path, patch_size, step_size, contour_holes, **kwargs)                    
                            
                        if self.hdf5_file1 is None and ib==0:
                            try:
                                first_patch = next(patch_gen)
                                
                            except StopIteration:
                                continue
                                
                            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                            self.hdf5_file1 = file_path

                        elif self.hdf5_file2 is None and ib==1:
                            try:
                                first_patch = next(patch_gen)
                                
                            except StopIteration:
                                continue
                                
                            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                            self.hdf5_file2 = file_path
                                
                        elif self.hdf5_file3 is None and ib==2:
                            try:
                                first_patch = next(patch_gen)
                                
                            except StopIteration:
                                continue
                                
                            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                            self.hdf5_file3 = file_path
                                
                        elif self.hdf5_file4 is None and ib==3:
                            try:
                                first_patch = next(patch_gen)
                                
                            except StopIteration:
                                continue
                                
                            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                            self.hdf5_file4 = file_path
                                
                        elif self.hdf5_file5 is None and ib==4:
                            try:
                                first_patch = next(patch_gen)
                                
                            except StopIteration:
                                continue
                                
                            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
                            self.hdf5_file5 = file_path

                        for patch in patch_gen:
                            savePatchIter_bag_hdf5(patch,m)
                            m = m + 1
                          
                        del patch_gen
                        
                    del contours, m,  ib , name_, first_patch, file_path
                    
                except:
                    continue
                

        if self.check==1:
            return self.hdf5_file1, self.hdf5_file2, self.hdf5_file3
        elif self.check==2:
            return self.hdf5_file1, self.hdf5_file2, self.hdf5_file3, self.hdf5_file4, self.hdf5_file5

            
    def _getPatchGenerator(self, name_, cont, cont_idx, patch_level, save_path, patch_size=256, step_size=256, holes_tissue=None, custom_downsample=1,
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))
        
        if custom_downsample > 1:
            assert custom_downsample == 2 
            target_patch_size = patch_size 
            patch_size = target_patch_size * 2 
            step_size = step_size * 2
            print("Custom Downsample: {}, Patching at {} x {}, But Final Patch Size is {} x {}".format(custom_downsample, patch_size, patch_size, 
                target_patch_size, target_patch_size))

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        
        if contour_fn == 'four_pt':
            cont_check_fn = self.isInContourV3
        elif contour_fn == 'center':
            cont_check_fn = self.isInContourV2
        elif contour_fn == 'basic':
            cont_check_fn = self.isInContourV1
        else:
            raise NotImplementedError

        img_w, img_h = self.level_dim[0]
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1])
            stop_x = min(start_x+w, img_w-ref_patch_size[0])

        count = 0
        for y in range(start_y, stop_y, step_size_y):
            for x in range(start_x, stop_x, step_size_x):
                if holes_tissue is not None:
                    if not self.isInContours(cont_check_fn, cont, (x,y), holes_tissue[cont_idx], ref_patch_size[0]): #point not inside contour and its associated holes
                        continue    
                        
                if holes_tissue is None:
                    if self.isInContours(cont_check_fn, cont, (x,y), holes_tissue, ref_patch_size[0]): #point not inside contour and its associated holes
                        continue    
                   
                
                count+=1
                patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                if custom_downsample > 1:
                    patch_PIL = patch_PIL.resize((target_patch_size, target_patch_size))
                
                if white_black:
                    if self.isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or self.isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue

                if count>0:
                    check_patches = 1
                else:
                    check_patches = 0
                    
                # x, y coordinates become the coordinates in the downsample, and no long correspond to level 0 of WSI
                patch_info = {'x':x // (patch_downsample[0] * custom_downsample), 'y':y // (patch_downsample[1] * custom_downsample), 'cont_idx':cont_idx, 'patch_level':patch_level, 
                'downsample': self.level_downsamples[patch_level], 'downsampled_level_dim': tuple(np.array(self.level_dim[patch_level])//custom_downsample), 'level_dim': self.level_dim[patch_level],
                'patch_PIL':patch_PIL, 'name':name_, 'save_path':save_path, 'check_patches':check_patches}

                yield patch_info

        
        print("patches extracted: {}".format(count), "\n")

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContourV1(cont, pt, patch_size=None):
        return 1 if cv2.pointPolygonTest(cont, pt, False) >= 0 else 0

    @staticmethod
    def isInContourV2(cont, pt, patch_size=256):
        return 1 if cv2.pointPolygonTest(cont, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) >= 0 else 0

    @staticmethod
    def isInContourV3(cont, pt, patch_size=256):
        center = (pt[0]+patch_size//2, pt[1]+patch_size//2)
        all_points = [(center[0]-patch_size//4, center[1]-patch_size//4),
                      (center[0]+patch_size//4, center[1]+patch_size//4),
                      (center[0]+patch_size//4, center[1]-patch_size//4),
                      (center[0]-patch_size//4, center[1]+patch_size//4)
                      ]
        for points in all_points:
            if cv2.pointPolygonTest(cont, points, False) >= 0:
                return 1

        return 0

    @staticmethod
    def isInContours(cont_check_fn, contour, pt, holes=None, patch_size=256):
        if cont_check_fn(contour, pt, patch_size):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0
    
    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]
    
    @staticmethod
    def isWhitePatch(patch, satThresh=5):
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

    @staticmethod
    def isBlackPatch(patch, rgbThresh=40):
        return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        
        return level_downsamples
