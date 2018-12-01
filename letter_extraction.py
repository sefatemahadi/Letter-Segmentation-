import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage import exposure
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening
from skimage.segmentation import clear_border
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from skimage.morphology import disk
from skimage.filters.rank import mean
from skimage.filters.rank import median
import numpy as np
from skimage.transform import resize
from scipy.misc import imsave
from scipy import ndimage as ndi
import cv2


def pre_image_processing(input_image):
    resized_image = resize(input_image, (300, 600))

    equal_adapt_hist_image = exposure.equalize_adapthist(resized_image)
    rescale_intensity_image = exposure.rescale_intensity(equal_adapt_hist_image)
    adjust_sigmoid_image = exposure.adjust_sigmoid(rescale_intensity_image)
    
    gray_scale_image =  rgb2gray(adjust_sigmoid_image)
    mean_image= mean(gray_scale_image, disk(1))
    median_image = median(mean_image, disk(1))
    otsu_image = filters.threshold_otsu(median_image)
    closing_image = closing(median_image > otsu_image, square(2))
    opening_image = opening(closing_image, square(2))
    clear_border_image = clear_border(opening_image)
    filled_area_image = ndi.binary_fill_holes(clear_border_image)
    label_objects, nb_labels = ndi.label(filled_area_image)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 40
    mask_sizes[0] = 0
    unnessasary_object_cleaned_image = mask_sizes[label_objects]
    labeled_image = label(unnessasary_object_cleaned_image)
    labeled_to_rgb_img = label2rgb(labeled_image, image=adjust_sigmoid_image)
    #plt.imshow(labeled_to_rgb_img,
    #         cmap='gray', interpolation='nearest')


    return labeled_image, resized_image, adjust_sigmoid_image, unnessasary_object_cleaned_image

def letter_segmentation(labeled_image, resize_image, adjust_sigmoid_image, unnessasary_object_cleaned_image):
    count = 0
    objects_regions = regionprops(labeled_image)
    for region in objects_regions:
        if region.area >= 650:
            minr, minc, maxr, maxc = region.bbox
            slice_hei = int((maxr - minr)* 0.1)
            colored_word = adjust_sigmoid_image[minr-slice_hei:maxr+slice_hei, minc:maxc]
            binary_image_word = unnessasary_object_cleaned_image[minr-slice_hei:maxr+slice_hei, minc:maxc]
            width = int(((maxc - minc)/((maxr - minr)*(1.05)))*100)
            resized_word_image = resize(colored_word, (100, width))
            binary_resized_word_image = resize(binary_image_word, (100, width)).astype(int)
            #plt.imshow(croped1)
            #print(binary_croped1)
            horizontal_histogram_projection = binary_resized_word_image.sum(axis=1)
            print(horizontal_histogram_projection)
            maximum_histogram_width = np.amax(horizontal_histogram_projection)
            last = None
            for i in range(50):
                
                if maximum_histogram_width-30 < horizontal_histogram_projection[i] < maximum_histogram_width+30:
                    binary_resized_word_image[i,:] = 0
                    last = i
            if not last == None:
                binary_resized_word_image[last,:] = 0
                binary_resized_word_image[last+1,:] = 0
                binary_resized_word_image[last+2,:] = 0
                binary_resized_word_image[last+3,:] = 0
                    
                #if 10<i<25:
                    #binary_resized_word_image[i,:] = 0 
            #plt.plot(binary_croped1)
            
            binary_resized_label_word_img = label(binary_resized_word_image)
            plt.imshow(binary_resized_label_word_img)
            char_regions = regionprops(binary_resized_label_word_img)
            #cv2.imshow('image',binary_croped_label)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyWindow()
            #print(croped)
            #print(char_regions)
            #plt.ion()
            for char_reg in char_regions:
                #print(char_reg.area)
                if char_reg.area >= 450:
                     import random
                     change = random.random()%255
                     #print(change)
                     cminr, cminc, cmaxr, cmaxc = char_reg.bbox
                     char_slice_hei = int((cmaxr - cminr)* 0.1)
                     #print(minr, minc, maxr, maxc)
                     #plt.plot(resized_word_image)
                     char_binary_img = binary_resized_word_image[cminr-char_slice_hei:cmaxr+char_slice_hei, cminc:cmaxc]
                     char_croped_si= resized_word_image[0:100, cminc:cmaxc]
                     #print("Here",camera.shape)
                     char_width = int(((cmaxc - cminc)/((cmaxr - cminr)*(1.1)))*100)
                     
                     char_img_resized= resize(char_croped_si, (100, char_width))
                     char_binary_img_resized = resize(char_binary_img, (100, char_width)).astype(int)
                     #plt.imshow(char_binary_img_resized)
                     io.imsave('charimage'+str(change)+'.png', char_img_resized)
            plt.show()
                     #rect = mpatches.Rectangle((cminc, cminr), cmaxc - cminc, (cmaxr - cminr)*(1.05),
                                      #fill=False, edgecolor=(change,0,0), linewidth=2)
                                      
            
            #print(histogram)
            
            #Show
            
            
            #his= np.histogram(croped1)
            #print(his)
            #slide(croped1)
            #io.all_warnings()
            #io.imsave('image'+str(count)+'.png', croped1)
            count+=1
            #plt.imshow(croped , cmap='gray', interpolation='nearest')
            rect = mpatches.Rectangle((minc, minr), maxc - minc, (maxr - minr)*(1.05),
                                    fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

inp_img = io.imread("1.jpg", as_gray=True)
fig, ax = plt.subplots(figsize=(10, 6))

labl_img, resized_image, adj_sig_img,unnessasary_object_cleaned_image = pre_image_processing(inp_img)
letter_segmentation(labl_img, resized_image, adj_sig_img, unnessasary_object_cleaned_image)

