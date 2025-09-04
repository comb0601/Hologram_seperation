import cv2
import numpy as np
from skimage import io, color
from skimage.filters import sobel


def color_metrics(image):
    # Convert image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Calculate the histograms and bin edges for each color channel
    red_hist, red_bin_edges = np.histogram(image[:, :, 0], bins=256, range=(0, 256))
    green_hist, green_bin_edges = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
    blue_hist, blue_bin_edges = np.histogram(image[:, :, 2], bins=256, range=(0, 256))
    
    # Calculate metrics for each channel
    metrics = {}
    for channel, hist, bin_edges in zip(['red', 'green', 'blue'], 
                                        [red_hist, green_hist, blue_hist], 
                                        [red_bin_edges, green_bin_edges, blue_bin_edges]):
        # Calculate the weighted mean (weighted by frequency of each bin)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean = np.average(bin_centers, weights=hist)
        # Calculate the weighted standard deviation
        variance = np.average((bin_centers - mean)**2, weights=hist)
        std = np.sqrt(variance)
        
        metrics[f'{channel}_mean'] = mean
        metrics[f'{channel}_std'] = std
        metrics[f'{channel}_max'] = np.max(hist)
    
    # Calculate the mean intensity for the grayscale image
    metrics['gray_mean_intensity'] = np.mean(gray_image)
    
    return metrics 


def edge_metrics(image):
    # Convert image to grayscale
    gray_image = color.rgb2gray(image)
    
    # Apply Sobel edge detection
    edges = sobel(gray_image)
    
    # Calculate edge metrics
    metrics = {
        'edge_mean_intensity': np.mean(edges),
        'edge_std_intensity': np.std(edges),
        'edge_max_intensity': np.max(edges)
    }
    
    return metrics




def calculate_specularity_index(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    specularity_index = np.max(gray_image)
    return specularity_index


def calculate_contrast_ratio(image): 

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    _, specular_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

    contrast_ratio = np.mean(gray_image[specular_mask == 255]) / np.mean(gray_image[specular_mask == 0])
    return contrast_ratio





image1 = cv2.imread('./real/output/min_image.jpg')
image2 = cv2.imread('./real/output/max_mean_image.jpg')
image3 = cv2.imread('./print/output/mean_image.jpg')


image1_color_metrics = color_metrics(image1)
image1_edge_metrics = edge_metrics(image1)
image1_specularity_index = calculate_specularity_index(image1)
image1_contrast_ratio = calculate_contrast_ratio(image1)

image2_color_metrics = color_metrics(image2)
image2_edge_metrics = edge_metrics(image2)
image2_specularity_index = calculate_specularity_index(image2)
image2_contrast_ratio = calculate_contrast_ratio(image2)


image3_color_metrics = color_metrics(image3)
print(" ")
print(" ")
print(image3_color_metrics)

print(" ")
print(" ")
print("Color_Metrics:", image1_color_metrics)
print("Edge_Metrics:", image1_edge_metrics)
print("Specularity Index:", image1_specularity_index)
print("Contrast Ratio:", image1_contrast_ratio)

print(" ")
print(" ")
print("Color_Metrics:", image2_color_metrics)
print("Edge_Metrics:", image2_edge_metrics)
print("Specularity Index:", image2_specularity_index)
print("Contrast Ratio:", image2_contrast_ratio)


print(" ")
print(" ")
print("Red_std", image2_color_metrics['red_std'] / image1_color_metrics['red_std']) 
print("Green_std:", image2_color_metrics['green_std'] / image1_color_metrics['green_std']) 
print("Blue_std:", image2_color_metrics['blue_std'] / image1_color_metrics['blue_std'])
print(" ")
print(" ")
print("Red_mean", image2_color_metrics['red_mean'] / image1_color_metrics['red_mean']) 
print("Green_mean:", image2_color_metrics['green_mean'] / image1_color_metrics['green_mean']) 
print("Blue_mean:", image2_color_metrics['blue_mean'] / image1_color_metrics['blue_mean'])
print(" ")
print(" ")
print("Red_max", image2_color_metrics['red_max'] / image1_color_metrics['red_max']) 
print("Green_max:", image2_color_metrics['green_max'] / image1_color_metrics['green_max']) 
print("Blue_max:", image2_color_metrics['blue_max'] / image1_color_metrics['blue_max'])
print(" ")
print(" ")
print("Edge_std:", image2_edge_metrics['edge_std_intensity'] / image1_edge_metrics['edge_std_intensity']) 
print("Edge_std:", image2_edge_metrics['edge_std_intensity'] / image1_edge_metrics['edge_std_intensity']) 
print(" ")
print(" ")
print("Edge_mean:", image2_edge_metrics['edge_mean_intensity'] / image1_edge_metrics['edge_mean_intensity']) 
print("Edge_mean:", image2_edge_metrics['edge_mean_intensity'] / image1_edge_metrics['edge_mean_intensity']) 
print(" ")
print(" ") 
print("Specular:", image2_specularity_index / image1_specularity_index) 