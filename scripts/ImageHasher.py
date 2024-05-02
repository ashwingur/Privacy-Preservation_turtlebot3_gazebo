
import math
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class ImageHasher:

    def __init__(self) -> None:
        pass


    @staticmethod
    def generate_circular_points(img_width, img_height, min_radius, max_radius, num_points=100):
        radius = random.randint(min_radius, max_radius)
        center_x = random.randint(1 + radius, img_width - radius - 1)
        center_y = random.randint(1 + radius, img_height- radius - 1)

        angles = np.linspace(0, 2*np.pi, num_points)

        # Calculate x and y coorindates for each angle
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)

        coordinates = np.column_stack((x_coords, y_coords))

        return np.round(coordinates).astype(int)
    
    @staticmethod
    def read_and_hash_image(image, coordinates):

        pixel_values = np.array([image[y,x] for x,y in coordinates])
        
        for x, y in coordinates:
            image[int(y), int(x)] = 255  # Setting intensity to 255 (white)

        # max_index = np.argmax(pixel_values)
        # min_index = np.argmin(pixel_values)
        # print(max_index)

        # image[coordinates[max_index, 1],coordinates[max_index, 0]] = 0
        # image[coordinates[min_index, 1],coordinates[min_index, 0]] = 0
        # cv2.imshow('Circle Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
 
        return pixel_values
    
    @staticmethod 
    def hash_image(image_path, n_samples, min_radius, max_radius):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        hashed_image = np.zeros((256, 256))
        # Loop through 
        for i in range(n_samples):
            coords = ImageHasher.generate_circular_points(width, height, min_radius, max_radius)
            pixel_values = ImageHasher.read_and_hash_image(image, coords)
            # Let x be the max, and y be the min
            hashed_image[np.min(pixel_values),np.max(pixel_values)] += 1
        # hashed_image /= 100
        cmap_colors = [(0, 0, 0), (1, 1, 1)]  # (R, G, B) values
        custom_cmap = LinearSegmentedColormap.from_list('custom', cmap_colors)
        # plt.imshow(hashed_image, cmap='hot', interpolation='nearest')
        plt.imshow(np.log(hashed_image+1), cmap=custom_cmap, interpolation='nearest')
        # plt.imshow(hashed_image, cmap=custom_cmap, interpolation='nearest')
        plt.colorbar()  # Add a colorbar to show the intensity scale
        plt.title('Heatmap')
        plt.xlabel('Max intensity')
        plt.ylabel('Min intensity')
        plt.show()
        




if __name__ == '__main__':
    min_radius = 10
    max_radius = 10
    # coords = ImageHasher.generate_circular_points(1920, 1080, min_radius, max_radius, int(max_radius*2*math.pi))
    # pixel_values = ImageHasher.read_and_hash_image('images/line2/1.png', coords)
    # print(pixel_values)
    ImageHasher.hash_image('images/line2/726.png', 10000, min_radius, max_radius)