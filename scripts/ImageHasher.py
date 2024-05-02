
import math
import random
import cv2

import numpy as np


class ImageHasher:

    def __init__(self) -> None:
        pass


    @staticmethod
    def generate_circular_points(img_width, img_height, min_radius, max_radius, num_points=100):
        radius = random.randint(min_radius, max_radius)
        center_x = random.randint(0 + radius, img_width - radius)
        center_y = random.randint(0 + radius, img_height- radius)

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
        cv2.imshow('Circle Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        return pixel_values
    
    @staticmethod 
    def hash_image(image_path, n_samples, min_radius, max_radius):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        # Loop through 
        for i in range(n_samples):
            coords = ImageHasher.generate_circular_points(width, height, min_radius, max_radius)
            pixel_values = ImageHasher.read_and_hash_image(image, coords)
            max_index = np.argmax(pixel_values)
            min_index = np.argmin(pixel_values)



if __name__ == '__main__':
    min_radius = 100
    max_radius = 200
    coords = ImageHasher.generate_circular_points(1920, 1080, min_radius, max_radius, int(max_radius*2*math.pi))
    pixel_values = ImageHasher.read_and_hash_image('images/line2/1.png', coords)

    print(pixel_values)