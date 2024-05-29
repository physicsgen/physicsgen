import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd
import argparse


def apply_distortion(x, y, k1, k2, k3, p1, p2, fx, fy, cx, cy):
    # Umrechnen in Kamerakoordinaten
    x = (x - cx) / fx
    y = (y - cy) / fy
    r = np.sqrt(x**2 + y**2)

    # Anwenden der radialen Verzerrung
    x_distorted = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)
    y_distorted = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6)

    # Anwenden der tangentialen Verzerrung
    x_distorted += 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_distorted += p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    # Zurückumrechnen in Bildkoordinaten
    x_distorted = x_distorted * fx + cx
    y_distorted = y_distorted * fy + cy

    return x_distorted, y_distorted

def calc_dist_maps(image, k1, k2, k3, p1, p2, fx, fy, cx=128, cy=128):
    # Get the size of the image
    height, width = image.shape[:2]

    # Initialize the output matrices
    x_map = np.zeros((height, width), dtype=np.float32)
    y_map = np.zeros((height, width), dtype=np.float32)

    # Apply the distortion to each pixel
    for y in range(height):  # Notice y is for height
        for x in range(width):  # and x is for width
            x_new, y_new = apply_distortion(x, y, k1, k2, k3, p1, p2, fx, fy, cx, cy)
            # You need to ensure the mapping is valid; if it's out of bounds, you could remap it to the closest valid pixel
            if 0 <= x_new < width and 0 <= y_new < height:
                x_map[y, x] = x_new
                y_map[y, x] = y_new
            else:
                # Handle the out-of-bounds x_new and y_new by mapping them to the nearest valid pixel, or set them to some default value
                x_map[y, x] = 0
                y_map[y, x] = 0
                pass

    return x_map, y_map


## main function for evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--data_dir", type=str, default="data/true", help="Directory where the data is stored.")
    parser.add_argument("--output", type=str, default=".", help="Directory to save the processed output.")
    parser.add_argument("--csv_name", type=str, default="test.csv", help="Name of the CSV file to process.")

    args = parser.parse_args()

    data_dir = args.data_dir
    output = args.output
    csv_name = args.csv_name

    # Read the specified csv in path
    test_df = pd.read_csv(f"{data_dir}/{csv_name}")

    # Create the output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Use tqdm to create a progress bar for the loop
    for index, sample_row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Applying distortion"):
        # Check if the label image is available
        if not os.path.exists(f"{data_dir}/{sample_row['label_path']}"):
            print(f"Label image for sample {index} not found.")
            print(f"{data_dir}/{sample_row['label_path']}")
            continue

        # Load the image
        image = cv2.imread(f"{data_dir}/{sample_row['label_path']}")
        # Calculate the distortion maps
        x_map, y_map = calc_dist_maps(image, k1=sample_row['k1'], k2=sample_row['k2'], k3=sample_row['k3'],
                                      p1=sample_row['p1'], p2=sample_row['p2'], fx=200, fy=200)
        distorted_img = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        image_name = os.path.basename(sample_row['label_path'])
        # Save the distorted image
        cv2.imwrite(f"{output}/{image_name}", distorted_img)