import os
import argparse
import pandas as pd
import face_alignment
import numpy as np
from tqdm import tqdm
from skimage import io


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')

def get_landmark_error(label, pred):
    # Vectorized Euclidean distance calculation
    return np.mean(np.sqrt(np.sum((np.array(label) - np.array(pred))**2, axis=1)))

def get_landmark_error_xy(label, pred):
    label = np.array(label)
    pred = np.array(pred)

    # Calculate the mean absolute error for x and y coordinates separately
    x_error = np.mean(np.abs(label[:, 0] - pred[:, 0]))
    y_error = np.mean(np.abs(label[:, 1] - pred[:, 1]))

    return x_error, y_error


def calculate_landmark_error(distorted_path:str, pred_path:str):
    # Load the label image
    label = io.imread(distorted_path)
    # Load the distorted image
    gan = io.imread(pred_path)
    # Get the landmarks for the label imag
    label_preds = fa.get_landmarks(label)
    # Get the landmarks for the distorted image
    gan_landmarks = fa.get_landmarks(gan)
    if label_preds is None or gan_landmarks is None:
        return None
        
    return get_landmark_error(label_preds[0], gan_landmarks[0]), get_landmark_error_xy(label_preds[0], gan_landmarks[0])

## main function for evaluation
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/true")
    parser.add_argument("--pred_dir", type=str, default="data/pred")
    parser.add_argument("--output", type=str, default=".")
    args = parser.parse_args()

    data_dir = args.data_dir
    pred_dir = args.pred_dir
    output = args.output

    test_df = pd.read_csv(f"{data_dir}/test.csv")
    results = []

    # Use tqdm to create a progress bar for the loop
    for index, sample_row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Evaluating samples"):
        # Check if prediction is available
        if not os.path.exists(f"{pred_dir}/y_{index}.png"):
            print(f"Prediction for sample {index} not found.")
            print(f"{pred_dir}/y_{index}.png")
            continue

        # Check if the label image is available
        if not os.path.exists(f"{data_dir}/{sample_row['distortion_path']}"):
            print(f"Label image for sample {index} not found.")
            print(f"{data_dir}/{sample_row['distortion_path']}")
            continue

        # Calculate the error
        error, (error_x, error_y) = calculate_landmark_error(f"{data_dir}/{sample_row['distortion_path']}", f"{pred_dir}/y_{index}.png")
        if error is not None:
            results.append([index, error, error_x, error_y])

    # Save the results to a CSV file
    results_df = pd.DataFrame(results, columns=["index", "error", "error_x", "error_y"])
    results_df.to_csv(os.path.join(output, "lens_output.csv"), index=False)
    print(f"Results saved to {output}")
    print(results_df.describe())
        

