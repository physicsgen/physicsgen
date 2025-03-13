import os, cv2, argparse
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, DownloadMode


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


def calc_dist_maps_vectorized(image, k1, k2, k3, p1, p2, fx, fy, cx=128, cy=128):
    height, width = image.shape[:2]
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xs_norm = (xs - cx) / fx
    ys_norm = (ys - cy) / fy
    r = np.sqrt(xs_norm**2 + ys_norm**2)
    factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    xs_distorted = xs_norm * factor + 2 * p1 * xs_norm * ys_norm + p2 * (r**2 + 2 * xs_norm**2)
    ys_distorted = ys_norm * factor + p1 * (r**2 + 2 * ys_norm**2) + 2 * p2 * xs_norm * ys_norm
    x_map = xs_distorted * fx + cx
    y_map = ys_distorted * fy + cy
    # Optionally clip to valid indices:
    x_map = np.clip(x_map, 0, width - 1).astype(np.float32)
    y_map = np.clip(y_map, 0, height - 1).astype(np.float32)
    return x_map, y_map


def distort_image(label_dir, row):
    # Load the image
    original_img = cv2.imread(os.path.join(label_dir, row["label_path"]))

    # Calculate the distortion maps
    x_map, y_map = calc_dist_maps_vectorized(
        original_img,
        k1=row["k1"],
        k2=row["k2"],
        k3=row["k3"],
        p1=row["p1"],
        p2=row["p2"],
        fx=row["fx"],
        fy=row["fx"],
    )

    # Apply the distortion
    distorted_img = cv2.remap(
        original_img, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    return original_img, distorted_img


def main():
    parser = argparse.ArgumentParser(
        description="Distort images using lens distortion parameters from a specified dataset split."
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "eval", "test"],
                        help="Dataset split to convert (train, eval, or test).")
    parser.add_argument("--label_dir", type=str, default="./labels_50k",
                        help="Directory containing label images.")
    parser.add_argument("--dataset_flavor", type=str, default="lens_p1",
                        help="Dataset flavor (e.g. lens_p1 or lens_p2).")
    parser.add_argument("--output_dir", type=str, default="./distorted_images",
                        help="Output directory for distorted images.")
    args = parser.parse_args()

    dataset = load_dataset(
        "mspitzna/physicsgen",
        name=args.dataset_flavor,
        trust_remote_code=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "distorted"), exist_ok=True)

    split = args.split
    if split not in dataset:
        print(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        return

    for idx, example in enumerate(tqdm(dataset[split])):
        original_img, distorted_img = distort_image(args.label_dir, example)
        orig_path = f"original_img_{idx}.jpg"
        dist_path = f"distorted_img_{idx}.jpg"
        cv2.imwrite(os.path.join(args.output_dir, "original", orig_path), original_img)
        cv2.imwrite(os.path.join(args.output_dir, "distorted", dist_path), distorted_img)

    print("Processing complete.")


if __name__ == "__main__":
    main()