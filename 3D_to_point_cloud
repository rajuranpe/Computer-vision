import cv2  # openCV for computer vision processing
import torch
import numpy as np
import open3d as o3d  # Open3D library for 3D data processing
from torchvision import transforms  # preprocessing and data augmentation in PyTorch
from tqdm import tqdm
from midas.dpt_depth import DPTDepthModel  # Depth model from MiDaS
from midas.transforms import dpt_transform  # Transformation pipeline for MiDaS
from torchvision.transforms import Compose, Resize, Normalize, ToTensor  # PyTorch image transformations

# Specify the type of depth model to load.
model_type = "DPT_Large" # Options: "DPT_Large", "DPT_Hybrid", or "MiDaS_small"

# Load the MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Select the transformation pipeline for the chosen model type;
# DPT_Large and DPT_Hybrid require a specific transform that preserves aspect ratio
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
else:
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Use GPU if available, otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move the MiDaS model to the selected device (GPU/CPU) and set it to evaluation mode.
midas.to(device)
midas.eval()

# print the PyTorch version and whether CUDA (GPU support) is available for debugging purposes
print(torch.__version__)
print(torch.cuda.is_available())

def resize_with_aspect_ratio(image, target_height):
    # Resize the image while maintaining its aspect ratio.
    height, width, _ = image.shape
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image

def midas_transforms(image, target_height=704):  # Increased the target resolution height
    # Prepare the image for MiDaS by resizing and applying normalization
    resized_image = resize_with_aspect_ratio(image, target_height)
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transformed = transform_pipeline(resized_image)
    return transformed, resized_image.shape[:2]

def preprocess_frame(frame):
    # Crop edges and focus on the region of interest (ROI) in the frame
    cropped_frame = crop_edges(frame, margin=50)
    height, width, _ = cropped_frame.shape
    roi_x_start = int(width * 0.2)
    roi_x_end = int(width * 0.8)
    roi_y_start = int(height * 0.1)
    roi_y_end = int(height * 0.9)

    cropped_frame = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Convert the frame to grayscale and equalize the histogram for better contrast.
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)
    processed_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)

    return processed_frame

def midas_depth_estimation(frames, midas, device, target_height=704):  # Targeting higher resolution
    # Perform depth estimation on a batch of frames using MiDaS
    depth_maps = []
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        transformed, resized_shape = midas_transforms(processed_frame, target_height)
        transformed = transformed.unsqueeze(0).to(device)

        with torch.no_grad():
            # Get depth prediction and resize it back to the original dimensions
            prediction = midas(transformed)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=resized_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_maps.append(depth_map)
    return depth_maps

def process_video(video_path, midas, device, skip_frames=1, batch_size=5, focal_length=800, min_depth_diff=0.05):  # Decreased skip_frames, adjusted min_depth_diff
    # Process a video, generating a point cloud from depth estimation
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    point_cloud = []

    frames = []
    previous_depth_map = None

    for i in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        if i % skip_frames == 0:
            frames.append(frame)
            if len(frames) >= batch_size or i == frame_count - 1:
                depth_maps = midas_depth_estimation(frames, midas, device)

                for depth_map in depth_maps:
                    if previous_depth_map is None:
                        previous_depth_map = depth_map
                        continue

                    # Generate 3D points from depth maps, filtering for consistency
                    for v in range(depth_map.shape[0]):
                        for u in range(depth_map.shape[1]):
                            z = depth_map[v, u]
                            if z > 0:
                                prev_z = previous_depth_map[v, u]
                                if abs(z - prev_z) > min_depth_diff:
                                    continue
                                
                                x = (u - depth_map.shape[1] / 2) * z / focal_length
                                y = (v - depth_map.shape[0] / 2) * z / focal_length
                                point_cloud.append([x, y, z])

                    previous_depth_map = depth_map

                frames = []

    cap.release()
    return point_cloud

def scale_point_cloud(point_cloud, known_radius, known_height):
    # Scale the point cloud based on known dimensions for accurate measurements.
    point_cloud = np.array(point_cloud)
    max_x, min_x = np.max(point_cloud[:, 0]), np.min(point_cloud[:, 0])
    max_y, min_y = np.max(point_cloud[:, 1]), np.min(point_cloud[:, 1])
    max_z, min_z = np.max(point_cloud[:, 2]), np.min(point_cloud[:, 2])

    # Calculate observed radius and height.
    observed_radius = max(max_x - min_x, max_y - min_y) / 2
    observed_height = max_z - min_z

    # Calculate scale factors for radius and height.
    scale_factor_radius = known_radius / observed_radius
    scale_factor_height = known_height / observed_height

    # Apply scaling to the point cloud.
    scaled_point_cloud = point_cloud.copy()
    scaled_point_cloud[:, 0] *= scale_factor_radius
    scaled_point_cloud[:, 1] *= scale_factor_radius
    scaled_point_cloud[:, 2] *= scale_factor_height

    return scaled_point_cloud

def save_point_cloud(point_cloud, output_file):
    # save the point cloud as a .ply file using Open3D
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    o3d.io.write_point_cloud(output_file, cloud)

# usage
video_path = r'path/to/input_file'
known_radius = 0.025  # Known radius (for scaling)
known_height = 3.0  # Known height (for scaling)

# Process the video to generate a point cloud
point_cloud = process_video(video_path, midas, device, min_depth_diff=0.05)

if len(point_cloud) > 0:
    # Scale the point cloud based on known dimensions.
    scaled_point_cloud = scale_point_cloud(point_cloud, known_radius, known_height)

    # Save the scaled point cloud to a file
    output_file_point_cloud = r'path/to/output_file'
    save_point_cloud(scaled_point_cloud, output_file_point_cloud)
    
    print(f"Point cloud saved to {output_file_point_cloud}")
else:
    print("No points generated in the point cloud.")

def load_and_visualize_ply(file_path):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Print basic information
    print(f"Point cloud loaded with {len(pcd.points)} points.")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    file_path = r'path/to/output_file'
    load_and_visualize_ply(file_path)
