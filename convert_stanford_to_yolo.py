import scipy.io
import os
from tqdm import tqdm
from PIL import Image
import random
import shutil
import numpy as np

# Paths
car_devkit_path = 'car_devkit'  # Path to your Stanford dataset
dataset_path = 'dataset'        # Output path for formatted dataset

# Load Annotations
train_annot_path = os.path.join(car_devkit_path, 'cars_train_annos.mat')
print(f"Loading annotations from {train_annot_path}")
train_annot = scipy.io.loadmat(train_annot_path)

# Read class names
meta = scipy.io.loadmat(os.path.join(car_devkit_path, 'cars_meta.mat'))
class_names = [c[0] for c in meta['class_names'][0]]

def convert_annotations(annots, split_ratio=0.8):
    # Create folders
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(os.path.join(images_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(images_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

    # Get annotations array
    annotations = annots['annotations'][0]  # Assuming it's a 1D array of annotations
    print(f"Total annotations: {len(annotations)}")
    
    # Shuffle indices for train-val split
    all_indices = list(range(len(annotations)))
    random.shuffle(all_indices)
    split_index = int(len(all_indices) * split_ratio)
    train_indices = all_indices[:split_index]
    val_indices = all_indices[split_index:]

    total_processed = 0
    missing_images = []
    
    # Process train and validation sets
    for mode, indices in [('train', train_indices), ('val', val_indices)]:
        total_labels_created = 0
        
        for idx in tqdm(indices, desc=f'Converting {mode} set'):
            annotation = annotations[idx]
            
            # Extract data from annotation based on the structure from debug output
            try:
                bbox_x1 = float(annotation[0][0][0])
                bbox_y1 = float(annotation[1][0][0])
                bbox_x2 = float(annotation[2][0][0])
                bbox_y2 = float(annotation[3][0][0])
                class_id = int(annotation[4][0][0]) - 1  # Convert to 0-indexed classes
                filename = str(annotation[5][0])
            except Exception as e:
                print(f"Error processing annotation {idx}: {e}")
                print(f"Annotation: {annotation}")
                continue
            
            # Find the source image
            # First, try to find in the standard dataset location 
            # (looks like images might already be in your dataset folder)
            source_paths = [
                os.path.join('dataset', 'images', 'train', filename),  # Already in train folder
                os.path.join('dataset', 'images', filename),           # In images folder
                os.path.join(car_devkit_path, 'cars_train', filename)  # In original dataset
            ]
            
            src_img_path = None
            for path in source_paths:
                if os.path.exists(path):
                    src_img_path = path
                    break
                    
            if not src_img_path:
                missing_images.append(filename)
                if len(missing_images) <= 10:  # Limit the output to avoid spam
                    print(f"⚠️ Missing image: {filename}")
                    print(f"Tried: {source_paths}")
                continue
                
            # Get image dimensions
            try:
                img = Image.open(src_img_path)
                width, height = img.size
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
                continue
                
            # Create label in YOLO format
            center_x = (bbox_x1 + bbox_x2) / 2.0 / width
            center_y = (bbox_y1 + bbox_y2) / 2.0 / height
            bbox_width = (bbox_x2 - bbox_x1) / width
            bbox_height = (bbox_y2 - bbox_y1) / height
            
            # Write YOLO format label
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, mode, label_filename)
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            # Copy image if needed - avoid copying if it's the same file
            dst_img_path = os.path.join(images_dir, mode, filename)
            if src_img_path != dst_img_path:
                try:
                    # Create the directory if it doesn't exist
                    os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                    # Copy the file
                    shutil.copy2(src_img_path, dst_img_path)
                except shutil.SameFileError:
                    # Skip if it's the same file (already in the right place)
                    pass
                except Exception as e:
                    print(f"Error copying file {filename}: {e}")
                    continue
            
            total_labels_created += 1
            total_processed += 1
        
        print(f"\n✅ Total labels created for {mode}: {total_labels_created}")
    
    # Final statistics
    print(f"\n⚠️ Missing images: {len(missing_images)} out of {len(annotations)}")
    print(f"Total processed successfully: {total_processed} out of {len(annotations)}")
    
    # Verify label files were created
    train_labels = os.listdir(os.path.join(labels_dir, 'train'))
    val_labels = os.listdir(os.path.join(labels_dir, 'val'))
    print(f"Labels in train folder: {len(train_labels)}")
    print(f"Labels in val folder: {len(val_labels)}")

if __name__ == "__main__":
    print("Starting Stanford Cars dataset conversion to YOLO format...")
    convert_annotations(train_annot, split_ratio=0.8)