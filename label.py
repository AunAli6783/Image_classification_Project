
#this file created all the class names that are in this dataset check the created file class_name.txt

import scipy.io

# Specify the path to your .mat file
mat_file_path = r"C:\Users\buzz 2\OneDrive\Desktop\AI_Project\car_devkit\cars_meta.mat"

# Load the .mat file
mat = scipy.io.loadmat(mat_file_path)

# Extract class names
class_names = [name[0] for name in mat['class_names'][0]]

# Print the class names
print("Number of classes:", len(class_names))
for i, name in enumerate(class_names):
    print(f"Class {i}: {name}")

# Optionally, save class names to a file for use in YAML
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(f"- {name}\n")