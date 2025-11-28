import os
import json

dataset_dir = "dataset/dataset"  # or your actual training folder

class_names = sorted(os.listdir(dataset_dir))
class_indices = {name: i for i, name in enumerate(class_names)}

with open("class_names.json", "w") as f:
    json.dump(class_indices, f)
