import numpy as np
import sys
import os
import argparse
import json
import tqdm
sys.path.insert(0, "/root/cleanlab")
from cleanlab.segmentation.filter import find_label_issues
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores
from cleanlab.segmentation.summary import display_issues, common_label_issues, filter_by_class
np.set_printoptions(suppress=True)

# 定义 mapillary_classes 和 palette
mapillary_classes = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
                    'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking',
                    'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane',
                    'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist',
                    'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
                    'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
                    'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack',
                    'Billboard', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant',
                    'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
                    'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole',
                    'Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)',
                    'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',
                    'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
                    'Wheeled Slow', 'Car Mount', 'Ego Vehicle']
palette = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
            [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
            [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
            [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
            [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
            [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
            [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
            [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
            [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
            [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
            [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
            [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
            [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
            [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
            [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
            [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10]]

# Argument parser to get file paths from command line
parser = argparse.ArgumentParser(description='Process some file paths.')
parser.add_argument('pred_probs_filepaths', type=str, help='Path to the prediction probabilities file')
parser.add_argument('label_filepaths', type=str, help='Path to the label file')
parser.add_argument('base_name', type=str, help='Base name for JSON key')
parser.add_argument('output_filepath', type=str, help='Path to save the image scores')
parser.add_argument('dict_prefix', type=str, help='dict prefix')
parser.add_argument('vis', type=str, help='Visualize issues or not')
parser.add_argument('vis_threshold', type=float, help='Visualize issues or not')

args = parser.parse_args()

# Load the files
pred_probs = np.load(args.pred_probs_filepaths, mmap_mode='r+')
labels = np.load(args.label_filepaths, mmap_mode='r+')

# Finding label issues
# issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=32, batch_size=100000)

# Getting label quality scores
image_scores, pixel_scores = get_label_quality_scores(labels, pred_probs, n_jobs=32, batch_size=100000)
print(image_scores)

issue_from_score = issues_from_scores(image_scores, pixel_scores, threshold=0.5)
if args.vis == "True":
    fig_paths = []
    output_dir = "%s%s" % (args.output_filepath, args.base_name)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(image_scores)):
        if image_scores[i] < args.vis_threshold:
            fig_paths.append(display_issues(np.expand_dims(issue_from_score[i], axis=0), labels=np.expand_dims(labels[i], axis=0), pred_probs=np.expand_dims(pred_probs[i], axis=0), class_names=mapillary_classes, output_dir=output_dir, palette=palette))

json_filepath = "%s%s.json" % (args.output_filepath, args.dict_prefix)
# Read the existing JSON file
with open(json_filepath, 'r') as f:
    data = json.load(f)

# Add the image scores to the JSON data
res_base_name = "_".join(args.base_name.split("_")[1:][::-1])
print(res_base_name)
if res_base_name in data:
    count = 0
    for i, score in enumerate(image_scores):
        if score < args.vis_threshold:
            data[res_base_name][i]['vis_path'] = fig_paths[count]
            count += 1
        data[res_base_name][i]['image_score'] = score
else:
    print(f"Base name {res_base_name} not found in JSON file.")

# Save the modified JSON data back to file
with open(json_filepath, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Image scores added to {json_filepath}")

# 删除 .npy 文件
if os.path.exists(args.pred_probs_filepaths):
    os.remove(args.pred_probs_filepaths)
    print(f"Deleted file: {args.pred_probs_filepaths}")

if os.path.exists(args.label_filepaths):
    os.remove(args.label_filepaths)
    print(f"Deleted file: {args.label_filepaths}")
