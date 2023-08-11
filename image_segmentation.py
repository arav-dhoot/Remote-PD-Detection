import os
import torch
import torchvision.transforms as T
import torchvision.models.segmentation as seg
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import numpy as np
import cv2

source_path = '/frames'
destination_path = '/segmented_images'

model = seg.deeplabv3_resnet50(pretrained=True, progress=True)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def remove_secondary_object(segmentation_mask):
    labeled_array, num_features = ndimage.label(segmentation_mask)
    largest_segment = 0
    max_area = 0
    for feature in range(1, num_features + 1):
        area = np.sum(labeled_array == feature)
        if area > max_area:
            max_area = area
            largest_segment = feature
    main_object_mask = labeled_array == largest_segment
    return main_object_mask.astype(np.uint8)

def full_segmentation(image_path, destination_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
    main_object_mask = remove_secondary_object(output_predictions)
    print("Output predictions shape:", output_predictions.shape)
    print("Main object mask shape:", main_object_mask.shape)
    main_object_mask_binary = (main_object_mask * 255).astype(np.uint8)
    cv2.imwrite(destination_path, main_object_mask_binary)

os.makedirs(destination_path, exist_ok=True)
for coded_folder in os.listdir(source_path):
    if not os.path.isdir(os.path.join(source_path, coded_folder)): os.remove(os.path.join(source_path, coded_folder))
    os.makedirs(os.path.join(destination_path, coded_folder), exist_ok=True)
    for severity in os.listdir(os.path.join(source_path, coded_folder)):
        if not os.path.isdir(os.path.join(source_path, coded_folder, severity)): os.remove(os.path.join(source_path, coded_folder, severity))
        os.makedirs(os.path.join(destination_path, coded_folder, severity), exist_ok=True)
        for video in os.listdir(os.path.join(source_path, coded_folder, severity)):
            if not os.path.isdir(os.path.join(source_path, coded_folder, severity, video)): os.remove(os.path.join(source_path, coded_folder, severity, video))
            os.makedirs(os.path.join(destination_path, coded_folder, severity, video), exist_ok=True)
            for image in os.listdir(os.path.join(source_path, coded_folder, severity, video)):
                full_segmentation(os.path.join(source_path, coded_folder, severity, video, image), os.path.join(destination_path, coded_folder, severity, video, image))