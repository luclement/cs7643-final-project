import glob
import os
import random

from PIL import Image
import yaml

img_original_dir = "results/test_set_original"
img_style_1_dir = "results/vgg19_Hayao_coco"
img_style_2_dir = "results/vgg16_Hayao_init_epoch_0"

style_1 = os.path.basename(img_style_1_dir)
style_2 = os.path.basename(img_style_2_dir)

# list of images to perform inference on
image_list = [os.path.basename(fp) for fp in glob.glob(os.path.join(img_original_dir, "*.jpg"))]

# dictionary to store image pairs and order
img_pairs = {}

for img_name in image_list:
    img_original = Image.open(os.path.join(img_original_dir, img_name))
    img_style_1 = Image.open(os.path.join(img_style_1_dir, img_name))
    img_style_2 = Image.open(os.path.join(img_style_2_dir, img_name))

    os.makedirs(os.path.join("results", "labelbox", f"{style_2}_vs_VGG19"), exist_ok=True)

    img_composite = Image.new('RGB', (2 * img_original.size[0], 2 * img_original.size[1]))
    img_composite.paste(img_original, (img_composite.size[0] // 4, 0))

    flip = random.uniform(0, 1)
    if (flip >= 0.5):
        img_composite.paste(img_style_2, (0, img_original.size[1]))
        img_composite.paste(img_style_1, (img_original.size[0], img_original.size[1]))
        new_img_name = f"{img_name}_[{style_2}]_[{style_1}].jpg"
        img_pairs[new_img_name] = [os.path.join(img_style_2_dir, img_name), os.path.join(img_style_1_dir, img_name)]
    else:
        img_composite.paste(img_style_1, (0, img_original.size[1]))
        img_composite.paste(img_style_2, (img_original.size[0], img_original.size[1]))
        new_img_name = f"{img_name}_[{style_1}]_[{style_2}].jpg"
        img_pairs[new_img_name] = [os.path.join(img_style_1_dir, img_name), os.path.join(img_style_2_dir, img_name)]
        
    img_composite.save(os.path.join("results", "labelbox", f"{style_2}_vs_VGG19", new_img_name))
    with open(os.path.join("results", "labelbox", f"{style_2}_vs_VGG19", "image_pairs.yaml"), 'w') as outfile:
        yaml.dump(img_pairs, outfile, default_flow_style=False)
    
    
    