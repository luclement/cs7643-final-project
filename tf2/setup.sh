touch __init__.py
touch tools/__init__.py

echo -e "\n##### Downloading vgg19 #####"
wget -nc --directory-prefix=vgg19_weight/ https://github.com/TachibanaYoshino/AnimeGAN/releases/download/vgg16%2F19.npy/vgg19.npy

echo -e "\n##### Downloading dataset #####"
wget -nc https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip
unzip -q -n dataset.zip -d ./dataset

echo -e "\n##### Preprocessing data #####"
python -m tools.edge_smooth --dataset "Hayao" --img_size 256
python -m tools.data_mean --dataset "Hayao"
