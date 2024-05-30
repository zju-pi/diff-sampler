import os
import urllib.request
from tqdm import tqdm
import zipfile

urls = {
    "cifar10": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl",
    "ffhq": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl",
    "afhqv2": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl",
    "imagenet64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
    "lsun_bedroom": "https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt",
    "imagenet256": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
    "imagenet256-classifier": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt",
    "lsun_bedroom_ldm": "https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip",
    "ffhq_ldm": "https://ommer-lab.com/files/latent-diffusion/ffhq.zip",
    "vq-f4": "https://ommer-lab.com/files/latent-diffusion/vq-f4.zip",
    "ms_coco": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",
    "prompts": "https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv",
}

def check_file_by_key(key, target_dir="./src"):
    if key not in urls:
        raise ValueError(f"Unknown key: {key}")
    
    target_dir = os.path.join(target_dir, key)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    url = urls[key]
    target_path = os.path.join(target_dir, url.split("/")[-1])

    if target_path.endswith(".zip"):        # for lsun_bedroom_ldm and ffhq_ldm
        model_path = os.path.join(target_dir, 'model.ckpt')
        if not os.path.exists(model_path):
            print(f'File does not exist, downloading from {url}')
            download_with_url(url, target_path)
            try:
                unzip_file(target_path, target_dir)
                os.remove(target_path)
            except:
                raise ValueError(f"Fail to unzip the file: {model_path}")
        else:
            print(f'File already exists: {model_path}')
        target_path = model_path
    else:
        if not os.path.exists(target_path):
            print(f'File does not exist, downloading from {url}')
            download_with_url(url, target_path)
        else:
            print(f'File already exists: {target_path}')
    
    tmp_path = None
    if key == "imagenet256":    # check the classifier
        url = urls["imagenet256-classifier"]
        tmp_path = os.path.join(target_dir, url.split("/")[-1])
        if not os.path.exists(tmp_path):
            print(f'The classifier does not exist, downloading from {url}')
            download_with_url(url, tmp_path)
        else:
            print(f'The classifier already exists: {tmp_path}')
    elif key in ["lsun_bedroom_ldm", "ffhq_ldm"]:    # check the vq_f4 model
        url = urls["vq-f4"]
        target_dir = "./models/ldm_models/first_stage_models/vq-f4"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        tmp_path = os.path.join(target_dir, url.split("/")[-1])
        model_path = os.path.join(target_dir, 'model.ckpt')
        if not os.path.exists(model_path):
            print(f'The vq-f4 model (model.ckpt) does not exist, downloading from {url}')
            download_with_url(url, tmp_path)
            try:
                unzip_file(tmp_path, target_dir)
                os.remove(tmp_path)
            except:
                raise ValueError(f"Fail to unzip the file: {tmp_path}")
        else:
            print(f'The vq-f4 model already exists: {tmp_path}')
        tmp_path = model_path

    return target_path, tmp_path

def download_with_url(url, target_path):
    req = urllib.request.urlopen(url)
    total_size = int(req.getheader('Content-Length').strip())
    with open(target_path, 'wb') as file, tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=total_size, desc=target_path) as bar:
        urllib.request.urlretrieve(url, target_path, reporthook=lambda block_num, block_size, total_size: bar.update(block_size))

def unzip_file(file_path, target_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)