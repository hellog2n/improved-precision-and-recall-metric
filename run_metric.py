# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Script to run StyleGAN truncation sweep or evaluate realism score of StyleGAN samples."""

import argparse
import os
import tensorflow as tf
import hashlib
import cv2
import dnnlib
import numpy as np
# from dnnlib.util import Logger
# from ffhq_datareader import load_dataset
from experiments import compute_stylegan_realism
from experiments import compute_Precision_and_Recall
# from utils import init_tf
import experiments
import torchvision.transforms as transforms
import pathlib
import torch
from PIL import Image

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

SAVE_PATH = os.path.dirname(__file__)

# ----------------------------------------------------------------------------
# Configs for truncation sweep and realism score.

realism_config = dnnlib.EasyDict(minibatch_size=8, num_images=50000, num_gen_images=1000, show_n_images=64,
                                 truncation=1.0, save_images=True, save_path=SAVE_PATH, num_gpus=1,
                                 random_seed=123456)


# truncation_config = dnnlib.EasyDict(minibatch_size=8, num_images=50000, truncations=[1.0, 0.7, 0.3],
#   save_txt=True, save_path=SAVE_PATH, num_gpus=1, random_seed=1234)

# ----------------------------------------------------------------------------
# Minimal CLI.

def parse_command_line_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Improved Precision and Recall Metric for Assessing Generative Models.',
        epilog='This script can be used to reproduce StyleGAN truncation sweep (Fig. 4) and' \
               ' computing realism score for StyleGAN samples (Fig. 11).')

    # 참고하려는 이미지가 포함되어 있는 Directory Name (필수)
    parser.add_argument('--save_path', type=str, required=False,
                        help='텍스트 파일이 저장되는 경로')

    parser.add_argument(
        '-d',
        '--data_dir',
        type=str,
        required=False,
        help='Absolute path to TFRecords directory.'
    )
    parser.add_argument(
        '-t',
        '--precision_and_recall',
        action='store_true',
        help='Calculate Improved Precision and Recall.'
    )
    parser.add_argument(
        '-r',
        '--realism_score',
        action='store_true',
        help='Calculate realism score for StyleGAN samples. Replicates Fig. 11 from Appendix.'
    )
    # 참고하려는 이미지가 포함되어 있는 Directory Name (필수)
    parser.add_argument('--reference_dir', type=str, required=True,
                        help='directory containing reference images')
    # 평가하려는 이미지가 포함되어 있는 Directory Name (필수), nargs가 +인 경우, 1개 이상의 값을 전부 받아들인다.  *인 경우, 0개 이상의 값을 전부 받아들인다.
    parser.add_argument('--eval_dirs', type=str, required=True,
                        help='directory or directories containing images to be '
                             'evaluated')
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args


# ----------------------------------------------------------------------------
# inceptionV3 모델의 pooling 계층을 이용하여 이미지의 feature를 뽑는다.
def generate_embedding(imgs, device):
    return experiments.save_embedding_Files(fileName='vis/inception.tsv', directory = imgs, device=device)


# VGG16을 통해서 임베딩을 한다.
def load_or_generate_embedding(directory, device):
    # hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    # path = os.path.join(cache_dir, hash + '.npy')

    # 디렉토리로부터 이미지를 갖고온다.
    imgs = load_images_from_dir(directory)
    embeddings = generate_embedding(imgs, device=device)
    return embeddings


# 디렉토리로부터 이미지를 갖고온다.
def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    directory = pathlib.Path(directory)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in directory.glob('*.{}'.format(ext))])
    dataset = ImagePathDataset(files, transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8)
    print('dataloader', len(dataloader))
    return dataloader


def main(args=None):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    # Parse command line arguments.
    parsed_args = parse_command_line_arguments(args)

    # Initialize logger.
    # Logger()

    # Initialize dataset object.
    # init_tf()
    # dataset_obj = load_dataset(tfrecord_dir=parsed_args.data_dir, repeat=True, shuffle_mb=0,
    # prefetch_mb=100, max_label_size='full', verbose=True)

    # ref 폴더 경로와 eval 폴더 경로의 절대 경로를 얻는다.
    reference_dir = os.path.abspath(parsed_args.reference_dir)
    eval_dirs = os.path.abspath(parsed_args.eval_dirs)
    save_path = parsed_args.save_path
    ref_features = load_or_generate_embedding(reference_dir, device)
    eval_features = load_or_generate_embedding(eval_dirs, device)
    if save_path:
        save_txt = True
    if parsed_args.realism_score:  # Compute realism score.
        realism_config.datareader = dataset_obj
        compute_stylegan_realism(**realism_config)


    nearest_k_list = [3, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if parsed_args.precision_and_recall:  # Compute truncation sweep.
        # truncation_config.datareader = dataset_obj
        for nearest_k in nearest_k_list:
            compute_Precision_and_Recall(ref_features, eval_features, save_txt  = save_txt, save_path = save_path, nhood_sizes=[nearest_k])
            print('-'*10)
    #peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
    #peak_gpu_mem_usage = peak_gpu_mem_op.eval()
    #print('Peak GPU memory usage: %g GB' % (peak_gpu_mem_usage * 1e-9))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()

# ----------------------------------------------------------------------------
