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
from experiments import compute_stylegan_truncation
# from utils import init_tf
import experiments

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
        '--truncation_sweep',
        action='store_true',
        help='Calculate StyleGAN truncation sweep. Replicates Fig. 4 from the paper.'
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
def generate_embedding(directory):
    return experiments.save_embedding_Files(directory = directory, fileName = 'vis/inception.tsv')


# VGG16을 통해서 임베딩을 한다.
def load_or_generate_embedding(directory):
    # hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    # path = os.path.join(cache_dir, hash + '.npy')

    # 디렉토리로부터 이미지를 갖고온다.
    imgs = load_images_from_dir(directory)
    embeddings = generate_embedding(directory)
    return embeddings


# 디렉토리로부터 이미지를 갖고온다.
def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    paths = [os.path.join(directory, fn) for fn in os.listdir(directory)
             if os.path.splitext(fn)[-1][1:] in types]
    # images are in [0, 255]
    imgs = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for path in paths]
    return np.array(imgs)


def main(args=None):
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
    ref_features = load_or_generate_embedding(reference_dir)
    eval_features = load_or_generate_embedding(eval_dirs)
    if save_path:
        save_txt = True
    if parsed_args.realism_score:  # Compute realism score.
        realism_config.datareader = dataset_obj
        compute_stylegan_realism(**realism_config)

    if parsed_args.truncation_sweep:  # Compute truncation sweep.
        # truncation_config.datareader = dataset_obj
        compute_stylegan_truncation(ref_features, eval_features, save_txt  = save_txt, save_path = save_path)

    #peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
    #peak_gpu_mem_usage = peak_gpu_mem_op.eval()
    #print('Peak GPU memory usage: %g GB' % (peak_gpu_mem_usage * 1e-9))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()

# ----------------------------------------------------------------------------
