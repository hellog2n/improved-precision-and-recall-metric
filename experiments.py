# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to compute realism score and StyleGAN truncation sweep."""

import numpy as np
import os
import PIL.Image
from time import time
import tensorflow as tf
# import dnnlib
from precision_recall import DistanceBlock
from precision_recall import knn_precision_recall_features
from precision_recall import ManifoldEstimator
# from utils import initialize_feature_extractor
# from utils import initialize_stylegan
from skimage.transform import resize
from numpy import asarray
from tensorflow.python.keras.applications.vgg16 import VGG16


# ----------------------------------------------------------------------------
# Helper functions.

def save_image(img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t.astype(np.uint8), 'RGB').save(filename)


def generate_single_image(Gs, latent, truncation, fmt):
    gen_image = Gs.run(latent, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True,
                       output_transform=fmt)
    gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
    return gen_image


# ----------------------------------------------------------------------------
def scale_images_GPU(images, new_shape):
    with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        images_list = list()
        for image in images:
            new_image = resize(image, new_shape, 0)
            images_list.append(new_image)
        return asarray(images_list)


# from tensorflow.python.keras.applications.vgg16 import VGG16
size = 32
vgg_model = VGG16(include_top=False, pooling='avg', input_shape=(size, size, 3))

def get_features(inputs, model):
    """Compose the preprocess_for_inception function with TFGAN run_inception."""
    inputs = scale_images_GPU(inputs, (size, size, 3))
    inputs = tf.keras.applications.vgg16.preprocess_input(inputs)
    return model.predict(inputs)


def embed_images_in_VGG16(imgs, batch_size=32):
    model = Sequential()
    for layer in vgg_model.layers[:-1]:
        model.add(layer)
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False
    model.summary()

    # 이미지를 담을 input_tensor를 선언한다.
    graph_def = tf.compat.v1.GraphDef()
    embeddings = []
    i = 0
    while i < len(imgs):
        input_tensor = imgs[i:i + batch_size]
        feature_tensor = get_features(input_tensor, model=model)
        embeddings.append(feature_tensor)
        i += batch_size
    # 해당 경로에서 inception graph를 갖고온다.
    """Get a GraphDef proto from a disk location."""
    return np.concatenate(embeddings, axis=0)


def compute_stylegan_truncation(ref_features, eval_features, minibatch_size=32, num_images=100, truncations=1.0,
                                num_gpus=1, save_txt=None, save_path=None):
    """StyleGAN truncation sweep. (Fig. 4)

        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            truncations (list): List of truncation psi values.
            save_txt (string): Name of result file.
            save_path (string): Absolute path to directory where result textfile is saved.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.

    """
    print('Running StyleGAN truncation sweep...')
    it_start = time()
    metric_results = np.zeros([1, 3], dtype=np.float32)
    ref_features = ref_features
    eval_features = eval_features

    # Calculate k-NN precision and recall.
    state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus, nhood_sizes=[3], row_batch_size=25000, col_batch_size=50000)

    # Store results.
    metric_results[0, 0] = 0.0
    metric_results[0, 1] = state['precision'][0]
    metric_results[0, 2] = state['recall'][0]

    # Print progress.
    print('Precision: %0.3f' % state['precision'][0])
    print('Recall: %0.3f' % state['recall'][0])
    print('Iteration time: %gs\n' % (time() - it_start))

    # Save results.
    if save_txt:
        result_path = save_path
        result_file = os.path.join(result_path, 'Result.txt')
        header = 'Header,precision,recall'
        np.savetxt(result_file, metric_results, header=header,
                   delimiter=',', comments='')


# ----------------------------------------------------------------------------

def compute_stylegan_realism(datareader, minibatch_size, num_images, num_gen_images,
                             show_n_images, truncation, num_gpus, random_seed,
                             save_images=False, save_path=None):
    """Calculate realism score for StyleGAN samples. (Fig. 11)

        Args:
            datareader (): FFHQ datareader object.
            minibatch_size (int): Minibatch size.
            num_images (int): Number of images used to evaluate precision and recall.
            num_gen_images (int): Number of generated images where low and high quality
                samples are selected.
            show_n_images (int): Number of low and high quality samples selected.
            truncation (float): Amount of truncation applied to StyleGAN.
            num_gpus (int): Number of GPUs used.
            random_seed (int): Random seed.
            save_images (bool): Save images.
            save_path (string): Absolute path to directory where result textfile is saved.

    """
    print('Running StyleGAN realism...')
    rnd = np.random.RandomState(random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    # Read real images.
    print('Reading real images...')
    real_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        real_batch, _ = datareader.get_minibatch_np(end - begin)
        real_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)

    # Estimate manifold of real images.
    print('Estimating manifold of real images...')
    distance_block = DistanceBlock(feature_net.output_shape[1], num_gpus)
    real_manifold = ManifoldEstimator(distance_block, real_features, clamp_to_percentile=50)

    # Generate images.
    print('Generating images...')
    latents = np.zeros([num_gen_images, Gs.input_shape[1]], dtype=np.float32)
    fake_features = np.zeros([num_gen_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_gen_images, minibatch_size):
        end = min(begin + minibatch_size, num_gen_images)
        latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])
        gen_images = Gs.run(latent_batch, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True,
                            output_transform=fmt)
        fake_features[begin:end] = feature_net.run(gen_images, num_gpus=num_gpus, assume_frozen=True)
        latents[begin:end] = latent_batch

    # Estimate quality of individual samples.
    _, realism_scores = real_manifold.evaluate(fake_features, return_realism=True)

    if save_images and save_path is not None:
        result_dir = os.path.join(save_path, 'stylegan_realism', 'truncation%0.2f' % truncation)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    # Save samples with lowest and highest realism.
    num_saved = show_n_images

    # Sort realism scores.
    highest_realism_idx = realism_scores.argsort()[-num_saved:][::-1]
    lowest_realism_idx = realism_scores.argsort()[:num_saved]

    print('Saving %i low and high quality samples...' % num_saved)
    for i in range(num_saved):
        low_idx = lowest_realism_idx[i]
        high_idx = highest_realism_idx[i]

        # Get corresponding latents.
        low_quality_latent = latents[low_idx]
        high_quality_latent = latents[high_idx]

        # Generate images.
        low_quality_img = generate_single_image(Gs, low_quality_latent[None, :], truncation, fmt)[0]
        high_quality_img = generate_single_image(Gs, high_quality_latent[None, :], truncation, fmt)[0]

        if save_images:
            low_realism_score = realism_scores[low_idx]
            high_realism_score = realism_scores[high_idx]
            save_image(low_quality_img, os.path.join(result_dir, 'low_realism_%f_%i.png' % (low_realism_score, i)))
            save_image(high_quality_img, os.path.join(result_dir, 'high_realism_%f_%i.png' % (high_realism_score, i)))
        else:
            low_quality_img.show()
            high_quality_img.show()

    print('Done evaluating StyleGAN realism.\n')

# ----------------------------------------------------------------------------
