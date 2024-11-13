"""Script for calculating precision, recall, density and coverage."""

import os
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset

import os
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch
import pathlib

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

import sklearn.metrics


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, realism=False):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    # dist.print0('Num real: {} Num fake: {}'.format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    d = dict(precision=precision, recall=recall,
                density=density, coverage=coverage)

    if realism:
        """
        Large errors, even if they are rare, would undermine the usefulness of the metric.
        We tackle this problem by discarding half of the hyperspheres with the largest radii.
        In other words, the maximum in Equation 3 is not taken over all φr ∈ Φr but only over 
        those φr whose associated hypersphere is smaller than the median.
        """
        mask = real_nearest_neighbour_distances < np.median(real_nearest_neighbour_distances)

        d['realism'] = (
                np.expand_dims(real_nearest_neighbour_distances[mask], axis=1)/distance_real_fake[mask]
        ).max(axis=0)

    return d

def get_representations(model, DataLoader, device, normalized=False):
    """Extracts features from all images in DataLoader given model.

    Params:
    -- model       : Instance of Encoder such as inception or CLIP or dinov2
    -- DataLoader  : DataLoader containing image files, or torchvision.dataset

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    start_idx = 0

    # for ibatch, batch in enumerate(tqdm(DataLoader.data_loader)):
    for ibatch, batch in enumerate(tqdm(DataLoader)):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        if not torch.is_tensor(batch):
            # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
            batch = batch['pixel_values']
            batch = batch[:,0]

        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        with torch.no_grad():
            # pred = model(batch)
            pred = model(batch, return_features=True).to(torch.float64)
            if not torch.is_tensor(pred): # Some encoders output tuples or lists
                pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        if normalized:
            pred = torch.nn.functional.normalize(pred, dim=-1)
        pred = pred.cpu().numpy()

        if ibatch == 0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            # pred_arr = np.empty((DataLoader.nimages, dims))
            pred_arr = np.empty((10000, dims))

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def save_outputs(out_path, reps):
    """Save representations and other info to disk at file_path"""
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    np.savez(os.path.join(out_path, 'reps.npz'), reps=reps)

def load_reps_from_path(save_path='./outputs', file_name="reps.npz"):
    """Save representations and other info to disk at file_path"""
    reps = None
    if os.path.exists(os.path.join(save_path, file_name)):
        dist.print0(f'Loading representations from {os.path.join(save_path, file_name)}...')
        saved_file = np.load(os.path.join(save_path, file_name))
        reps = saved_file['reps']
    return reps

#----------------------------------------------------------------------------

def compute_representations(DL, model, device, save=False, load=False):
    if load:
        repsi = load_reps_from_path()
        if repsi is not None: 
            return repsi

    repsi = get_representations(model, DL, device, normalized=False)

    if save:
        dist.print0(f'Saving representations to "./outputs"')
        save_outputs('./outputs', repsi)

    return repsi

def compute_scores(metrics, reps, labels=None):

    scores = {}

    if 'prdc' in metrics:
        dist.print0("Computing precision, recall, density, and coverage")
        reduced_n = min(10000, reps[0].shape[0], reps[1].shape[0])
        inds0 = np.random.choice(reps[0].shape[0], reduced_n, replace=False)
        inds1 = np.arange(reps[1].shape[0])

        if 'realism' not in metrics:
            # Realism is returned for each sample, so do not shuffle if this metric is desired.
            # Else filenames and realism scores will not align
            inds1 = np.random.choice(inds1, min(inds1.shape[0], reduced_n), replace=False)

        prdc_dict = compute_prdc(
            reps[0][inds0], 
            reps[1][inds1], 
            nearest_k=5,
            realism=True if 'realism' in metrics else False)
        scores = dict(scores, **prdc_dict)

    for key, value in scores.items():
        if key == 'realism': continue
        dist.print0(f'{key}: {value:.5f}')

    return scores

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate precision, recall, density and coverage.
    python prdc.py calc --images=path/to/images --images_ref=path/to/reference/images
    torchrun --standalone --nproc_per_node=1 prdc.py calc --images=path/to/images --images_ref=path/to/reference/images
    """
    
#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path',   help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--images_ref', 'ref_path', help='Path to the reference images', metavar='PATH|ZIP',    type=str, required=True)  # default="/wangcan/cdf/zyu/data/COCO/val2017/val2017"
@click.option('--seed',                   help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                  help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--desc',                   help='A description string', metavar='str',                 type=str)

def calc(image_path, ref_path, seed, batch, desc=None, device=torch.device('cuda')):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        model = pickle.load(f).to(device)

    # Loarding generated images
    dist.print0(f'Loading generated images from "{image_path}"...')
    dataset_gen = dataset.ImageFolderDataset(path=image_path, max_size=5000, random_seed=seed)
    assert len(dataset_gen) == 5000
    
    # Loarding dataset
    dist.print0(f'Loading reference images from "{ref_path}..."')
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=ref_path, max_size=5000, use_labels=False, xflip=False, cache=True)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    dataset_test = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(f'Computing representations...')
    num_batches = ((len(dataset_gen) - 1) // (batch * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_gen)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=rank_batches, **data_loader_kwargs)
    dataloader_gen = torch.utils.data.DataLoader(dataset_gen, batch_sampler=rank_batches, **data_loader_kwargs)
    reps_test = compute_representations(dataloader_test, model, device, save=True, load=False)
    reps_gen = compute_representations(dataloader_gen, model, device)
    reps = [reps_test, reps_gen]
    
    dist.print0(f'Computing scores...')
    scores = compute_scores(['prdc'], reps, labels=None)

    torch.distributed.barrier()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
