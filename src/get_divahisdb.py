import argparse
import inspect
import json
import os
import re
import shutil
import sys
import urllib
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torchvision
from PIL import Image

from datasets.util.dataset_splitter import split_dataset
from util.misc import make_folder_if_not_exists

def download_url(url, file_path, chunk_size=4096):
    r = requests.get(url, allow_redirects=True, stream=True)

    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def extract_zip(zip_path, root, remove_finished=False):
    print('Extracting {}'.format(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root)
    if remove_finished:
        os.unlink(zip_path)


def extract_list_of_classes(files):
    classes = {}
    idx_to_classes = {}
    for i, fname in enumerate(files):
        with open(fname) as f:
            d = json.load(f)
            idx_to_classes[i] = []
            for a in d["annotations"]:
                cls_name = a["name"]
                try:
                    classes[cls_name].add(i)
                except KeyError:
                    classes[cls_name] = set([i])
                if cls_name not in idx_to_classes[i]:
                    idx_to_classes[i].append(cls_name)
    return classes, idx_to_classes


def diva_hisdb(output_folder, **kwargs):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the DIVA HisDB-all dataset for semantic segmentation to the location specified
    on the file system

    See also: https://diuf.unifr.ch/main/hisdoc/diva-hisdb

    Output folder structure: ../HisDB/CB55/train
                             ../HisDB/CB55/val
                             ../HisDB/CB55/test

                             ../HisDB/CB55/test/data -> images
                             ../HisDB/CB55/test/gt   -> pixel-wise annotated ground truth

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # make the root folder
    dataset_root = os.path.join(output_folder, 'HisDB')
    make_folder_if_not_exists(dataset_root)

    # links to HisDB data sets
    link_public = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip')
    link_test_private = urllib.parse.urlparse(
        'https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/private-test/all-privateTest.zip')
    download_path_public = os.path.join(dataset_root, link_public.geturl().rsplit('/', 1)[-1])
    download_path_private = os.path.join(dataset_root, link_test_private.geturl().rsplit('/', 1)[-1])

    # download files
    print('Downloading {}...'.format(link_public.geturl()))
    urllib.request.urlretrieve(link_public.geturl(), download_path_public)

    print('Downloading {}...'.format(link_test_private.geturl()))
    urllib.request.urlretrieve(link_test_private.geturl(), download_path_private)
    print('Download complete. Unpacking files...')

    # unpack relevant folders
    zip_file = zipfile.ZipFile(download_path_public)

    # unpack imgs and gt
    data_gt_zip = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file.namelist() if 'img' in f}
    dataset_folders = [data_file.split('-')[-1][:-4] for data_file in data_gt_zip.keys()]
    for data_file, gt_file in data_gt_zip.items():
        dataset_name = data_file.split('-')[-1][:-4]
        dataset_folder = os.path.join(dataset_root, dataset_name)
        make_folder_if_not_exists(dataset_folder)

        for file in [data_file, gt_file]:
            zip_file.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
                # delete zips
                os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for partition in ['train', 'val', 'test', 'test-public']:
            for folder in ['data', 'gt']:
                make_folder_if_not_exists(os.path.join(dataset_folder, partition, folder))

    # move the files to the correct place
    for folder in dataset_folders:
        for k1, v1 in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            for k2, v2 in {'public-test': 'test-public', 'training': 'train', 'validation': 'val'}.items():
                current_path = os.path.join(dataset_root, folder, k1, k2)
                new_path = os.path.join(dataset_root, folder, v2, v1)
                for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                    shutil.move(os.path.join(current_path, f), os.path.join(new_path, f))
            # remove old folders
            shutil.rmtree(os.path.join(dataset_root, folder, k1))

    # fix naming issue
    for old, new in {'CS18': 'CSG18', 'CS863': 'CSG863'}.items():
        os.rename(os.path.join(dataset_root, old), os.path.join(dataset_root, new))

    # unpack private test folders
    zip_file_private = zipfile.ZipFile(download_path_private)

    data_gt_zip_private = {f: re.sub(r'img', 'pixel-level-gt', f) for f in zip_file_private.namelist() if 'img' in f}

    for data_file, gt_file in data_gt_zip_private.items():
        dataset_name = re.search('-(.*)-', data_file).group(1)
        dataset_folder = os.path.join(dataset_root, dataset_name)

        for file in [data_file, gt_file]:
            zip_file_private.extract(file, dataset_folder)
            with zipfile.ZipFile(os.path.join(dataset_folder, file), "r") as zip_ref:
                zip_ref.extractall(os.path.join(dataset_folder, file[:-4]))
            # delete zip
            os.remove(os.path.join(dataset_folder, file))

        # create folder structure
        for folder in ['data', 'gt']:
            make_folder_if_not_exists(os.path.join(dataset_folder, 'test', folder))

        for old, new in {'pixel-level-gt': 'gt', 'img': 'data'}.items():
            current_path = os.path.join(dataset_folder, "{}-{}-privateTest".format(old, dataset_name), dataset_name)
            new_path = os.path.join(dataset_folder, "test", new)
            for f in [f for f in os.listdir(current_path) if os.path.isfile(os.path.join(current_path, f))]:
                # the ground truth files in the private test set have an additional ending, which needs to be remove
                if new == "gt":
                    f_new = re.sub('_gt', r'', f)
                else:
                    f_new = f
                shutil.move(os.path.join(current_path, f), os.path.join(new_path, f_new))

            # remove old folders
            shutil.rmtree(os.path.dirname(current_path))

    print('Finished. Data set up at {}.'.format(dataset_root))


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=True,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    diva_hisdb(**args.__dict__)
