import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
from skimage.filters import gaussian
import torch
import torch.utils.data as udata
from utils.utils import data_augmentation, normalize, contrast_reduction_
from pathlib import Path

def img_to_patches(img, win, stride=1):
    r"""Converts an image to an array of patches.

    Args:
        img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
            image
        win: size of the output patches
        stride: int. stride
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])
def prepare_data_bis(data_paths, \
        val_data_path, \
        max_num_patches=None, \
        filenames = ['h5files/train_color.h5','h5files/val_color.h5'],\
        gray_mode=False):
    # training database
    print('> Training database')
    types = ('*.bmp', '*.png')

    files_list = []
    ordered = True
    for data_path in data_paths:
        files = []
        for tp in types:
            if ordered :
                files.extend(glob.glob(os.path.join(data_path, tp)))
            else :
                files_list.extend(glob.glob(os.path.join(data_path, tp)))
        if ordered:
            files.sort()
            files_list.append(files)
    if not ordered:
        files_list = [list(np.random.permutation(files_list))]
        print(len(files_list[0]))
    if gray_mode:
        traindbf = 'h5files/train/'+filenames[0]
        valdbf = 'h5files/train/'+filenames[1]
    else:
        traindbf = 'h5files/train/'+filenames[0]
        valdbf = 'h5files/train/'+filenames[1]

    if max_num_patches is None:
        max_num_patches = 500000
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))

    with h5py.File(traindbf, 'w') as h5f:
        train_num2 = 0
        for k  in range(len(files_list)):
            train_num = 0
            files = files_list[k]
            i = 0
            while i < len(files) and train_num2 < max_num_patches:
                imgor = cv2.imread(files[i])
                # h, w, c = img.shape
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(imgor, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
                img = normalize(img)

                h5f.create_dataset(str(train_num2), data=img)
                train_num += 1
                train_num2+=1
                i += 1
            print(train_num)

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num2)
    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in enumerate(files):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
            img =  normalize(img)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()


def prepare_data(data_paths, \
        val_data_path, \
        patch_size, \
        strides, \
        scale_numbers,\
        max_num_patches=None, \
        filenames = ['train/','val/'], \
        aug_times=1, \
        gray_mode=False, \
        mask = 0):
    r"""Builds the training and validations datasets by scanning the
    corresponding directories for images and extracting	patches from them.

    Args:
        data_path: path containing the training image dataset
        val_data_path: path containing the validation image dataset
        patch_size: size of the patches to extract from the images
        stride: size of stride to extract patches
        stride: size of stride to extract patches
        filenames : the filenames to which we save the different datasets
        max_num_patches: maximum number of patches to extract
        aug_times: number of times to augment the available data minus one
        gray_mode: build the databases composed of grayscale patches
    """
    # training database
    print('> Training database')
    if scale_numbers == 1:
        scales = [1]
    elif scale_numbers == 2:
        scales = [1, 0.6]
    elif scale_numbers ==3:
        scales = [1, 0.8,0.65]
    # scales = [1]
    types = ('*.bmp', '*.png')

    files_list = []
    ordered = True
    for data_path in data_paths:
        files = []
        for tp in types:
            if ordered :
                files.extend(glob.glob(os.path.join(data_path, tp)))
            else :
                files_list.extend(glob.glob(os.path.join(data_path, tp)))
        if ordered:
            files.sort()
            files_list.append(files)
    if not ordered:
        files_list = [list(np.random.permutation(files_list))]
        print(len(files_list[0]))
   
    if not(os.path.isdir("datasets/h5files/"+filenames[0])):
        os.makedirs("datasets/h5files/"+filenames[0])
    if not(os.path.isdir("datasets/h5files/"+filenames[1])):
        os.makedirs("datasets/h5files/"+filenames[1])
    traindbf = os.path.join("datasets/h5files/"+filenames[0], "train.h5")
    valdbf = os.path.join("datasets/h5files/"+filenames[1], "val.h5")
    if max_num_patches is None:
        max_num_patches = 500000
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))

    with h5py.File(traindbf, 'w') as h5f:
        train_num2 = 0
        for k  in range(len(files_list)):
            print(k)
            stride = strides[k]
            train_num = 0
            files = files_list[k]
            i = 0
            while i < len(files) and train_num < max_num_patches:
                print(files[i])
                imgor = cv2.imread(files[i])
                
                # h, w, c = img.shape
                if not(imgor is None):
                    for sca in scales:
                        img = imgor
                        if sca <1:
                            img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                            interpolation=cv2.INTER_CUBIC)
                        # img = imgor
                        if not gray_mode:
                            # CxHxW RGB image
                            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                        else:
                            # CxHxW grayscale image (C=1)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = np.expand_dims(img, 0)
                        # img = normalize(img)
                        patches = img_to_patches(img, win=patch_size, stride=stride)
                        print("\tfile: %s scale %.1f # samples: %d" % \
                            (files[i], sca, patches.shape[3]*aug_times))
                        for nx in range(patches.shape[3]):
                            data = np.uint8(np.clip(data_augmentation(patches[:, :, :, nx].copy(), \
                                    np.random.randint(0, 7)),0,255))

                            h5f.create_dataset(str(train_num2), data=data)
                            train_num += 1
                            train_num2 +=1
                            for mx in range(aug_times-1):
                                data_aug = data_augmentation(data, np.random.randint(1, 4))
                                h5f.create_dataset(str(train_num2)+"_aug_%d" % (mx+1), data=data_aug)
                                train_num += 1
                                train_num2 +=1
                i += 1
            print(train_num)

    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in enumerate(files):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
            
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
            
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num2)
    print('\tvalidation set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, train=True, gray_mode=False, shuffle=False,filenames = ['train_gray.h5','val_gray.h5','train_mix_color.h5','val_color.h5']):
        super(Dataset, self).__init__()
        self.train = train
        self.gray_mode = gray_mode
        if not self.gray_mode:
            self.traindbf = 'datasets/h5files/'+filenames[2]
            self.valdbf = 'datasets/h5files/'+filenames[3]
        else:
            self.traindbf = 'datasets/h5files/'+filenames[0]
            self.valdbf = 'datasets/h5files/'+filenames[1]

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')
        self.keys = list(h5f.keys())
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
        else:
            h5f = h5py.File(self.valdbf, 'r')

        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

class HDF5Dataset(udata.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data,mask = 0,contrast_reduction = False, data_cache_size=3, add_blur=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.add_blur = add_blur
        self.mask = mask
        self.contrast_reduction = contrast_reduction

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data(index)
        if x.dtype is np.dtype('uint8'):
            x = np.float32(self.get_data(index)/255.)
        if self.add_blur:
            blur = np.random.random()
            if blur>0.9:
                blur_value = np.random.uniform(1,3)
                x = gaussian(x, blur_value,channel_axis = 0)
        
        if self.contrast_reduction:
            tmp = np.random.random()
            if tmp>0.66:
                interval = np.random.uniform(0.1,0.3)
                contrast_reduction_(x,interval)

        x = torch.from_numpy(x)
        return ((self.mask,x))

    def __len__(self):
        return len(self.data_info)
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all the data values, extracting datasets
            c = 0
            for dname, ds in h5_file.items():
                # c+=1
                # if c == 1000:
                #     break
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds[()], file_path)

                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            c = 0
            for dname, ds in h5_file.items():
                # add data to the data cache and retrieve
                # the cache index
                # c+=1
                # if c == 1000:
                #     break
                idx = self._add_to_cache(ds[()], file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data(self, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.data_info[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.data_info[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
    def get_info(self,i):
        fp = self.data_info[i]['file_path']
        return(fp)

class SubsetSequentialSampler(udata.Sampler):
    r""" Sample that samples from two datasets to form a batch

    """
    def __init__(self,indices,desc_order):
        self.indices = indices
        self.desc_order = desc_order
    def __iter__(self):
        if self.desc_order :
            return(self.indices[-i-1] for i in range(len(self.indices)))
        else :
            return(self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return(len(self.indices))

class BatchComposer(udata.Sampler):
    r"""
    compose two batch samplers that sample patches independantly
    """
    def __init__(self, sampler1, sampler2):
        self.sampler1 = sampler1
        self.sampler2 = sampler2
        self.composedbatch = zip(self.sampler1,self.sampler2)
    def __iter__(self):
        for indices in self.composedbatch :
            idx = indices[0] + indices[1]
            yield (idx)
    def __len__(self):
        return(sum([1 for _ in self.composedbatch]))

class BatchSchedulerSampler(udata.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.size = sum([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = udata.RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * (self.number_of_datasets+1)
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.size

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets+1):
                cur_batch_sampler = sampler_iterators[i//2]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i//2]
                        cur_samples.append(cur_sample)
                    except StopIteration: 
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i//2] = samplers_list[i//2].__iter__()
                        cur_batch_sampler = sampler_iterators[i//2]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i//2]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

