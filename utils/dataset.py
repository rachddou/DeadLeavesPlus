import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from utils.utils import data_augmentation, normalize
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

def prepare_data_curriculum(root_path, \
        val_data_path, \
        patch_size, \
        strides, \
        scale_numbers,\
        max_num_patches=None, \
        filenames = ['train/','val/'], \
        aug_times=1, \
        gray_mode=False):
    types = ('*.bmp', '*.png')
    if scale_numbers == 1:
        scales = [1]
    elif scale_numbers == 2:
        scales = [1, 0.6]
    elif scale_numbers ==3:
        scales = [1, 0.8,0.65]
        
    sub_directory_paths = [os.path.join(root_path,path) for path in os.listdir(root_path)]
    
    nested_files_list = []
    for data_path in sub_directory_paths:    
        files_list = []
        for tp in types:
            files_list.extend(glob.glob(os.path.join(data_path, tp)))
        files_list.sort()
        nested_files_list.append(files_list)
        
        
    if not(os.path.isdir("datasets/h5files/"+filenames[0])):
        os.makedirs("datasets/h5files/"+filenames[0])
    if not(os.path.isdir("datasets/h5files/"+filenames[1])):
        os.makedirs("datasets/h5files/"+filenames[1])
    traindbf = os.path.join("datasets/h5files/"+filenames[0], "train.h5")
    valdbf = os.path.join("datasets/h5files/"+filenames[1], "val.h5")

    if max_num_patches is None:
        max_num_patches = 500000
        
    with h5py.File(traindbf, 'w') as h5f:
        counter = 0
        for curriculum in range(len(nested_files_list)):
            print(curriculum)
            stride = strides[curriculum]
            files_list = nested_files_list[curriculum]
            file_counter = 0
            patch_counter = 0
            while file_counter < len(files_list) and patch_counter < max_num_patches:
                img_original = cv2.imread(files_list[i])

                if not gray_mode:
                    img = (cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)

                patches = img_to_patches(img, win=patch_size, stride=stride)
                for nx in range(patches.shape[3]):
                    data = np.uint8(np.clip(data_augmentation(patches[:, :, :, nx].copy(), \
                            np.random.randint(0, 7)),0,255))
                    h5f.create_dataset("{}_{}".format(str(curriculum).zfill(2),str(counter).zfill(8)), data=img)                    
                    patch_counter+=1
                file_counter += 1
            counter+=patch_counter
            

    print('\n> Total')
    print('\ttraining set, # samples %d' % counter)
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
        gray_mode=False):
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


    class DifficultyLevelDataset(udata.Dataset):
        """
        A PyTorch dataset that loads images from an HDF5 file where data is stored in subgroups by difficulty level.
        
        Args:
            file_path (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        def __init__(self, file_path, transform=None):
            super().__init__()
            self.file_path = file_path
            self.transform = transform
            self.data = {}  # Dictionary to store images by difficulty level
            self.indices = []  # List of (difficulty, image_id) tuples for indexing
            
            # Load the entire dataset into memory
            self._load_data()
            
        def _load_data(self):
            """Load all data from the HDF5 file into memory as a nested dictionary."""
            with h5py.File(self.file_path, 'r') as h5f:
                # Iterate through difficulty levels (1-5)
                for difficulty in range(1, 6):
                    # Initialize empty dictionary for this difficulty level
                    self.data[difficulty] = {}
                    
                    # Check if this difficulty level exists in the HDF5 file
                    if str(difficulty) in h5f:
                        group = h5f[str(difficulty)]
                        
                        # Load all datasets in this difficulty group
                        for img_key in group.keys():
                            self.data[difficulty][img_key] = group[img_key][()]
                            self.indices.append((difficulty, img_key))
        
        def __len__(self):
            """Return the total number of images across all difficulty levels."""
            return len(self.indices)
        
        def __getitem__(self, idx):
            """
            Get an image and its difficulty level by index.
            
            Args:
                idx (int): Index of the sample to retrieve
                
            Returns:
                tuple: (image, difficulty_level) where image is a numpy array and 
                       difficulty_level is an integer from 1 to 5
            """
            difficulty, img_key = self.indices[idx]
            img = self.data[difficulty][img_key]
            
            # Convert to float32 and normalize if image is in uint8 format
            if img.dtype == np.uint8:
                img = np.float32(img / 255.0)
            
            # Convert to PyTorch tensor
            img = torch.from_numpy(img)
            
            # Apply transforms if any
            if self.transform:
                img = self.transform(img)
            
            return img, difficulty

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
    def __init__(self, file_path, recursive, load_data, data_cache_size=3):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

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
        x = self.get_data(index)
        if x.dtype is np.dtype('uint8'):
            x = np.float32(self.get_data(index)/255.)
        x = torch.from_numpy(x)
        return (x)

    def __len__(self):
        return len(self.data_info)
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            c = 0
            for dname, ds in h5_file.items():
                idx = -1
                if load_data:
                    idx = self._add_to_cache(ds[()], file_path)
                self.data_info.append({'file_path': file_path, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            c = 0
            for dname, ds in h5_file.items():
                idx = self._add_to_cache(ds[()], file_path)
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
                self.data_info[file_idx + idx]['cache_idx'] = idx

        if len(self.data_cache) > self.data_cache_size:
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
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



# class Curriculum_dataset(udata.Dataset):
#     def __init__(self, file_path,load_data,):
#         super().__init__()
#         self.file_path = file_path
#         self.load_data = load_data
#         self.data = {"level1":{}, "level2":{},"level3":{}, "level4":{}, "level5":{},"level6":{}}
#         if self.load_data:
#             self._load_data()

#     def _load_data(self):
#         with h5py.File(self.file_path) as h5_file:
#             for dname, ds in h5_file.items():
#                 self

#     def __getitem__(self, index):
#         return super().__getitem__(index)
#     def __len__(self):
#         return super().__len__()