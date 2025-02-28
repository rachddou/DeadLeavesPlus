"""
Trains a FFDNet model

By default, the training starts with a learning rate equal to 1e-3 (--lr).
After the number of epochs surpasses the first milestone (--milestone), the
lr gets divided by 100. Up until this point, the orthogonalization technique
described in the FFDNet paper is performed (--no_orthog to set it off).

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils


from tensorboardX import SummaryWriter
from utils.models import FFDNet,UNetRes
from utils.dataset import HDF5Dataset
from utils.utils import weights_init_kaiming, batch_psnr, init_logger, \
			svd_orthogonalization, compute_noise_map


from torch.utils.data.dataset import ConcatDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def scheduler_1(epoch, epoch_max,ratio_max,alpha, mode = "linear"):
    x = epoch/epoch_max
    if mode == "linear":
        ratio = min(ratio_max*x,ratio_max)
    elif mode == "quadratic":
        ratio = min(ratio_max,ratio_max*np.sqrt(x))
    elif mode == "exponential":
        ratio = min(ratio_max,ratio_max*(1-np.exp(-alpha*x))/(1-np.exp(-alpha)))
    return(ratio)


def train(args):
    torch.manual_seed(1)
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')
    dataset_train_1 = HDF5Dataset(args.filenames[0], recursive=False, load_data=True, mask = 0,data_cache_size=4,  add_blur = args.blur)
    if not args.natural_only:
        dataset_train_2 = HDF5Dataset(args.filenames[1], recursive=False, load_data=True, mask = 1, \
                                        data_cache_size=4,  add_blur = True)
        dataset_train = ConcatDataset([dataset_train_1, dataset_train_2])
        dataset_val = HDF5Dataset(args.filenames[2], recursive=False, load_data=True, data_cache_size=4,  add_blur = False)
    else:
        dataset_train = dataset_train_1
        dataset_val = HDF5Dataset(args.filenames[1], recursive=False, load_data=True, data_cache_size=4, add_blur = False)
    

    print("\t# of training samples: %d\n" % int(len(dataset_train)))

    # Init loggers
    if not os.path.exists("TRAINING_LOGS/"+args.log_dir):
        os.makedirs("TRAINING_LOGS/"+args.log_dir)
    writer = SummaryWriter("TRAINING_LOGS/"+args.log_dir)
    logger = init_logger(args)

    # Create model
    if not args.gray:
        in_ch = 3
    else:
        in_ch = 1
    if args.model == "FFDNet":
        net = FFDNet(num_input_channels=in_ch)
        sca = 2
    elif args.model == "DRUNet":
        net = UNetRes(in_nc = 3, out_nc = 3)
        sca = 1
    # Initialize model with He init

    net.apply(weights_init_kaiming)
    # Define loss
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Resume training or start anew
    if args.resume_training:
        resumef = os.path.join("TRAINING_LOGS/"+args.log_dir, args.check_point)
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = args.epochs
            new_milestone = args.milestone
            save_every_epochs = args.save_every_epochs
            new_log_dir = args.log_dir
            current_lr = args.lr
            new_natural_only = args.natural_only
            new_no_orthog = args.no_orthog
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            args.epochs = new_epoch

            args.milestone = new_milestone
            args.lr = current_lr
            args.log_dir = new_log_dir
            args.save_every_epochs = save_every_epochs
            args.natural_only = new_natural_only
            args.no_orthog = new_no_orthog

            print("=> loaded checkpoint '{}' (epoch {})"\
                  .format(resumef, start_epoch))
            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['args'])
            print("==> checkpoint['args']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))

    else:
        start_epoch = 0
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = args.no_orthog
    # Training

    for epoch in range(start_epoch, args.epochs):
        # Learning rate value scheduling according to args.milestone
        if len(args.milestone) == 3 and epoch > args.milestone[2]:
            current_lr = args.lr / 1000.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[1]:
            current_lr = args.lr / 1000.
            training_params['no_orthog'] = True
        elif epoch > args.milestone[0]:
            current_lr = args.lr / 10.
        else:
            current_lr = args.lr

        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # create a data_loader here for curriculum learning
        loader_train = DataLoader(dataset=dataset_train, num_workers=30, \
            batch_size=args.batch_size, shuffle=True)
        limit = len(dataset_train)//(args.batch_size)
        # torch.set_num_threads(1)

        print("number of batches for this epoch : {}".format(limit))
        for i, data in enumerate(loader_train, 0):
            if i>= limit :
                break
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image
            loss_mask = data[0]
            img_train = data[1]
            noise = torch.zeros(img_train.size())
            stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                            size=noise.size()[0])
            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                noise[nx, :, :, :] = torch.FloatTensor(sizen).\
                                    normal_(mean=0, std=stdn[nx])
            imgn_train = img_train + noise
            # Create input Variables
            # loss_mask = Variable(torch.mean(loss_mask.float()).cuda())
            loss_mask = Variable(loss_mask.float()).cuda()
            img_train = Variable(img_train.cuda())
            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))
            noise_map = compute_noise_map(imgn_train, stdn_var,sca, mode = "constant")
            # Evaluate model and optimize it
            out_train = model(imgn_train, noise_map)

            if args.residual:
                loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            else:
                loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)


            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            # Results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train, noise_map), 0., 1.)
            psnr_train = batch_psnr(out_train, img_train, 1.)
            # PyTorch v0.4.0: loss.data[0] --> loss.item()

            if training_params['step'] % args.save_every == 0:

                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log the scalar values
                
                writer.add_scalar('loss', loss.data.item(), training_params['step'])

                print("[epoch %d][%d/%d] loss: %.4f  PSNR_train: %.4f" %\
                            (epoch+1, i+1, limit, loss.data.item(),psnr_train))

                writer.add_scalar('PSNR on training data', psnr_train, \
                    training_params['step'])


            training_params['step'] += 1
        # The end of each epoch
        model.eval()

        # Validation
        psnr_val = 0
        for _,valimg in dataset_val:
            img_val = torch.unsqueeze(valimg, 0).float()
            print(img_val.size())
            noise = torch.FloatTensor(img_val.size()).\
                    normal_(mean=0, std=args.val_noiseL).float()
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            sigma_noise = Variable(torch.cuda.FloatTensor([args.val_noiseL]))
            noise_map = compute_noise_map(img_val, sigma_noise, mode = "constant")
            if args.residual:
                out_val = torch.clamp(imgn_val-model(imgn_val, noise_map), 0., 1.)
            else:
                out_val = torch.clamp(model(imgn_val, noise_map), 0., 1.)
            psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', current_lr, epoch)

        # Log val images
        try:
            if epoch == 0:
                # Log graph of the model
                writer.add_graph(model, (imgn_val, sigma_noise), )
                # Log validation images
                for idx in range(2):
                    imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
                                            nrow=2, normalize=False, scale_each=False)
                    imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
                                            nrow=2, normalize=False, scale_each=False)
                    writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
                    writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
            for idx in range(2):
                imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
                                        nrow=2, normalize=False, scale_each=False)
                writer.add_image('Reconstructed validation image {}'.format(idx), \
                                imrecons, epoch)
            # Log training images
            imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
                         scale_each=True)
            writer.add_image('Training patches', imclean, epoch)

        except Exception as e:
            logger.error("Couldn't log results: {}".format(e))

        # save model and checkpoint
        training_params['start_epoch'] = epoch + 1
        torch.save(model.state_dict(), os.path.join("TRAINING_LOGS/"+args.log_dir, 'net.pth'))
        save_dict = { \
            'state_dict': model.state_dict(), \
            'optimizer' : optimizer.state_dict(), \
            'training_params': training_params, \
            'args': args\
            }
        filename = 'ckpt'
        
        if epoch+1 == args.epochs :
            final_filename = filename+"_back_up"
            torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, final_filename+'.pth'))
        torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, filename+'.pth'))
        if epoch % args.save_every_epochs == 0:
            torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, \
                                      filename+'_e{}.pth'.format(epoch+1)))
        del save_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--gray", action='store_true',\
                        help='train grayscale image denoising instead of RGB')


    #Training parameters
    parser.add_argument("--batch_size","--bs",type=int, default=256, 	\
                     help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, \
                     help="Number of total training epochs")
    parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
                     help="Noise training interval")
    parser.add_argument("--val_noiseL", type=float, default=25, \
                        help='noise level used on validation set')
    parser.add_argument("--milestone",nargs = '+', type=int, default=[50, 60,80], \
                        help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-3, \
                     help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true',\
                        help="Don't perform orthogonalization as regularization")
    parser.add_argument("--resume_training", action='store_true',\
                        help="Don't perform orthogonalization as regularization")

    parser.add_argument("--grad_clip", action='store_true',\
                        help="perform gradient clipping")
    parser.add_argument("--natural_only", action='store_true',\
                        help="activates natural training only")
    parser.add_argument("--filenames", "--fn", type=str, nargs = '+', default=['datasets/h5files/train_imnat/','datasets/h5files/train_dl/','datasets/h5files/val'], \
                        help="How many times to perform data augmentation")

    # Saving parameters
    parser.add_argument("--check_point", "--cp", type=str, default = "ckpt.pth",\
                        help="resume training from a previous checkpoint")
    parser.add_argument("--save_every", type=int, default=10,\
                        help="Number of training steps to log psnr and perform \
                        orthogonalization")
    parser.add_argument("--save_every_epochs", type=int, default=50,\
                        help="Number of training epochs to save state")
    parser.add_argument("--log_dir", type=str, default="logs", \
                     help='path of log files')
    argspar = parser.parse_args()
    # Normalize noise between [0, 1]
    argspar.val_noiseL /= 255.
    argspar.noiseIntL[0] /= 255.
    argspar.noiseIntL[1] /= 255.

    print("\n### Training FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    train(argspar)

