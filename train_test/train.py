import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils


from tensorboardX import SummaryWriter
from utils.models import UNetRes,FFDNet
from utils.utils import weights_init_kaiming
from utils.dataset import HDF5Dataset
from utils.utils import batch_psnr, init_logger, \
			svd_orthogonalization,compute_noise_map

def train_ffdnet(args):
    torch.manual_seed(1)
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')
    dataset_train = HDF5Dataset(args.filenames[0], recursive=False, load_data=True,data_cache_size=4)

    dataset_val = HDF5Dataset(args.filenames[1], recursive=False, load_data=True, data_cache_size=4)
    

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
    net = FFDNet(num_input_channels=in_ch)
    sca = 2

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
    
    # Data loader
    loader_train = DataLoader(dataset=dataset_train, num_workers=30, \
    batch_size=args.batch_size, shuffle=True)
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


        limit = len(dataset_train)//(args.batch_size)
        for i, data in enumerate(loader_train, 0):
            if i>= limit :
                break
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image
            img_train = data
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
            
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train, noise_map), 0., 1.)
            psnr_train = batch_psnr(out_train, img_train, 1.)

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


def train_drunet(args):
    torch.manual_seed(1)
    r"""Performs the main training loop
    """
    # Load dataset
    print('> Loading dataset ...')
    dataset_train= HDF5Dataset(args.filenames[0], recursive=False, load_data=True, data_cache_size=4)
    dataset_val = HDF5Dataset(args.filenames[1], recursive=False, load_data=True, data_cache_size=4)
    

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
    if args.blind:
        net = UNetRes(in_nc = in_ch, out_nc = in_ch)
    else:    
        net = UNetRes(in_nc = in_ch+1, out_nc = in_ch)
        noise_map_mode = "single"
    sca = 1
    # Initialize model with He init

    # Define loss
    criterion = nn.L1Loss(size_average=False)

    # Move to GPU
    model = net.cuda()
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

            new_log_dir = args.log_dir
            current_lr = args.lr
            new_no_orthog = args.no_orthog
            args = checkpoint['args']
            training_params = checkpoint['training_params']
            start_epoch = training_params['start_epoch']
            
            args.lr = current_lr
            args.log_dir = new_log_dir
            args.no_orthog = new_no_orthog
            
            args.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".\
                   format(resumef))

    else:
        training_params = {}
        training_params['step'] = 0
        training_params['current_lr'] = 0
        training_params['no_orthog'] = args.no_orthog
    # Training
    schedule = 0
    current_lr = 1e-4
    sub_epoch = 0
    while current_lr >= 5e-7:
        if schedule > 0:
            current_lr = current_lr/2
        # set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        rdm_sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=1600000, generator=None)
        
        loader_train = DataLoader(dataset=dataset_train,sampler = rdm_sampler, num_workers=20, \
            batch_size=16)
        
        for i, data in enumerate(loader_train, 0):
            if i>= 100000 :
                break
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # inputs: noise and noisy image
            img_train = data
            noise = torch.zeros(img_train.size())
            stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], \
                            size=noise.size()[0])
            for nx in range(noise.size()[0]):
                sizen = noise[0, :, :, :].size()
                noise[nx, :, :, :] = torch.FloatTensor(sizen).\
                                    normal_(mean=0, std=stdn[nx])
            imgn_train = img_train + noise
            # Create input Variables
            img_train = Variable(img_train.cuda())
            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            stdn_var = Variable(torch.cuda.FloatTensor(stdn))
            if not(args.blind):
                noise_map = compute_noise_map(imgn_train, stdn_var,sca, mode = noise_map_mode)
                input_tensor = torch.cat((imgn_train, noise_map), dim=1)
                out_train = model(input_tensor)
            else:
            # Evaluate model and optimize it
                out_train = model(imgn_train)

            if args.residual:
                loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            else:
                loss = criterion(out_train, img_train) / (imgn_train.size()[0]*2)


            loss.backward()
            if args.grad_clip:
                nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()

            if training_params['step'] % args.save_every == 0:
                model.eval()
                    
                if args.residual:
                    out_train = torch.clamp(imgn_train-out_train, 0., 1.)
                else:
                    out_train = torch.clamp(out_train,0,1)
                psnr_train = batch_psnr(out_train, img_train, 1.)
                # Apply regularization by orthogonalizing filters
                if not training_params['no_orthog']:
                    model.apply(svd_orthogonalization)

                # Log the scalar values
                
                writer.add_scalar('loss', loss.data.item(), training_params['step'])

                print("[epoch %d][%d/%d] loss: %.4f  PSNR_train: %.4f" %\
                            (schedule+1, i+1, 100000, loss.data.item(),psnr_train))

                writer.add_scalar('PSNR on training data', psnr_train, \
                    training_params['step'])


            training_params['step'] += 1
            if training_params['step'] % 1000 == 0:
                sub_epoch +=1
                model.eval()

                # Validation
                psnr_val = 0
                for _,valimg in dataset_val:
                    img_val = torch.unsqueeze(valimg[:,:496,:496], 0).float()
                    print(img_val.size())
                    noise = torch.FloatTensor(img_val.size()).\
                            normal_(mean=0, std=args.val_noiseL).float()
                    imgn_val = img_val + noise
                    img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                    sigma_noise = Variable(torch.cuda.FloatTensor([args.val_noiseL]))
                    if not(args.blind):
                        noise_map = compute_noise_map(img_val, sigma_noise,sca, mode = noise_map_mode )
                        input_tensor = torch.cat((imgn_val, noise_map), dim=1)
                        out_val = model(input_tensor)
                    else:
                        out_val = model(imgn_val)
                    if args.residual:
                        out_val = torch.clamp(imgn_val-out_val, 0., 1.)
                    else:
                        out_val = torch.clamp(out_val, 0., 1.)
                    psnr_val += batch_psnr(out_val, img_val, 1.)
                psnr_val /= len(dataset_val)
                print("\n[epoch %d] [sub epoch %d] PSNR_val: %.4f" % (schedule+1,sub_epoch, psnr_val))
                writer.add_scalar('PSNR on validation data', psnr_val, sub_epoch)
                writer.add_scalar('Learning rate', current_lr, sub_epoch)
                try:
                    if schedule == 0:
                        # Log graph of the model
                        writer.add_graph(model, (imgn_val, sigma_noise), )
                        # Log validation images
                        for idx in range(2):
                            imclean = utils.make_grid(img_val.data[idx].clamp(0., 1.), \
                                                    nrow=2, normalize=False, scale_each=False)
                            imnsy = utils.make_grid(imgn_val.data[idx].clamp(0., 1.), \
                                                    nrow=2, normalize=False, scale_each=False)
                            writer.add_image('Clean validation image {}'.format(idx), imclean, sub_epoch)
                            writer.add_image('Noisy validation image {}'.format(idx), imnsy, sub_epoch)
                    for idx in range(2):
                        imrecons = utils.make_grid(out_val.data[idx].clamp(0., 1.), \
                                                nrow=2, normalize=False, scale_each=False)
                        writer.add_image('Reconstructed validation image {}'.format(idx), \
                                        imrecons, sub_epoch)
                    # Log training images
                    imclean = utils.make_grid(img_train.data, nrow=8, normalize=True, \
                                scale_each=True)
                    writer.add_image('Training patches', imclean, sub_epoch)

                except Exception as e:
                    logger.error("Couldn't log results: {}".format(e))
        # The end of each epoch
        schedule+=1
        # save model and checkpoint
        training_params['start_epoch'] = schedule + 1
        torch.save(model.state_dict(), os.path.join("TRAINING_LOGS/"+args.log_dir, 'net.pth'))
        save_dict = { \
            'state_dict': model.state_dict(), \
            'optimizer' : optimizer.state_dict(), \
            'training_params': training_params, \
            'args': args\
            }
        filename = 'ckpt'
        torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, filename+'.pth'))

        torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, \
                                      filename+'_e{}.pth'.format(schedule+1)))
    final_filename = filename+"_back_up"
    torch.save(save_dict, os.path.join("TRAINING_LOGS/"+args.log_dir, final_filename+'.pth'))















