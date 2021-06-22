from __future__ import unicode_literals, print_function, division
import random

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import scipy
import sklearn
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm

from model import *
from data import *
from loss import *
from plot_utils import show_plot

import argparse
import datetime
from os import makedirs


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('--finetune', default=True)
parser.add_argument('-k', '--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('-e', '--num_epochs', type=int, default=1,
                    help='number of epochs (default: 500)')
parser.add_argument('-b', '--batch_size', type=int, default=500,
                    help='size of batch (default: 100)')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')
parser.add_argument('--neg_samples', type=int, default=3,
                    help='# of negative samples (default: 3)')
parser.add_argument('--model_path', default='./model/AE/210621-184852/',
                    help='path for pretrained model')
parser.add_argument('--exp_path', default='./data/exp/exp_train.pkl',
                    help='path for train exp (default: ./data/exp/exp_train.pkl')
parser.add_argument('--lookup_path', default='./data/lookup/lookup_train.csv',
                    help='path for train exp (default: ./data/lookup/lookup_train.csv')
parser.add_argument('--shRNA_path', default='./data/shRNA/shRNA_processed.csv')
parser.add_argument('--shRNA_dict', default='./data/shRNA/RNA_dict.pkl')
parser.add_argument('--seed_similarity', default='./data/shRNA/seed_similarity_7mer.pkl')
parser.add_argument('--margin', type=int, default=1)
parser.add_argument('--plot_every', type=int, default=1,
                    help='number of epochs for plotting (default: 50)')
parser.add_argument('--print_every', type=int, default=1,
                    help='number of epochs for printing losses for plot (default: 1)')

args = parser.parse_args()
cur_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')


if __name__ == '__main__':
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    k_folds = args.k_folds
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    plot_every = args.plot_every
    
    result_img_name = 'finetune'
    save_model_path = f'./model/AE/{cur_date}_ft/'
    save_result_path = f'./result/AE/{cur_date}_ft/'
    
    criterion = TripletLoss(args.margin)
    results = {}
    
    torch.manual_seed(42)
    
    with open(args.exp_path,'rb') as f:
        exp = pickle.load(f)
        
    lookup = pd.read_csv(args.lookup_path)
    shRNA = pd.read_csv(args.shRNA_path)
    with open(args.shRNA_dict, 'rb') as f:
        shRNA_dict = pickle.load(f)
    with open(args.seed_similarity, 'rb') as f:
        seed_similarity = pickle.load(f)
    
    dataset = Dataset_AE(exp, lookup)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    # Make directory for saved model
    try:
        makedirs(save_model_path, exist_ok=True)
        makedirs(save_result_path, exist_ok=True)
    except:
        save_model_path = './model/'
        save_result_path = './result/'
        
    print('----------------------------------')
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
        
        print(f'Fold {fold}')
        print('----------------------------------')
        
        # parameters for printing
        loss_plot_list = []
        loss_plot = 0.0
        
        avg_losses = []
        avg_vlosses = []
        
    
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
    
        trainloader = get_loader(dataset, train_subsampler, batch_size)
        testloader = get_loader(dataset, test_subsampler, batch_size)
        #import pdb; pdb.set_trace()
        model_path = f'{args.model_path}encoder_fold{fold}.pth'
        checkpoint = torch.load(model_path)
        
        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = False
            
        model.MLP_ft = nn.Linear(512,256)
        
        model = model.to(device)
        #decoder = VanillaDecoder().to(device)
        
        optimizer = torch.optim.Adam(model.MLP_ft.parameters(), lr=learning_rate)
        
        for epoch in range(1, num_epochs+1):            
            loss = 0.0
            vloss = 0.0
            losses = []
            vlosses = []
            
            model.train()
            
            for i, data in enumerate(tqdm(trainloader, 0)):                
                tensor_list = data
                anchor_shRNA = tensor_list[2]
                anchor_target = tensor_list[4]
                while True:
                    pos_idx = random.sample(lookup[lookup['cmap_name'] == anchor_target].index(), args.neg_samples)
                    if tensor_list[5] not in pos_idx:
                        break
                idx_shRNA = shRNA_dict[anchor_shRNA]
                while True:
                    idx_shRNA_neg = torch.multinomial(seed_similarity[idx_shRNA,:], args.neg_samples)
                    if idx_shRNA not in list(idx_shRNA_neg):
                        break
                print('-------- Calculating TripletLoss --------')
                for i in range(tqdm(args.neg_samples)):
                    pos = exp[pos_idx[i]]
                    shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value == idx_shRNA_neg[i]][0]
                    neg_idx = random.sample(lookup[lookup['pert_mfc_id'] == shRNA_neg].index(),1)
                    neg = exp[neg_idx]
                    anchor_embed, anchor_embed, neg_embed = model(tensor_list[0], pos, neg, args.finetune)              
                    losses = criterion(anchor_embed, anchor_embed, neg_embed)      
                    losses.backward()
                    optimizer.step()
                  
                    loss += losses.item()
                    loss_plot += losses.item()
                
            model.eval()

            for i, data in enumerate(testloader, 0):
                tensor_list = data
                anchor_shRNA = tensor_list[2]
                anchor_target = tensor_list[4]
                while True:
                    pos_idx = random.sample(lookup[lookup['cmap_name'] == anchor_target].index(), args.neg_samples)
                    if tensor_list[5] not in pos_idx:
                        break
                idx_shRNA = shRNA_dict[anchor_shRNA]
                while True:
                    idx_shRNA_neg = torch.multinomial(seed_similarity[idx_shRNA,:], args.neg_samples)
                    if idx_shRNA not in list(idx_shRNA_neg):
                        break
                for i in range(tqdm(args.neg_samples)):
                    pos = exp[pos_idx[i]]
                    shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value == idx_shRNA_neg[i]][0]
                    neg_idx = random.sample(lookup[lookup['pert_mfc_id'] == shRNA_neg].index(),1)
                    neg = exp[neg_idx]
                    anchor_embed, anchor_embed, neg_embed = model(tensor_list[0], pos, neg, args.finetune)
                    losses = criterion(anchor_embed, anchor_embed, neg_embed)

                    vloss += losses.item()
                    vlosses.append(losses.item())

                    # avg(loss) per epoch
                    avg_loss = np.average(losses.cpu().detach().numpy())
                    avg_losses.append(avg_loss)
            avg_vloss = np.average(vlosses)
            avg_vlosses.append(avg_vloss)
            
            # Save model
            save_path = save_model_path + f'/model_fold{fold}_ft.pth'
            torch.save(model.state_dict(), save_path)

            # Plot
            avg_loss_plot = loss_plot / (args.plot_every*(i+1))
            #import pdb; pdb.set_trace()
            if epoch % args.print_every == 0:
                print('Epoch %d / %d (%d%%) train loss: %.4f, valid loss: %.4f' \
                    % (epoch, num_epochs, epoch / num_epochs * 100, avg_loss, avg_vloss))

            if epoch % args.plot_every == 0:
                loss_plot_list.append(avg_loss_plot)
                loss_plot = 0.0

        
        show_plot(loss_plot_list, args.plot_every, fold, save_path=save_result_path, file_name=result_img_name)
              
        print('-------- Starting testing --------')
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint)
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
                        
            for i, data in enumerate(tqdm(testloader, 0)):
                tensor_list = data
                anchor_shRNA = tensor_list[2]
                anchor_target = tensor_list[4]
                while True:
                    pos_idx = random.sample(lookup[lookup['cmap_name'] == anchor_target].index(), args.neg_samples)
                    if tensor_list[5] not in pos_idx:
                        break
                idx_shRNA = shRNA_dict[anchor_shRNA]
                while True:
                    idx_shRNA_neg = torch.multinomial(seed_similarity[idx_shRNA,:], args.neg_samples)
                    if idx_shRNA not in list(idx_shRNA_neg):
                        break
                for i in range(tqdm(args.neg_samples)):
                    pos = exp[pos_idx[i]]
                    shRNA_neg = [shRNA for shRNA, value in shRNA_dict.items() if value == idx_shRNA_neg[i]][0]
                    neg_idx = random.sample(lookup[lookup['pert_mfc_id'] == shRNA_neg].index(),1)
                    neg = exp[neg_idx]
                    anchor_embed, anchor_embed, neg_embed = model(tensor_list[0], pos, neg, args.finetune)
                    losses = criterion(anchor_embed, anchor_embed, neg_embed)

                    val_loss += losses.item()
            
            print('val_loss of fold: %.4f' % (val_loss/len(testloader)))
            print('-----------------------------------')
            results[fold] = val_loss/len(testloader)
        print(f'K-Fold CV Results of {k_folds} Folds')
        print('-----------------------------------')
        for f in range(fold+1):
            print('[Fold %d] valid loss: %.4f' % \
                (f, results[f]))
        print('%d folds valid loss: %.4f' \
            % ((fold+1), sum(results.values())/len(results.items())))
        print('Saved Model Path: %s' % save_model_path)
        print('-----------------------------------')