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
from sampling import *
from plot_utils import show_plot

import argparse
import datetime
import os
from os import makedirs


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('--test_mode', type=bool, default=False)

parser.add_argument('--finetune', default=True)
parser.add_argument('--triplet', default=True)
parser.add_argument('--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs (default: 500)')
parser.add_argument('--batch_size', type=int, default=500,
                    help='size of batch (default: 100)')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')

parser.add_argument('--train_exp_path', default='./data/exp/exp_train.pkl',
                    help='path for train exp (default: ./data/exp/exp_train.pkl')
parser.add_argument('--test_exp_path', default='./data/exp/exp_test.pkl',
                    help='path for test exp (default: ./data/exp/exp_test.pkl')

parser.add_argument('--train_lookup_path', default='./data/lookup/lookup_train.csv',
                    help='path for train lookup (default: ./data/lookup/lookup_train.csv')
parser.add_argument('--test_lookup_path', default='./data/lookup/lookup_test.csv',
                    help='path for test lookup (default: ./data/lookup/lookup_test.csv')

parser.add_argument('--neg_samples', type=int, default=3,
                    help='number of negative samples (default: 3)')
parser.add_argument('--margin', type=int, default=0.3)
#parser.add_argument('--model_path', default='./model/AE/210625-173704/',
#                    help='path for pretrained model')
parser.add_argument('--model_path', default='./model/mjnet/base_double_triple_i1_mrr_best.pth',
                    help='path for pretrained model')
parser.add_argument('--model_path_test', default='./model/EN/210928-192531_3_0.3ft/',
                    help='path for finetuned model')

parser.add_argument('--neg_dict', default='./data/neg_dict.pkl')
parser.add_argument('--neg_dict_test', default='./data/neg_dict_test.pkl')

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
    neg_dict = os.path.isfile(args.neg_dict)
    neg_dict_test = os.path.isfile(args.neg_dict_test)
    
    print('test_mode:', args.test_mode)
    print('finetune:', args.finetune)
    if args.finetune==True:
        print('pretrained model:', args.model_path)
    
    criterion = TripletLoss(args.margin)
    results = {}
    
    torch.manual_seed(42)
    
    if args.test_mode == True:
        result_img_name = 'test'
        save_result_path = f'./result/EN/{cur_date}_test_mode/'
        
        with open(args.test_exp_path, 'rb') as f:
            test_exp = pickle.load(f)
            
        test_lookup = pd.read_csv(args.test_lookup_path)
        test_dataset = Dataset_AE(test_exp, test_lookup)
        
        testloader = get_loader_infer(test_dataset, batch_size)
        
        for fold in range(k_folds):
            
            print('-------- Start testing --------')
            
            #model = VanillaEncoder().to(device)
            model = MJnetEN().to(device)
            model_path = f'{args.model_path_test}model_fold{fold}.pth'
            #model_path = args.model_path_test
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint,strict=False)
            
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                np_embed = np.zeros((len(test_dataset), 256))
                shRNA, cell, gene, index = [], [], [], []
                k = 0
                for i, data in enumerate(testloader, 0):
                    tensor_list = data
                    shRNA.extend(tensor_list[2])
                    cell.extend(tensor_list[3])
                    gene.extend(tensor_list[4])
                    index.extend(tensor_list[5])

                    anchor_embed = model(tensor_list[0]).cpu().detach().numpy()
                    np_embed[k:k+anchor_embed.shape[0],:] = anchor_embed
                    k = k+anchor_embed.shape[0]

                labels = {'shRNA' : shRNA, 'cell' : cell, 'gene' : gene, 'index' : index}
                    
                with open(f'./result/embedding/embedding_mj_ft{fold}.pkl', 'wb') as f:
                    pickle.dump(np_embed, f)
                with open(f'./result/embedding/labels_mj_ft{fold}.pkl', 'wb') as f:
                    pickle.dump(labels, f)
    
    else:
        if args.finetune==True:
            result_img_name = 'finetune'
            save_model_path = f'./model/EN/{cur_date}_{args.neg_samples}_{args.margin}ft/'
            save_result_path = f'./result/EN/{cur_date}_{args.neg_samples}_{args.margin}ft/'
        else:
            result_img_name = 'train'
            save_model_path = f'./model/EN/{cur_date}_{args.neg_samples}_{args.margin}/'
            save_result_path = f'./result/EN/{cur_date}_{args.neg_samples}_{args.margin}/'
        
        with open(args.train_exp_path, 'rb') as f:
            train_exp = pickle.load(f)
        with open(args.test_exp_path, 'rb') as f:
            test_exp = pickle.load(f)
            
        train_lookup = pd.read_csv(args.train_lookup_path)
        test_lookup = pd.read_csv(args.test_lookup_path)
        
        if neg_dict == False:
            get_neg_dict(train_lookup, args.neg_samples)
        if neg_dict_test == False:
            get_neg_dict(test_lookup, args.neg_samples, test=True)
        
        with open(args.neg_dict, 'rb') as f:
            neg_dic = pickle.load(f)
        with open(args.neg_dict_test, 'rb') as f:
            neg_dic_test = pickle.load(f)
            
        train_dataset = Dataset_AE(train_exp, train_lookup)
        test_dataset = Dataset_AE(test_exp, test_lookup)
        kfold = KFold(n_splits=k_folds, shuffle=True)
    
        testloader = get_loader_infer(test_dataset, batch_size)
        
        # Make directory for saved model
        try:
            makedirs(save_model_path, exist_ok=True)
            makedirs(save_result_path, exist_ok=True)
        except:
            save_model_path = './model/'
            save_result_path = './result/'
        
        print('----------------------------------')
    
        for fold, (train_index, valid_index) in enumerate(kfold.split(train_dataset)):
        
            print(f'Fold {fold}')
            print('----------------------------------')
        
            # parameters for printing
            tloss_plot_list = []
            tloss_plot = 0.0
            vloss_plot_list = []
            vloss_plot = 0.0
        
            avg_losses = []
            avg_vlosses = []
        
    
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_index)
    
            trainloader = get_loader(train_dataset, train_subsampler, batch_size)
            validloader = get_loader(train_dataset, valid_subsampler, batch_size)
            
            
            model = VanillaEncoder()
            #model = MJnetEN()
            model = model.to(device)
            if args.finetune==True:
                model_path = args.model_path
                #model_path = f'{args.model_path}model_fold{fold}.pth'
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint,strict=False)
                for param in model.parameters():            
                    param.requires_grad = False
            
                model.profile_encoder.MLP_profile[3] = nn.Linear(512,256)
                optimizer = torch.optim.Adam(model.profile_encoder.MLP_profile[3].parameters(), lr=learning_rate)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
            for epoch in range(1, num_epochs+1):
            
                tloss = 0.0
                vloss = 0.0
                tlosses = []
                vlosses = []
            
                model.train()
                for i, data in enumerate(trainloader, 0):
                    tensor_list = data
                    pos, neg = get_posneg_samples(tensor_list, train_exp, train_lookup, args.neg_samples, neg_dic)

                    anchor_embed, pos_embed, neg_embed = model(tensor_list[0], pos.to(device), neg.to(device), args.triplet)
                    losses = criterion(anchor_embed, pos_embed, neg_embed)
                    losses.backward()
                    optimizer.step()
                  
                    tloss += losses.item()
                    tloss_plot += losses.item()
                    tlosses.append(losses.item())
                
                # avg(loss) per epoch
                avg_loss = np.average(tlosses)
                avg_losses.append(avg_loss)
                # Plot
                avg_tloss_plot = tloss_plot / (args.plot_every*(i+1))         
                
                model.eval()
                for i, data in enumerate(validloader, 0):
                    tensor_list = data
                    pos, neg = get_posneg_samples(tensor_list, train_exp, train_lookup, args.neg_samples, neg_dic)

                    anchor_embed, pos_embed, neg_embed = model(tensor_list[0], pos.to(device),neg.to(device),args.triplet)

                    losses = criterion(anchor_embed, pos_embed, neg_embed)   
                    vloss += losses.item()
                    vloss_plot += losses.item()
                    vlosses.append(losses.item())
                    
                # avg(loss) per epoch    
                avg_vloss = np.average(vlosses)
                avg_vlosses.append(avg_vloss)
                #plot
                avg_vloss_plot = vloss_plot / (args.plot_every*(i+1))
                #import pdb; pdb.set_trace()
                # Save model
                save_path = save_model_path + f'/model_fold{fold}.pth'
                torch.save(model.state_dict(), save_path)
                
                if epoch % args.print_every == 0:
                    print('Epoch %d / %d (%d%%) train loss: %.4f, valid loss: %.4f' \
                        % (epoch, num_epochs, epoch / num_epochs * 100, avg_loss, avg_vloss))

                if epoch % args.plot_every == 0:
                    tloss_plot_list.append(avg_tloss_plot)
                    tloss_plot = 0.0
                    vloss_plot_list.append(avg_vloss_plot)
                    vloss_plot = 0.0
            #import pdb; pdb.set_trace()
            show_plot(tloss_plot_list, args.plot_every, fold, eval_points=vloss_plot_list, save_path=save_result_path, file_name=result_img_name)
              
            print('-------- Starting testing --------')
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint)
        
            test_loss = 0.0
            model.eval()
            with torch.no_grad():
                        
                for i, data in enumerate(testloader, 0):
                    tensor_list = data
                    pos, neg = get_posneg_samples(tensor_list, test_exp, test_lookup, args.neg_samples, neg_dic_test, test=True)

                    anchor_embed, pos_embed, neg_embed = model(tensor_list[0], pos.to(device),neg.to(device),args.triplet)
                
                    losses = criterion(anchor_embed, pos_embed, neg_embed)   
                    test_loss += losses.item()
            
                print('test_loss of fold: %.4f' % (test_loss/len(testloader)))
                print('-----------------------------------')
                results[fold] = test_loss/len(testloader)
            print(f'K-Fold CV Results of {k_folds} Folds')
            print('-----------------------------------')
            for f in range(fold+1):
                print('[Fold %d] test loss: %.4f' % \
                    (f, results[f]))
            print('%d folds test loss: %.4f' \
                % ((fold+1), sum(results.values())/len(results.items())))
            print('Saved Model Path: %s' % save_model_path)
            print('-----------------------------------')