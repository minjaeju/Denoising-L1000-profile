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
from plot_utils import show_plot

import argparse
import datetime
from os import makedirs


parser = argparse.ArgumentParser(description='Parser for training.')
parser.add_argument('--test_mode', type=bool, default=False)

parser.add_argument('--k_folds', type=int, default=5,
                    help='number of folds for cross validation (default: 5)')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of epochs (default: 500)')
parser.add_argument('--batch_size', type=int, default=500,
                    help='size of batch (default: 100)')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate (default: 1e-5)')

parser.add_argument('--train_exp_path', default='./data/exp/exp_train.pkl',
                    help='path for train exp (default: ./data/exp/exp_train.pkl')
parser.add_argument('--test_exp_path', default='./data/exp/exp_test.pkl',
                    help='path for train exp (default: ./data/exp/exp_test.pkl')
parser.add_argument('--train_lookup_path', default='./data/lookup/lookup_train.csv',
                    help='path for train exp (default: ./data/lookup/lookup_train.csv')
parser.add_argument('--test_lookup_path', default='./data/lookup/lookup_test.csv',
                    help='path for train exp (default: ./data/lookup/lookup_test.csv')
#parser.add_argument('--shRNA_path', default='./data/shRNA/shRNA_processed.csv')
parser.add_argument('--model_path', default='./model/AE/210621-184852/',
                    help='path for pretrained model')

parser.add_argument('--plot_every', type=int, default=50,
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
    
    criterion = nn.MSELoss()
    results = {}
    results_corr = {}
    
    torch.manual_seed(42)
    
    if args.test_mode == True:
        result_img_name = 'test'
        save_result_path = f'./result/AE/{cur_date}_test_mode/'
        
        with open(args.test_exp_path,'rb') as f:
            test_exp = pickle.load(f)
            
        test_lookup = pd.read_csv(args.test_lookup_path)
        test_dataset = Dataset_AE(test_exp, test_lookup)
            
        testloader = get_loader_infer(test_dataset, batch_size)
        
        for fold in range(k_folds):
            
            print('-------- Starting testing --------')
            
            encoder = VanillaEncoder().to(device)
            model = VanillaAE().to(device)
            
            save_path = args.model_path + f'model_fold{fold}.pth'
            save_path_encoder = args.model_path + f'encoder_fold{fold}.pth'
            checkpoint = torch.load(save_path)
            checkpoint_encoder = torch.load(save_path_encoder)
            model.load_state_dict(checkpoint)
            encoder.load_state_dict(checkpoint_encoder)
            
            test_loss = 0.0
            corr_list = []
            model.eval()
            with torch.no_grad():
                np_embed = np.zeros((len(test_dataset),256))
                shRNA, cell, gene = [], [], []
                k = 0
                for i, data in enumerate(tqdm(testloader, 0)):
                    tensor_list = data
                    shRNA.extend(tensor_list[2])
                    cell.extend(tensor_list[3])
                    gene.extend(tensor_list[4])
                    embedding = encoder(tensor_list[0]).cpu().detach().numpy()
                    np_embed[k:k+embedding.shape[0],:] = embedding
                    output = model(tensor_list[0])
                    k = k+embedding.shape[0]
                    losses = criterion(output, tensor_list[1])
                
                    test_loss += losses.item()
                                    
                    if len(output) >= 2:
                        cor, p = scipy.stats.pearsonr(list(output.view(-1).cpu().detach().numpy()), list(tensor_list[1].view(-1).cpu().detach().numpy()))
                        corr_list.append(cor)
                        
                labels = {'shRNA' : shRNA, 'cell' : cell, 'gene' : gene}
                
                with open(f'./result/embedding/embedding{fold}.pkl', 'wb') as f:
                    pickle.dump(np_embed, f)
                with open(f'./result/embedding/labels{fold}.pkl', 'wb') as f:
                    pickle.dump(labels, f)
                    
                avg_corr = np.average(corr_list)
            
                print('test_loss of fold: %.4f, correlation of fold: %.4f' % (test_loss/len(testloader), avg_corr))
                print('-----------------------------------')
                results[fold] = test_loss/len(testloader)
                results_corr[fold] = avg_corr
            print(f'K-Fold CV Results of {k_folds} Folds')
            print('-----------------------------------')
            for f in range(fold+1):
                print('[Fold %d] test loss: %.4f, correlation: %.4f' % \
                    (f, results[f], results_corr[f]))
            print('%d folds test loss: %.4f' \
                % ((fold+1), sum(results.values())/len(results.items())))
            
            
    else:
        result_img_name = 'train'
        save_model_path = f'./model/AE/{cur_date}/'
        save_result_path = f'./result/AE/{cur_date}/'
    
        with open(args.train_exp_path,'rb') as f:
            train_exp = pickle.load(f)
        
        with open(args.test_exp_path,'rb') as f:
            test_exp = pickle.load(f)    
        
        train_lookup = pd.read_csv(args.train_lookup_path)
        test_lookup = pd.read_csv(args.test_lookup_path)
    
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
            loss_plot_list = []
            loss_plot = 0.0
        
            avg_losses = []
            avg_vlosses = []
        
            corr_plot_list = []
    
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_index)
    
            trainloader = get_loader(train_dataset, train_subsampler, batch_size)
            validloader = get_loader(train_dataset, valid_subsampler, batch_size)
            #import pdb; pdb.set_trace()
            encoder = VanillaEncoder().to(device)
            decoder = VanillaDecoder().to(device)
            model = VanillaAE().to(device)
        
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
            for epoch in range(1, num_epochs+1):            
                loss = 0.0
                vloss = 0.0
                losses = []
                vlosses = []
            
                corr_list = []
            
                model.train()
            
                for i, data in enumerate(tqdm(trainloader, 0)):                
                    tensor_list = data
                    output = model(tensor_list[0])              
                    losses = criterion(output, tensor_list[1])      
                    losses.backward()
                    optimizer.step()
                  
                    loss += losses.item()
                    loss_plot += losses.item()
                
                    if len(output) >= 2:
                        cor, p = scipy.stats.pearsonr(list(output.view(-1).cpu().detach().numpy()), list(tensor_list[1].view(-1).cpu().detach().numpy()))
                        corr_list.append(cor)
                
                model.eval()

                for i, data in enumerate(validloader, 0):
                    tensor_list = data
                    output = model(tensor_list[0])              
                    losses = criterion(output, tensor_list[1])

                    vloss += losses.item()
                    vlosses.append(losses.item())

                # avg(loss) per epoch
                avg_loss = np.average(losses.cpu().detach().numpy())
                avg_losses.append(avg_loss)
                avg_vloss = np.average(vlosses)
                avg_vlosses.append(avg_vloss)
            
                # Save model
                save_path =  save_model_path + f'/model_fold{fold}.pth'
                save_path_encoder = save_model_path + f'./encoder_fold{fold}.pth'
                torch.save(model.state_dict(), save_path)
                torch.save(encoder.state_dict(), save_path_encoder)            

                # Plot
                avg_loss_plot = loss_plot / (args.plot_every*(i+1))
                avg_corr = np.average(corr_list)
                #import pdb; pdb.set_trace()
                if epoch % args.print_every == 0:
                    print('Epoch %d / %d (%d%%) train loss: %.4f, valid loss: %.4f, correlation: %.4f' \
                        % (epoch, num_epochs, epoch / num_epochs * 100, avg_loss, avg_vloss, avg_corr))

                if epoch % args.plot_every == 0:
                    loss_plot_list.append(avg_loss_plot)
                    loss_plot = 0.0

        
            show_plot(loss_plot_list, args.plot_every, fold, save_path=save_result_path, file_name=result_img_name)
              
            print('-------- Starting testing --------')
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint)
        
            test_loss = 0.0
            corr_list = []
            model.eval()
            with torch.no_grad():
                        
                for i, data in enumerate(tqdm(testloader, 0)):
                    tensor_list = data
                    output = model(tensor_list[0])              
                    losses = criterion(output, tensor_list[1])

                    test_loss += losses.item()
                
                    if len(output) >= 2:
                        cor, p = scipy.stats.pearsonr(list(output.view(-1).cpu().detach().numpy()), list(tensor_list[1].view(-1).cpu().detach().numpy()))
                        corr_list.append(cor)
                avg_corr = np.average(corr_list)
            
                print('test_loss of fold: %.4f, correlation of fold: %.4f' % (test_loss/len(testloader), avg_corr))
                print('-----------------------------------')
                results[fold] = test_loss/len(testloader)
                results_corr[fold] = avg_corr
            print(f'K-Fold CV Results of {k_folds} Folds')
            print('-----------------------------------')
            for f in range(fold+1):
                print('[Fold %d] test loss: %.4f, correlation: %.4f' % \
                    (f, results[f], results_corr[f]))
            print('%d folds test loss: %.4f' \
                % ((fold+1), sum(results.values())/len(results.items())))
            print('Saved Model Path: %s' % save_model_path)
            print('-----------------------------------')