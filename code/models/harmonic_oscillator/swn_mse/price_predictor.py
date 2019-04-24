from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np


import matplotlib.pyplot as plt
import pandas as pd 
import json
import shutil
import pickle
import math

import torch
from torch.autograd import Variable
import timeit

import seaborn as sns
from model import Model

from models.harmonic_oscillator.mmd import mix_rbf_mmd2_and_ratio
from models.harmonic_oscillator.trading_utils import *



np.random.seed(111)
torch.manual_seed(111)

'''
TODO:
- generalize the params
'''

class PricePredictor(object):
    def __init__(self, config, dataset):
        # Initialize the model
        self._model = Model(config)
        self._config = config
        self._dataset = dataset
        # make a folder to keep all info about this model
        if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
        if not os.path.exists('saved_models/'+self._config.model_name):
                    os.makedirs('saved_models/'+self._config.model_name)
        self.save_path = 'saved_models/'+self._config.model_name


            
    @staticmethod
    def adjust_kd(epoch, total_epoch, init_kd, end_kd):
        if (epoch > total_epoch):
            return end_kd
        return end_kd + (init_kd - end_kd) * ((math.cos(0.5 * math.pi * float(epoch) / total_epoch)))

    @staticmethod
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.net.eval()
        model.opt.zero_grad()
        # imagine batch size is 10, seq_len is 1000 and 1 channel
        bs = config.batch_size
        seq_len = config.input_seq_length
        channels = 1
        x = np.ones([bs, seq_len, channels])
        # for pytorch convs it is [batch_size, channels, width, height]
        x = np.einsum('ijk->jik', x)
        y = x.copy()
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        y = Variable(torch.from_numpy(y).float())
        mask_x = Variable(torch.from_numpy(np.ones([seq_len, bs])).float())
        # self._model.net.eval()
        _, _, pars = model.net([x,y, mask_x])
        mu = pars[0]
        grad=torch.zeros(mu.size())
        # imagien only 1 output in the time axis has a gradient
        grad[-1, :, :] = 1.0
        mu.backward(gradient=grad)
        # see what this gradient is wrt the input
        zeros=np.where(x.grad.data!=0)
        RF = len(set(zeros[0]))
        print('RF: ', RF)
        return RF

    def evaluate(self, x, y, mask):
        x = np.einsum('ijk->jik', x)
        y = np.einsum('ijk->jik', y)
        self._model.net.eval()
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()
        loss, kld_loss, outputs = self._model.net([x,y, mask]);
        total_loss = loss - kld_loss 
        total_loss = total_loss.item()

        return total_loss, loss.item(), kld_loss.item(), outputs


    def _train(self):
        ################## build model ##############################
        t1 = time.time()
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        print('--------------------------------------------------------------------')
        print('NOTE: the receptive field is ', receptive_field, ' and your input is ', self._config.input_seq_length)
        print('--------------------------------------------------------------------')
        t2 = time.time()
        print('Finished building the model: ' + str(t2-t1) +' sec \n')
        # ################# get data ################################
        self._dataset.prepare_data(self._config, shuffle = self._config.shuffle)
        print('train size: ', np.shape(self._dataset._train_x))
        print('test size: ', np.shape(self._dataset._test_x))
        t3 = time.time()
        print('Finished preparing the dataset: ' + str(t3-t2) +' sec \n')
        ################### prepare log structure ##################
        log_folder = 'saved_models/'+self._config.model_name+'/logs/'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        else:
            for root, dirs, files in os.walk(log_folder):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        
        steps_per_epoch = len(self._dataset._train_y)//self._config.batch_size
        ############## initialize all the stuff ##################
        kld_step = 0.000000001
        kld_weight = kld_step
        
        print('train size: ', np.shape(self._dataset._train_x))
        print('test size: ', np.shape(self._dataset._test_x))
        

        t = timeit.default_timer()
        mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        test_mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        test_mask[receptive_field-1:, :] = 1
        for epoch in range(1, int(self._config.epochs+1)):
            costs = []
            test_costs = []
            self._model.net.train()
            loss_sum = 0
            kld_loss_sum = 0
            logp_loss_sum = 0

            test_loss_sum = 0
            test_kld_loss_sum = 0
            test_logp_loss_sum = 0

            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                self._model.opt.zero_grad()
                x, y = self._dataset.get_batch(self._config.batch_size)
                x = np.einsum('ijk->jik', x)
                y = np.einsum('ijk->jik', y)
                x = Variable(torch.from_numpy(x).float())
                y = Variable(torch.from_numpy(y).float())
                mask_x = Variable(torch.from_numpy(mask).float())
                loss, kld_loss, _ = self._model.net([x,y, mask_x])
                # if epoch<2:
                #     total_loss = loss
                # else:
                total_loss = loss - kld_loss * kld_weight
                total_loss.backward();
                total_loss += total_loss.item()
                kld_loss_sum += kld_loss.item()
                logp_loss_sum += loss.item()
                
                costs.append(total_loss.item())     
                # print('HHHHAAAAI', costs)
                # torch.nn.utils.clip_grad_norm_(self._model.net.parameters(), 0.1, 'inf')
                self._model.opt.step()
                loss_sum += total_loss;


                ################ occasionally show the (test) performance #############  
                # if train_step % self._config.print_every == 0:

            x, y = self._dataset.get_batch(self._config.batch_size, test = True)
            test_cost, test_nll, test_kld_loss, test_pars = self.evaluate(x, y, self._model.net, test_mask)


            test_costs.append(test_cost)
            test_loss_sum += test_cost
            test_kld_loss_sum += test_kld_loss
            test_logp_loss_sum += test_nll


            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f, logp_loss:%f, kld_loss: %f,\
             \n                       test_loss: %f, test_logp_loss:%f, test_kld_loss: %f, kld_weight: %f' % (
                s-t, epoch, self._config.epochs, train_step, steps_per_epoch,
                loss_sum / train_step, logp_loss_sum/train_step, -kld_loss_sum/train_step,
                test_loss_sum / train_step, test_logp_loss_sum/train_step, -test_kld_loss_sum/train_step,
                kld_weight
                )
            print(log_line)
            
            # print("- batch {0:.1f} | {1:.8f} | {2:.8f} | ".format(train_step, 
            #                                         np.mean(costs),
            #    
            # print('HAOOOIII', costs)
            # print(train_step)
            # print(np.mean(costs))
            # print(np.mean(test_costs))
            # print(-kld_loss_sum/train_step)
            # print(-test_kld_loss_sum/train_step)
            # print(train_step, np.mean(costs), np.mean(test_costs), -kld_loss_sum/train_step, -test_kld_loss_sum/train_step )                                     np.mean(test_costs) ))
            log.append([train_step, np.mean(costs), np.mean(test_costs), -kld_loss_sum/train_step, -test_kld_loss_sum/train_step ])
            # print([train_step, np.mean(costs), np.mean(test_costs), -kld_loss_sum/train_step, -test_kld_loss_sum/train_step ])

            # adjust the KL weight and also the learning rate
            print('Adjusting kld weight and learning rate')
            kld_weight = self.adjust_kd(epoch, 1000, kld_step, 0.001)
            print('KL weight: ', kld_weight)
            # self.adjust_lr(self._model.opt, epoch, 1000, self._config.learning_rate, 0.)

            if epoch%10==0:
                state = {
                    'epoch': epoch,
                    'state_dict': self._model.net.state_dict(),
                    'optimizer': self._model.opt.state_dict(),       
                    'kld_weight':kld_weight             
                    }

                torch.save(state, 'saved_models/'+self._config.model_name+str(epoch)+'.pth.tar')
                print('Saved model of epoch ', epoch)
                # dump confg json to keep track of model properties
                with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
                    json.dump(vars(self._config), fp)
                with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
                    pickle.dump( self._config, fp )
            # write results to a log file
            log = pd.DataFrame(np.array(log), columns = ['step', 'train loss', 'test loss', 'kld train', 'kld test'])
            log.to_csv('saved_models/'+self._config.model_name+'/epoch'+str(epoch)+'.csv')
            


    def _validate(self, steps = 5, epoch=200):
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)
        ###### first build the graph again ########
        x, y  = self._dataset.get_validation_set()
        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1

        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.net.load_state_dict(state['state_dict'])

        MSE, KLloss, lowerbound = [],[],[]
        kld_weight = self._config.kld_step
        for i in range(1, epoch):
            kld_weight = self.adjust_kd(i, self._config.kld_epochs, self._config.kld_step, self._config.kld_max)
        print('kld_weight: ', kld_weight)

        for step in range(steps):
            print('step: ', step)
            test_cost, test_nll, test_kld_loss, test_pars = self.evaluate(x[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], self._model._model.net, mask)
            # print(np.shape(test_pars[0].detach().numpy()))
            test_pred = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:]))).float()
            x = np.concatenate([x, test_pred], axis = 1)
            # MLE.append(ll)
            MSE.append(test_nll)
            KLloss.append(test_kld_loss)
            lowerbound.append(test_nll - kld_weight*test_kld_loss)

        tars = y[:,-steps:, 0]
        preds = x[:, -steps:, 0]
        
        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]]
        _, mmd, that = mix_rbf_mmd2_and_ratio(Variable(torch.from_numpy(preds)).float(), 
                                              Variable(torch.from_numpy(tars)).float(), sigma_list)

        print('MMD : ', mmd.item(), ' THAT: ', that.item())
        print('MSE', np.sum(MSE))
        print('KLloss', np.sum(KLloss))
        print('lowerbound', np.sum(lowerbound))
        
        MSE = np.mean(np.power(preds - tars, 2), axis=0)
        print('MSE@1: ', MSE[0])
        print('MSE@total: ', np.sum(MSE))
        f = open('results_'+self._config.model_name+'.txt', 'w')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('THAT: '+str(that.item())+'\n')
        f.write('kld@1: '+str(np.sum(KLloss[0]))+'\n')
        f.write('kld@total: '+str(np.sum(KLloss))+'\n')
        f.write('MSE@1: '+str(np.sum(MSE[0]))+'\n')
        f.write('MSE@total: '+str(np.sum(MSE))+'\n')
        f.write('elbo@1: '+str(np.sum(lowerbound[0]))+'\n')
        f.write('elbo@total: '+str(np.sum(lowerbound))+'\n')
        f.close()

   
    def _make_figs(self, steps = 5, epoch=500):

        if not os.path.exists('images/'):
            os.makedirs('images/')

        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)
        x, y = self._dataset.get_validation_set()
        
        x = x[0:1,:,:]
        y = y[0:1,:,:]

        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.net.load_state_dict(state['state_dict'])

        seeds = 100
        df = pd.DataFrame([])
        df['y'] = y[0,:,0]

        for seed in range(seeds):
            print('roll-out nr.: ', seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            X = x[:, :receptive_field, :].copy()
            for step in range(steps):
                _, _, _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
                loc = test_pars[0].detach().numpy()[-1:,:, :]
                test_pred = np.einsum('ijk->jik',loc)
                X = np.concatenate([X, test_pred], axis = 1)
            df[str(seed)] = X[0,1:,0]
            plt.plot(X[0,1:,0], alpha=0.5, c='b')
        plt.plot(y[0,:,0], alpha=1., c='r')
        plt.show()

        np.random.seed(111)
        torch.manual_seed(111)

        df.to_csv('images/swn_mse_samples.csv')

        df = pd.DataFrame([])
        df['y'] = y[0,:,0]

        X = x[:, :receptive_field, :].copy()
        for step in range(steps):
            _, _,_,test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
            loc = test_pars[0].detach().numpy()[-1:,:, :]
            test_pred = np.einsum('ijk->jik',loc)
            X = np.concatenate([X, test_pred], axis = 1)
        df['x'] = X[0,1:,0]
        plt.plot(X[0,1:,0], alpha=0.5, c='b')
        plt.plot(y[0,:,0], alpha=0.5, c='r')
        plt.show()

        df.to_csv('images/swn_mse_prediction.csv')