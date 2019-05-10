from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse
import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt


import pandas as pd 
import json
import shutil
import pickle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

import timeit


import seaborn as sns
from model import Model

from models.mnist.mmd import mix_rbf_mmd2_and_ratio

from models.mnist.t_sne import tsne


np.random.seed(111)
tf.set_random_seed(111)
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
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.gen.eval()
        model.gen_opt.zero_grad()
        # imagine batch size is 10, seq_len is 1000 and 1 channel
        bs = config.batch_size
        seq_len = 16
        channels = 28
        x = np.ones([bs, seq_len, channels])
        # for pytorch convs it is [batch_size, channels, width, height]
        x = np.einsum('ijk->jik', x)
        y = x.copy()
        x = Variable(torch.from_numpy(x).float(), requires_grad=True)
        y = Variable(torch.from_numpy(y).float())
        mask_x = Variable(torch.from_numpy(np.ones([seq_len, bs])).float())
        # self._model.net.eval()
        _,  pars = model.gen([x,y, mask_x])
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

    def evaluate(self, x, y, mask_gen):
        x = np.einsum('ijk->jik', x)
        y = np.einsum('ijk->jik', y)
        self._model.gen.eval()
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()
        loss, pred = self._model.gen([x,y, mask_gen]);
    
        return loss.item(), pred[0]

    def _train(self):
        ################## build model ##############################
        t1 = time.time()
        self._model._build_model()
        self._model.gen
    
        receptive_field = self.get_receptive_field(self._model, self._config)
        print('--------------------------------------------------------------------')
        print('NOTE: the receptive field is ', receptive_field, ' and your input is ', 16)
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

        t = timeit.default_timer()
        mask = np.zeros([27, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())
        

        for epoch in range(1, int(self._config.epochs+1)):
            self._model.gen.train()
            self._dataset.reset(shuffle = self._config.shuffle)
            loss_sum = 0
            test_loss_sum = 0
            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                print('train step: ', train_step)
                
                x, _ = self._dataset.get_batch(self._config.batch_size)
                # m,n = x.shape
                # out = np.ones((m,2*n),dtype=x.dtype)
                # out[:,::2] = x
                # x = out

                # m,n = x.shape
                # out = np.ones((2*m,n),dtype=x.dtype)
                # out[::2, :] = x


                plt.imshow(x[0,:,:], cmap='gray')
                plt.Circle((17, 14), 1, color='w')
                plt.show()
                
                y = x[:,1:,:]
                x = x[:,:-1,:]

                
                x = np.einsum('ijk->jik', x)
                y = np.einsum('ijk->jik', y)

                x = Variable(torch.from_numpy(x).float())
                y = Variable(torch.from_numpy(y).float())

                loss, _ = self._model.gen([x,y, mask])

                loss_sum+=loss.item()

                self._model.gen.zero_grad()
                loss.backward()
                self._model.gen_opt.step()

            ################ occasionally show the (test) performance #############  
            x, y = self._dataset.get_batch(self._config.batch_size, test = True)
            y=x[:,1:,:]
            x=x[:,:-1,:]
            test_cost, mmd, _ = self.evaluate(x, y, test_mask)
            self._model.gen.train()              
            test_loss_sum += test_cost

            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f,  test_loss: %f' % (
               s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum )
            print(log_line)
        
            log.append([train_step, loss_sum/train_step, test_loss_sum])
            if epoch%10==0:
                state = {
                    'epoch': epoch,
                    'state_dict': self._model.gen.state_dict(),
                    'optimizer': self._model.gen_opt.state_dict()                    
                    }

                torch.save(state, 'saved_models/'+self._config.model_name+str(epoch)+'.pth.tar')
                print('Saved model of epoch ', epoch)
                # dump confg json to keep track of model properties
                with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
                    json.dump(vars(self._config), fp)
                with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
                    pickle.dump( self._config, fp )
            # write results to a log file
            log = pd.DataFrame(np.array(log), columns = ['step', 'train ll', 'test ll'])
            log.to_csv('saved_models/'+self._config.model_name+'/epoch'+str(epoch)+'.csv')
            

    def _validate(self, epoch=500):
        sns.set()
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)
        x = self._dataset.get_validation_set()
        # x = x[:100,:,:]
        y = x.copy()


        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict'])

        X = x[:, :receptive_field, :].copy()        

        loss = []
        for step in range(28 - receptive_field):
            print('step: ', step)
            test_cost, test_pars = self.evaluate(X[:, step:step+receptive_field, :], x[:, step+1:step+receptive_field+1, :], mask_gen)
            test_pred = np.einsum('ijk->jik',test_pars.detach().numpy()[-1:,:,:])
            X = np.concatenate([X, test_pred], axis = 1)
            loss.append(test_cost)
        tars = y
        preds = X

        pixels_real = np.reshape(0.5*(1+y[:,16:,:]), (1, np.shape(y)[0]*28*12))

        pixels_fake = np.reshape(0.5*(1+X[:,16:,:]), (1, np.shape(X)[0]*28*12))
        a = np.histogram(pixels_real, bins=50, range = (0,1))
        a = a[0]/np.sum(a[0])
        print(np.sum(a))

        # plt.bar(np.arange(0+0.5/len(a), 1+0.5/len(a), 1/len(a)), a, width = 1/len(a)*0.99)
        # # plt.ylim(-2, 2)
        # plt.savefig('imagses/mnist_wn_pixeldist_real.pdf')
        # plt.show()
        
        b =  np.histogram(pixels_fake, bins=50, range = (0,1))
        b = b[0]/np.sum(b[0])
        plt.bar(np.arange(0+0.5/len(a), 1+0.5/len(a), 1/len(a)), a-b, width = 1/len(a)*0.99)
        plt.ylim(-.025, .075)
        plt.savefig('images/mnist_wn_pixeldist.pdf')
        plt.show()



        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in np.arange(5,10,0.5)]
        _, mmd, that = mix_rbf_mmd2_and_ratio(torch.reshape(Variable(torch.from_numpy(X)).float(), (np.shape(x)[0], np.shape(x)[1]*28)), 
                            torch.reshape(Variable(torch.from_numpy(y)).float(), (np.shape(x)[0], np.shape(x)[1]*28)), sigma_list)
        

        if not os.path.exists('results/'):
                    os.makedirs('results/')
        f = open('results/results_'+self._config.model_name+'.txt', 'w')
        f.write('epoch: '+str(epoch)+'\n')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('MSE1: '+str(loss[0])+'\n')
        f.write('MSEtotal: '+str(np.sum(loss))+'\n')
        f.close()
        f = open('results/results_'+self._config.model_name+'.txt', 'r')
        for line in f:
            print(line)

    def _make_figs(self,  epoch=500):
        if not os.path.exists('images/'):
            os.makedirs('images/')

        self._model._build_model()
        receptive_field = 16
        mask = np.zeros([receptive_field, 1])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())

        self._dataset.prepare_data(self._config)

        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict'])

        df = pd.DataFrame([])
        for k in range(10):
            x = self._dataset.get_digit_set(k)
            X = x[:1, :receptive_field, :].copy()
            for step in range(28 - receptive_field):
                print('step: ', step)   
                _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], x[:, step:step+receptive_field, :], mask_gen)
                test_pred = np.einsum('ijk->jik',test_pars.detach().numpy()[-1:,:, :])
                X = np.concatenate([X, test_pred], axis = 1)
            

            df[str(k)+'_real'] = np.reshape(x[0,:,:], (28*28))
            df[str(k)+'_fake'] = np.reshape(X[0,:,:], (28*28))

            plt.imshow(x[0,:,:])
            plt.show()
            plt.imshow(X[0,:,:])
            plt.show()
            plt.imshow(x[0,:,:] - X[0,:,:])
            plt.show()
        df.to_csv('images/digits.csv')
    

    def _tsne(self,  epoch=500):
        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict'])
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)

        for digit in range(10):
            x = self._dataset.get_digit_set(digit)
            x = x[:200,:,:]
            print(np.shape(x))
            self._config.batch_size = len(x)
            mask = np.zeros([receptive_field, self._config.batch_size])
            mask[receptive_field-1:, :] = 1
            mask = Variable(torch.from_numpy(mask).float())
            X = x[:, :receptive_field, :]
            for step in range(28 - receptive_field):
                _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], x[:, step+1:step+receptive_field+1, :], mask)
                test_pred = np.einsum('ijk->jik',test_pars.detach().numpy()[-1:,:, :])
                X = np.concatenate([X, test_pred], axis = 1)

            if digit == 0:
                DataX = np.reshape(x, [len(x), 28*28])
                FakeX = np.reshape(X, [len(X), 28*28])
                labels = np.ones(self._config.batch_size)*digit
            else:
                DataX = np.concatenate( [DataX, np.reshape(x, [len(x), 28*28])], axis=0)
                FakeX = np.concatenate( [FakeX, np.reshape(X, [len(X), 28*28])], axis=0)
                labels = np.concatenate( [labels, np.ones(self._config.batch_size)*digit])
        pickle.dump( labels, open( "labels.p", "wb" ) )

        X = np.concatenate( [FakeX, DataX], axis=0)
        Y = tsne(X, 2, 50, 20.0)

        pickle.dump( Y[:len(FakeX),:], open( "images/FakeY.p", "wb" ) )
        pickle.dump( Y[len(FakeX):,:], open( "images/RealY.p", "wb" ) )

