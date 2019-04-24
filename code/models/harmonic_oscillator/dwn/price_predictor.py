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
import timeit
import seaborn as sns

import torch
from torch.autograd import Variable



from model import Model
from models.harmonic_oscillator.mmd import mix_rbf_mmd2_and_ratio, mix_rbf_mmd2




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
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.gen.eval()
        model.gen_opt.zero_grad()
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
        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [0.005, 0.01, 0.05, 0.1, 0.5, 1., 5.]]
        mmd = mix_rbf_mmd2(pred[0][:,-10:, 0], y[:,-10:, 0], sigma_list)
        return loss.item(), mmd.mean().item(), pred[0]


    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self._config.batch_size, 1)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + (1 - alpha) * fake_data

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self._model.dis(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs= torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients[-1,:,:]
        gradients = gradients.view(self._config.batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1

        return gradient_penalty

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
        self._dataset.prepare_data(self._config, shuffle = self._config.shuffle, skip = self._config.input_seq_length - receptive_field)
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
        mask = np.zeros([69, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        mask_dis = np.zeros([69, self._config.batch_size*5])
        mask_dis[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())
        mask_dis = Variable(torch.from_numpy(mask_dis).float())

        for epoch in range(1, int(self._config.epochs+1)):

            loss_sum = 0
            test_loss_sum = 0

            Dloss = 0
            GP = 0
            Gloss = 0

            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                

                for p in self._model.dis.parameters():  
                    p.requires_grad = True

                
                for _ in range(self._config.critic_updates):
                    x, y = self._dataset.get_batch(self._config.batch_size)
                    assert np.max(x[:,1,:] - y[:,0,:]) == 0
                    
                    x = np.einsum('ijk->jik', x)
                    y = np.einsum('ijk->jik', y)

                    x = Variable(torch.from_numpy(x).float())
                    trueY = Variable(torch.from_numpy(y).float())

                    loss, fakeY = self._model.gen([x, trueY, mask])

                    for i in range(5):
                        fakePile = torch.cat((x[i+1:i+receptive_field,:,:], fakeY[0][i+receptive_field-1:i+receptive_field,:,:]))
                        if i == 0:
                            fakeStack = fakePile
                        else: 
                            fakeStack = torch.cat((fakeStack, fakePile), dim=1)
                    
                    for i in range(5):
                        truePile = torch.cat((x[i+1:i+receptive_field,:,:], trueY[i+receptive_field-1:i+receptive_field,:,:]))
                        if i == 0:
                            trueStack = truePile
                        else: 
                            trueStack = torch.cat((trueStack, truePile), dim=1)

                    loss_sum+=loss.item()
                    costs.append(loss.item())

                    fakeScore = torch.squeeze(self._model.dis(fakeStack), -1)
                    trueScore = torch.squeeze(self._model.dis(trueStack), -1)

                    d_loss = (mask_dis*trueScore).mean() - (mask_dis*fakeScore).mean()
                    
                    Dloss+=d_loss.item()

                    gradient_penalty = self.calc_gradient_penalty(trueY[:receptive_field, :,:], fakeY[0][:receptive_field, :,:])
                    GP+= gradient_penalty.item()

                    self._model.dis.zero_grad()
                    d_loss_p = d_loss + gradient_penalty
                    d_loss_p.backward()
                    self._model.dis_opt.step()

                
                for p in self._model.dis.parameters():  
                    p.requires_grad = False

                x, y = self._dataset.get_batch(self._config.batch_size, test=True)
                x = np.einsum('ijk->jik', x)
                y = np.einsum('ijk->jik', y)

                x = Variable(torch.from_numpy(x).float())
                trueY = Variable(torch.from_numpy(y).float())
                
                loss, fakeY = self._model.gen([x,trueY, mask])
                loss_sum+=loss.item()
                
                for i in range(5):
                    fakePile = torch.cat((x[i+1:i+receptive_field,:,:], fakeY[0][i+receptive_field-1:i+receptive_field,:,:]))
                    if i == 0:
                        fakeStack = fakePile
                    else: 
                        fakeStack = torch.cat((fakeStack, fakePile), dim=1)

                fakeScore = torch.squeeze(self._model.dis(fakeStack), -1)

                g_loss = (mask_dis*fakeScore).mean()
                Gloss+=g_loss.item()
                self._model.gen.zero_grad()
                g_loss.backward()
                self._model.gen_opt.step()

            ################ occasionally show the (test) performance #############  
            x, y = self._dataset.get_batch(self._config.batch_size, test = False)
            test_cost, mmd, _ = self.evaluate(x, y, test_mask_dis, test_mask_gen)
            test_loss_sum += test_cost

            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f, test_loss: %f gloss: %f, dloss: %f gp: %f' % (
                s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum,
                Gloss/train_step, Dloss/(self._config.critic_updates*train_step), GP/(self._config.critic_updates*train_step) )
            print(log_line)
            

            log.append([train_step, loss_sum / train_step, test_loss_sum, mmd, Gloss/train_step, 
                Dloss/(self._config.critic_updates*train_step), GP/(self._config.critic_updates*train_step)])
           
            if epoch%5==0:
                state = {
                    'epoch': epoch,
                    'gen_state_dict': self._model.gen.state_dict(),
                    'dis_state_dict': self._model.dis.state_dict(),
                    'gen_optimizer': self._model.gen_opt.state_dict(),
                    'dis_optimizer': self._model.dis_opt.state_dict()                    
                    }

                torch.save(state, 'saved_models/'+self._config.model_name+str(epoch)+'.pth.tar')
                print('Saved model of epoch ', epoch)
                # dump confg json to keep track of model properties
                with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
                    json.dump(vars(self._config), fp)
                with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
                    pickle.dump( self._config, fp )
            # write results to a log file
            log = pd.DataFrame(np.array(log), columns = ['step', 'train_ll', 'test_ll', 'mmd', 'gloss', 'dloss', 'gp'])
            log.to_csv('saved_models/'+self._config.model_name+'/epoch'+str(epoch)+'.csv')
            





    def _validate(self, steps = 5, epoch=500):
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)

        x, y = self._dataset.get_validation_set()
        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])
        X = x[:, :receptive_field, :].copy()

        MSE = []
        for step in range(steps):
            # print('step: ', step)
            test_cost, _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
            test_pred = np.einsum('ijk->jik',test_pars.detach().numpy()[-1:,:, :])
            X = np.concatenate([X, test_pred], axis = 1)
            print('shape x', np.shape(X))
            MSE.append(test_cost)

        preds = X[:, -steps:, 0]
        tars = y[:,-steps:,0]

        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]]
        _, mmd, that = mix_rbf_mmd2_and_ratio(Variable(torch.from_numpy(preds)).float(), 
                                              Variable(torch.from_numpy(tars)).float(), sigma_list)


        print('MMD : ', mmd.item(), ' THAT: ', that.item())
        print('MSE@1: ', MSE[0])
        print('MSE@10: ', np.sum(MSE))

        f = open('results_'+self._config.model_name+'.txt', 'w')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('That: '+str(that.item())+'\n')
        f.write('MSE1: '+str(MSE[0])+'\n')
        f.write('MSEtotal: '+str(np.sum(MSE))+'\n')
        f.close()



   

    