from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse
import json
import shutil
import pickle
import math
import timeit


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import torch
from torch.autograd import Variable

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
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.gen.eval()
        model.gen_opt.zero_grad()
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
        _,  pars = model.gen([x,y, mask_x])
        mu = pars[0]
        grad=torch.zeros(mu.size())
        # imagien only 1 output in the time axis has a gradient
        grad[-1, :, :] = 1.0
        mu.backward(gradient=grad)
        zeros=np.where(x.grad.data!=0)
        RF = len(set(zeros[0]))
        print('RF: ', RF)
        return RF

    def evaluate(self, x, y,  mask_gen):
        x = np.einsum('ijk->jik', x)
        y = np.einsum('ijk->jik', y)
        self._model.gen.eval()
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()
        loss, pred = self._model.gen([x,y, mask_gen]);
        return loss.item(), pred


    def _train(self):
        ################## build model ##############################
        t1 = time.time()
        self._model._build_model()

        self._model.gen
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
        
        steps_per_epoch = len(self._dataset._train_y)//(self._config.batch_size)
        ############## initialize all the stuff ##################

        t = timeit.default_timer()
        #  mask the input to make conditional on full receptive field
        mask = np.zeros([69, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())

        tmask = np.zeros([69, self._config.batch_size*5])
        tmask[receptive_field-1:, :] = 1
        tmask = Variable(torch.from_numpy(tmask).float())

        for epoch in range(1, int(self._config.epochs+1)):
            loss_sum = 0
            test_loss_sum = 0
            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                
                x, y = self._dataset.get_batch(self._config.batch_size)
                x = np.einsum('ijk->jik', x)
                y = np.einsum('ijk->jik', y)
                x = Variable(torch.from_numpy(x).float())
                trueY = Variable(torch.from_numpy(y).float())
                loss, pred = self._model.gen([x,trueY, mask])
                print(np.shape(pred[0]))
                boe
                loss_sum+=loss.item()
                self._model.gen.zero_grad()
                loss.backward()
                self._model.gen_opt.step()
            
            ################ occasionally show the (test) performance #############  
            x, y = self._dataset.get_batch(self._config.batch_size*5, test = True)
            test_cost, _ = self.evaluate(x, y, tmask)
            test_loss_sum += test_cost

            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f,  test_loss: %f' % (
               s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum )
            print(log_line)
            
            log.append([train_step, loss_sum / train_step, test_loss_sum])
            if epoch%10==0:
                state = {
                    'epoch': epoch,
                    'state_dict_gen': self._model.gen.state_dict(),
                    'optimizer_gen': self._model.gen_opt.state_dict(),              
                    
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
            

    def _validate(self, steps = 5, epoch=500):
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._dataset.prepare_data(self._config)
        x, y = self._dataset.get_validation_set()
        X = x[:, :receptive_field, :].copy()

        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])

        LL= []
        for step in range(steps):
            # print('step: ', step)
            ll, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
            LL.append(ll)
            loc = test_pars[0].detach().numpy()[-1:,:, :]
            if False:#self._config.loss == 'Gaussian':
                locs = Variable(torch.from_numpy(np.einsum('ijk->jik',loc))).float()
                logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
                test_pred = Dist.sample()
            else:
                test_pred = np.einsum('ijk->jik',loc)
            X = np.concatenate([X, test_pred], axis = 1)
            print('shape x', np.shape(X))
        
        preds = X[:, -steps:, 0]
        tars = y[:,-steps:,0]
        MSE = np.mean((preds-tars)**2, axis=0)

        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]]
        _, mmd, that = mix_rbf_mmd2_and_ratio(Variable(torch.from_numpy(preds)).float(), 
                                              Variable(torch.from_numpy(tars)).float(), sigma_list)


        if not os.path.exists('results/'):
            os.makedirs('results/')
        f = open('results/results_'+self._config.model_name+'.txt', 'w')
        f.write('epoch: '+str(epoch)+'\n')
        f.write('steps: '+str(steps)+'\n')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('THAT: '+str(that.item())+'\n')
        f.write('LL1: '+str(LL[0])+'\n')
        f.write('LL5: '+str(LL[4])+'\n')
        f.write('LL10: '+str(LL[9])+'\n')
        f.write('LLtotal: '+str(np.sum(LL))+'\n')
        f.write('MSE1: '+str(MSE[0])+'\n')
        f.write('MSEtotal: '+str(np.sum(MSE))+'\n')
        f.close()
        f = open('results/results_'+self._config.model_name+'.txt', 'r')
        for line in f:
            print(line)




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
        self._model.gen.load_state_dict(state['state_dict_gen'])

        seeds = 100
        df = pd.DataFrame([])
        df['y'] = y[0,:,0]

        for seed in range(seeds):
            print('roll-out nr.: ', seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            X = x[:, :receptive_field, :].copy()
            for step in range(steps):
                _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
                loc = test_pars[0].detach().numpy()[-1:,:, :]
                locs = Variable(torch.from_numpy(np.einsum('ijk->jik',loc))).float()
                logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
                test_pred = Dist.sample()
                X = np.concatenate([X, test_pred], axis = 1)
            df[str(seed)] = X[0,1:,0]
            plt.plot(X[0,1:,0], alpha=0.5, c='b')
        plt.plot(y[0,:,0], alpha=1., c='r')
        plt.show()

        np.random.seed(111)
        torch.manual_seed(111)

        df.to_csv('images/wn_'+self._config.loss+'_samples.csv')

        df = pd.DataFrame([])
        df['y'] = y[0,:,0]

        X = x[:, :receptive_field, :].copy()
        for step in range(steps):
            _, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask)
            loc = test_pars[0].detach().numpy()[-1:,:, :]
            test_pred = np.einsum('ijk->jik',loc)
            X = np.concatenate([X, test_pred], axis = 1)
        df['x'] = X[0,1:,0]
        plt.plot(X[0,1:,0], alpha=0.5, c='b')
        plt.plot(y[0,:,0], alpha=0.5, c='r')
        plt.show()

        df.to_csv('images/wn_'+self._config.loss+'_prediction.csv')
        

    def _backtest(self, samples_per_tick=50, view = 1, risk_view=1, epoch=1000, alpha=5):
        t1 = time.time()
        self._dataset.prepare_data(self._config)
        _, y = self._dataset.get_validation_set()
        # something I have to because of coding decision in utils
        run_index = 1 # choose 0, 1, or 2 for validation
        x = y[run_index*10:10*(run_index+1), :,:]
        y = y[1+run_index*10:1+10*(run_index+1), :,:]
        print(np.shape(x), np.shape(y))

        bs = np.shape(x)[0]
        self._config.batch_size = bs
        runs = bs
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        
        ticks_per_run = np.shape(x)[1] - (receptive_field + view)
        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())
        
        # initial situation
        have_state = np.full((bs), False)

        actions = np.full((bs,ticks_per_run), 'hold')
        transaction_cost = 0.00
        D = (1+transaction_cost)/(1-transaction_cost)
        print('Discount factor :' , D)
        var = np.zeros(np.shape(actions))
        es = np.zeros(np.shape(actions))
        es_unc = np.zeros(np.shape(actions))
        observed_prices = np.zeros(np.shape(actions))
        t2 = time.time()
        print('init done in :', str(t2-t1), 'sec')

        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])

        d = 0
        for tick in range(ticks_per_run):
            x0 = x[:, tick+receptive_field-1, 0]
            context  = x[:, tick:tick+receptive_field,:]
            assert (np.sum(np.subtract(x0, context[:,-1, 0]))) == 0
            print('tick: ', str(tick) , '/', str(ticks_per_run))
            # fill this with sampled futures
            future_prices = np.zeros([bs, view, samples_per_tick])

            for sample in range(samples_per_tick):
                context_ = context.copy()
                np.random.seed(sample)
                torch.manual_seed(sample)
                # free run the model to predict multiple steps
                for step in range(view):  
                    _,  test_pars = self.evaluate(context_[:, step:, :], context_[:, step:, :], mask)
                    loc = test_pars[0].detach().numpy()[-1:,:, :]
                    if self._config.loss == 'Gaussian':
                        locs = Variable(torch.from_numpy(np.einsum('ijk->jik',loc))).float()
                        logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                        Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
                        pred = Dist.sample()
                    else:
                        pred = torch.from_numpy(np.einsum('ijk->jik',loc))
                    context_ = np.concatenate([context_, pred], axis = 1)
                
                future_prices[:, :, sample] = context_[:, receptive_field:, d]

            expected_future = np.mean(future_prices, axis=-1)
            
            var[:, tick] = value_at_risk(future_prices[:, risk_view-1, :])
            es[:, tick] = expected_shortfall(future_prices[:, risk_view-1, :])
            es_unc[:, tick] = unconditional_expected_shortfall(future_prices[:, risk_view-1, :])

            next_price = x[:, tick+receptive_field, d]
            observed_prices[:, tick] = next_price

        
            for run in range(runs):
                # for each example in the batch, find trading action
                if not have_state[run]:
                    # check when and if there is a profitable sell moment
                    t_sell = get_next_or_none(tau for tau, m in enumerate(expected_future[run,:]) if m > x0[run]*D)

                    if t_sell is not None:
                        # check if until that sell moment arrives, there is a better buy moment
                        if (t_sell == 0 or expected_future[run,0:t_sell].min() >= x0[run]):  
                            have_state[run] = True
                            actions[run,tick] = 'buy'
                            buying_price = x0[run]
                            # plt.plot(expected_future[run,:])
                            # plt.axhline(next_price[run])
                            # plt.show()
                else:
                    # check if there is a moment when buying a new share is cheaper than keeping this one
                    t_buy = get_next_or_none(tau for tau, m in enumerate(expected_future[run,:]) if m < x0[run])
                    # check if until that moment arrives, there is a better sell moment

                    if t_buy is not None:
                        if (t_buy == 0 or expected_future[run,0:t_buy].max() <= x0[run]):
                            have_state[run] = False
                            actions[run,tick] = 'sell'

                

        # observed_returns = self.denorm(Y[:, receptive_field-1:receptive_field+ticks_per_run-1, 0])
        violations = observed_prices-var
        np.putmask(violations, violations>=0, np.ones(np.shape(observed_prices)))
        np.putmask(violations, violations<0, 2*np.ones(np.shape(observed_prices)))
        violations -= 1
        T1 = np.sum(violations)
        T0 = (runs*ticks_per_run)-T1
        pihat = T1/(T0+T1)
        lr = likelihood_ratio(pihat, alpha/100., T1, T0)
        ratio = np.sum(violations*observed_prices / es)

        R = []
        P = []
        for run in range(runs):
            roi, profit = compute_roi(x[run,receptive_field-1:,d], actions[run,:], transaction_cost)
            R.append(roi)
            P.append(profit)

        run_seed = np.random.rand()
        f = open('backtest_results_'+self._config.model_name+'_'+str(run_seed)+'.txt', 'w')
        f.write('run index: '+str(run_index)+'\n')
        f.write('epoch: '+str(epoch)+'\n')
        f.write('runs: '+str(runs)+'\n')
        f.write('ticks_per_run: '+str(ticks_per_run)+'\n')
        f.write('samples_per_tick: '+str(samples_per_tick)+'\n')
        f.write('view: '+str(view)+'\n')
        f.write('risk_view: '+str(risk_view)+'\n')
        f.write('transaction_cost: '+str(transaction_cost)+'\n')
        f.write('T1: '+str(T1)+'\n')
        f.write('T0: '+str(T0)+'\n')
        f.write('ROI: '+str(np.sum(R))+'\n')
        f.write('profit: '+str(np.sum(P))+'\n')
        f.write('pihat: '+str(pihat)+'\n')
        f.write('lr: '+str(lr)+'\n')
        f.write('ratio: '+str(ratio)+'\n')
        f.write('es unconditional observed: ' +str(-1*np.mean(violations*observed_prices /(alpha/100.)))+'\n')
        f.write('es unconditional expected: ' +str(np.mean(es_unc))+'\n')
        f.write('es unconditional predicted: ' +str(np.mean(es_unc*violations /(alpha/100.)))+'\n')
        f.write('es conditional observed: ' +str(-1*np.sum(violations*observed_prices) /np.sum(violations))+'\n')
        f.write('es conditional expected: ' +str(np.mean(es))+'\n')
        f.write('es conditional predicted: ' +str(np.sum(violations*es) /np.sum(violations))+'\n')
        f.close()

        f = open('backtest_results_'+self._config.model_name+'_'+str(run_seed)+'.txt', 'r')
        for line in f:
            print(line)
        f.close()

    
   
    
        

