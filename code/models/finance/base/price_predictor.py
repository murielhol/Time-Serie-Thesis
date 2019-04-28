from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd 
import json
import shutil
import pickle


import torch
from torch.autograd import Variable

import timeit

from model import Model


from models.finance.mmd import mix_rbf_mmd2_and_ratio

from models.finance.trading_utils import *




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

        np.random.seed(self._config.seed)
        tf.set_random_seed(self._config.seed)
        torch.manual_seed(self._config.seed)

            
    
    @staticmethod
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.gen.eval()
        model.gen_opt.zero_grad()
        # imagine batch size is 10, seq_len is 1000 and 1 channel
        bs = config.batch_size
        seq_len = config.input_seq_length
        channels = len(config.features)
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

    def evaluate(self, x, y,  mask_gen):
        x = np.einsum('ijk->jik', x)
        y = np.einsum('ijk->jik', y)
        self._model.gen.eval()
        x = Variable(torch.from_numpy(x)).float()
        y = Variable(torch.from_numpy(y)).float()
        loss, pred = self._model.gen([x,y, mask_gen]);
        return loss.item(), pred


    def _train(self):
        

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
        self._dataset.prepare_data(self._config, shuffle = self._config.shuffle, skip = 1)
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
        
        steps_per_epoch = len(self._dataset.train_y)//(self._config.batch_size)
        ############## initialize all the stuff ##################
 

        t = timeit.default_timer()
        mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())
                
        test_mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        test_mask[receptive_field-1:, :] = 1
        test_mask_gen = Variable(torch.from_numpy(test_mask).float())
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
                
                loss, pred = self._model.gen([x,trueY, mask_gen])
                loss_sum+=loss.item()
                
                self._model.gen.zero_grad()
                loss.backward()
                self._model.gen_opt.step()
                ################ occasionally show the (test) performance #############  
            x, y = self._dataset.get_batch(self._config.batch_size, test = True)
            test_cost, _ = self.evaluate(x, y, test_mask_gen)
            test_loss_sum += test_cost

            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f,  test_loss: %f' % (
               s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum )
            print(log_line)
          
            log.append([train_step, loss_sum / train_step, test_loss_sum])
            #self.adjust_lr(self._model.opt, epoch, self._config.epochs, self._config.learning_rate, 0.)
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
        x, y, target = self._dataset.get_validation_set()

        mask = np.zeros([receptive_field, np.shape(x)[0]])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])

        LL=[]

        X = x[:, :receptive_field, :].copy()
        for step in range(steps):
            # print('step: ', step)
            test_cost, test_pars = self.evaluate(X[:, step:step+receptive_field, :], y[:, step:step+receptive_field, :], mask_gen)
            if False:#self._config.loss == 'Gaussian':
                locs = Variable(torch.from_numpy(np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:, :]))).float()
                logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
                test_pred = Dist.sample()
            else:
                test_pred = np.einsum('ijk->jik',test_pars[0].detach().numpy()[-1:,:, :])
            X = np.concatenate([X, test_pred], axis = 1)
            print('shape x', np.shape(X))
            LL.append(test_cost)


        d = 1

        # from log returns to returns
        Yreturns = self.compound(self.denorm(y[:,-steps:,d]), dim=1) 
        Xreturns = self.compound(self.denorm(X[:,-steps:,d]), dim=1)

        plt.plot(Yreturns[:,-1])
        plt.plot(Xreturns[:,-1])
        plt.show()

        # from returns to dollar
        tars = self.convert(Yreturns, np.expand_dims(target[receptive_field-1:receptive_field-1+np.shape(X)[0]], 1))
        preds = self.convert(Xreturns, np.expand_dims(target[receptive_field-1:receptive_field-1+np.shape(X)[0]], 1))

        plt.plot(tars[:,-1])
        plt.plot(preds[:,-1])
        plt.show()

        MSE = (preds-tars)**2
        L1 = np.sqrt(MSE)
        MSE = np.mean(MSE, axis=0)
        L1 = np.mean(L1, axis=0)

        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [5, 10, 50, 100]]
        _, mmd, that = mix_rbf_mmd2_and_ratio(Variable(torch.from_numpy(preds)).float(), 
                                              Variable(torch.from_numpy(tars)).float(), sigma_list)

        if not os.path.exists('results/'):
            os.makedirs('results/')

        f = open('results/results_'+self._config.model_name+'_'+self._config.backtest_target+'.txt', 'w')
        f.write('epoch: '+str(epoch)+'\n')
        f.write('steps '+str(steps)+'\n')
        f.write('target '+str(self._config.backtest_target)+'\n')
        f.write('N '+str(np.shape(X)[0])+'\n')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('THAT: '+str(that.item())+'\n')
        f.write('MSE@1: '+str(np.sum(MSE[0]))+'\n')
        f.write('MSE@total: '+str(np.sum(MSE))+'\n')
        f.write('MAE@1: '+str(np.sum(L1[0]))+'\n')
        f.write('MAE@total: '+str(np.sum(L1))+'\n')
        f.write('LL@1: '+str(np.sum(LL[0]))+'\n')
        f.write('LL@total: '+str(np.sum(LL))+'\n')
        f.close()
        f = open('results/results_'+self._config.model_name+'_'+self._config.backtest_target+'.txt', 'r')
        for line in f:
            print(line)

        # dump confg json to keep track of model properties
        with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
            json.dump(vars(self._config), fp)
        with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
            pickle.dump( self._config, fp )

    def denorm(self, x):
        '''
        '''
        if self._config.normalize == 'minmax':
            # first denorm between 0 and 1
            y = (x+1)*0.5
            # denorm to original range
            denorm_pars = self._dataset.denorm_pars
            mi = denorm_pars[0]
            ma = denorm_pars[1]
            y = y * (ma-mi) + mi
            # from log return to returns
            # print('denorm minmax')
            return np.exp(y) - 1.0
            
        return np.exp(x) - 1.0
        

    @staticmethod
    def compound(returns, dim=1):
       
        return np.cumprod(returns+1.0, axis=dim) - 1.0

    @staticmethod
    def convert(compound_returns, seed):
    
        return (compound_returns + 1.0) * seed


    def _backtest(self, samples_per_tick=200, view = 1, risk_view=1, epoch=1000, alpha=5):
        t1 = time.time()
        self._dataset.prepare_data(self._config)
        
        self._config.batch_size = 1
        runs = 1
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)

        data, close = self._dataset.get_backtest_set()
        ticks_per_run = np.shape(data)[1]-receptive_field-view

        d = 2
        # check if indexing is correct
        assert self._config.backtest_target.split('close_')[0] in self._config.features[d]
        target = close


        # test if denormalization is correct
        check = self.convert(self.compound(self.denorm(data[:,1:,d])), close[0])
        if np.sum(np.around(close[1:101], 1) - np.around(check[0,:100], 1)) > 0:
            plt.plot(close[1:101] - check[0,:100])
            plt.show()

        X = data[:, :receptive_field, :]
        mask = np.zeros([receptive_field, 1])
        mask[receptive_field-1:, :] = 1
        mask = Variable(torch.from_numpy(mask).float())
        mask_big = np.zeros([receptive_field, samples_per_tick])
        mask_big [receptive_field-1:, :] = 1
        mask_big  = Variable(torch.from_numpy(mask_big).float())

        # initial situation
        have_state = False
        actions = np.full((1,ticks_per_run), 'hold')


        var = np.zeros(np.shape(actions))
        es = np.zeros(np.shape(actions))
        es_unc = np.zeros(np.shape(actions))
        observed_returns = np.zeros(np.shape(actions))

        # compute minimal discount factor
        transaction_cost = 0.00
        D = (1+transaction_cost)/(1-transaction_cost)
        print('Discount factor :' , D)

        actions_ma2 = self.baseline(target[receptive_field-2:], actions.copy(), 2, D)
        actions_ma5 = self.baseline(target[receptive_field-2:], actions.copy(), 5, D)
        actions_ma10 = self.baseline(target[receptive_field-2:], actions.copy(), 10, D)
        actions_ma20 = self.baseline(target[receptive_field-2:], actions.copy(), 20, D)
        actions_ma50 = self.baseline(target[receptive_field-2:], actions.copy(), 50, D)
        roi_ma2, profit_ma2 = compute_roi(target[receptive_field-1:], actions_ma2[0,:], transaction_cost)
        roi_ma5, profit_ma5 = compute_roi(target[receptive_field-1:], actions_ma5[0,:], transaction_cost)
        roi_ma10, profit_ma10 = compute_roi(target[receptive_field-1:], actions_ma10[0,:], transaction_cost)
        roi_ma20, profit_ma20 = compute_roi(target[receptive_field-1:], actions_ma20[0,:], transaction_cost)
        roi_ma50, profit_ma50 = compute_roi(target[receptive_field-1:], actions_ma50[0,:], transaction_cost)



        t2 = time.time()
        print('init done in :', str(t2-t1), 'sec')

        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])
       

        for tick in range(ticks_per_run):
            # x0 is the current tick price
            x0 = target[tick+receptive_field-1]
            # normalized context for this tick 
            context  = data[:, tick:tick+receptive_field,:]
            print('tick: ', str(tick) , '/', str(ticks_per_run))
            # we fill this with sampled futures
            future_returns = np.zeros([1, view, samples_per_tick])
            future_prices = np.zeros([1, view, samples_per_tick])

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
                
                # from log returns to cumulative returns
                future_returns[:, :, sample] = self.compound(self.denorm(context_[:, receptive_field:, d]), dim=-1)
                # from cumulative returns to prices
                future_prices[:, :, sample] = self.convert(future_returns[:, :, sample], x0 )

            # sample mean
            expected_future = np.mean(future_prices, axis=-1)
            # returns that we expect at risk_view
            returns = future_returns[0,risk_view-1,:]
            
            # estimate risk  
            var[:, tick] = value_at_risk(returns)
            es[:, tick] = expected_shortfall(returns)
            es_unc[:, tick] = unconditional_expected_shortfall(returns)

            next_return = self.denorm(data[:, tick+receptive_field, d])
            observed_returns[:, tick] = next_return

    
            # choose trading action
            if not have_state:
                # check when and if there is a profitable sell moment
                t_sell = get_next_or_none(tau for tau, m in enumerate(expected_future[0,:]) if m > x0*D)

                if t_sell is not None:
                    # check if until that sell moment arrives, there is a better buy moment
                    if (t_sell == 0 or expected_future[0,0:t_sell].min() >= x0): 
                        have_state = True
                        actions[0,tick] = 'buy'
                        buying_price = x0
                        print('buying')
            else:
                # check if there is a moment when buying a new share is cheaper than keeping this one
                t_buy = get_next_or_none(tau for tau, m in enumerate(expected_future[0,-1:]) if m < x0)
                
                if t_buy is not None:
                    # check if until that moment arrives, there is a better sell moment
                    if (t_buy == 0 or expected_future[0,0:t_buy].max() <= x0):
                        have_state = False
                        actions[0,tick] = 'sell'
                        print('selling')

            
        # a violation is when the observed return is below the value at risk return
        violations = observed_returns-var
        np.putmask(violations, violations>=0, np.ones(np.shape(observed_returns)))
        np.putmask(violations, violations<0, 2*np.ones(np.shape(observed_returns)))
        violations -= 1
        T1 = np.sum(violations)
        T0 = (runs*ticks_per_run)-T1
        # evaluate risk estimation performance
        pihat = T1/(T0+T1)
        lr = likelihood_ratio(pihat, alpha/100., T1, T0)
        z1 = z1_score(observed_returns, violations, es, T1, alpha/100.)
        z2 = z2_score(observed_returns, violations, es_unc, T1+T0, alpha/100.)
        z3 = z2_score(observed_returns, violations, es, T1+T0, alpha/100.)

        # see if the chosen actions would have been profitable or not
        roi, profit = compute_roi(target[receptive_field-1:], actions[0,:], transaction_cost)

        run_seed = np.random.rand()
        if not os.path.exists('results/'):
            os.makedirs('results/')
        f = open('results/backtest_results_'+self._config.model_name+'_'+str(run_seed)+'_'+self._config.backtest_target+'.txt', 'w')
        f.write('runs: '+str(runs)+'\n')
        f.write('ticks_per_run: '+str(ticks_per_run)+'\n')
        f.write('samples_per_tick: '+str(samples_per_tick)+'\n')
        f.write('view: '+str(view)+'\n')
        f.write('risk_view: '+str(risk_view)+'\n')
        f.write('transaction_cost: '+str(transaction_cost)+'\n')
        f.write('T1: '+str(T1)+'\n')
        f.write('T0: '+str(T0)+'\n')
        f.write('ROI: '+str(roi)+'\n')
        f.write('profit: '+str(profit)+'\n')
        f.write('pihat: '+str(pihat)+'\n')
        f.write('lr: '+str(lr)+'\n')
        f.write('z1: '+str(z1)+'\n')
        f.write('z2: '+str(z2)+'\n')
        f.write('z3: '+str(z3)+'\n')
        f.write('es unconditional observed: ' +str(-1*np.mean(violations*observed_returns /(alpha/100.)))+'\n')
        f.write('es unconditional expected: ' +str(np.mean(es_unc))+'\n')
        f.write('es unconditional predicted: ' +str(np.mean(es_unc*violations /(alpha/100.)))+'\n')
        f.write('es conditional observed: ' +str(-1*np.sum(violations*observed_returns) /np.sum(violations))+'\n')
        f.write('es conditional expected: ' +str(np.mean(es))+'\n')
        f.write('es conditional predicted: ' +str(np.sum(violations*es) /np.sum(violations))+'\n')
        f.write('\n')
        f.write('Dummy baselines: \n')
        f.write('ma2: '+str(roi_ma2)+', '+str(profit_ma2)+'\n')
        f.write('ma5: '+str(roi_ma5)+', '+str(profit_ma5)+'\n')
        f.write('ma10: '+str(roi_ma10)+', '+str(profit_ma10)+'\n')
        f.write('ma20: '+str(roi_ma20)+', '+str(profit_ma20)+'\n')
        f.write('ma50: '+str(roi_ma20)+', '+str(profit_ma50)+'\n')
        
        f.close()

        f = open('results/backtest_results_'+self._config.model_name+'_'+str(run_seed)+'_'+self._config.backtest_target+'.txt', 'r')
        for line in f:
            print(line)
        f.close()

        # dump confg json to keep track of model properties
        with open('saved_models/'+self._config.model_name+'/config.json', 'w') as fp:
            json.dump(vars(self._config), fp)
        with open('saved_models/'+self._config.model_name+'/config.p', 'wb') as fp:
            pickle.dump( self._config, fp )


    def baseline(self, target, actions, factor, D):
        ticks = np.shape(actions)[1]
        target_ma = [np.mean(target[i-(factor-1):i+1]) if i > (factor-2) else target[i] for i in range(len(target))]
        plt.plot(target_ma, c='m')
        have_state = False
        for i in range(ticks):

            if (not have_state) and (target_ma[i+1]>(D*target_ma[i])):
                actions[0, i] = 'buy'
                have_state = True
                plt.scatter(i+1, target_ma[i+1], c='g')
            elif have_state and target_ma[i+1]<target_ma[i]:
                actions[0, i] = 'sell'
                have_state = False
                plt.scatter(i+1, target_ma[i+1], c='r')
        # plt.show()
        return actions







    
    
  