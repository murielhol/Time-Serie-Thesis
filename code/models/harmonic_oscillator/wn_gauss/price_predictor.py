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

        for epoch in range(1, int(self._config.epochs+1)):
            loss_sum = 0
            test_loss_sum = 0
            log = []
            ########## before each epoch, reset batch (and shuffle) ########
            self._dataset.reset(self._config.shuffle)
            print('--------- Epoch nr ', epoch, ' ------------')
            print('- train step  | train loss | test loss |')
            for train_step in range(1, int(steps_per_epoch)):
                
                print('train step: ', train_step)
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
            test_cost, _ = self.evaluate(x, y, mask)
            test_loss_sum += test_cost

            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f,  test_loss: %f' % (
               s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum )
            print(log_line)
            
            log.append([train_step, np.mean(costs), np.mean(test_costs)])
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
            locs = Variable(torch.from_numpy(np.einsum('ijk->jik',loc))).float()
            logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
            Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
            test_pred = Dist.sample()
            X = np.concatenate([X, test_pred], axis = 1)
            print('shape x', np.shape(X))
        
        preds = X[:, -steps:, 0]
        tars = y[:,-steps:,0]
        MSE = np.mean((preds-tars)**2, axis=0)

        sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]]
        _, mmd, that = mix_rbf_mmd2_and_ratio(Variable(torch.from_numpy(preds)).float(), 
                                              Variable(torch.from_numpy(tars)).float(), sigma_list)

        print('MMD : ', mmd.item(),  ' THAT: ', that.item())
        print('MSE@1: ', MSE[0])
        print('MSE@sum: ', np.sum(MSE))
        print('LL@1 ', LL[0])
        print('LL@10', np.sum(LL))


        f = open('results_'+self._config.model_name+'.txt', 'w')
        f.write('MMD: '+str(mmd.item())+'\n')
        f.write('THAT: '+str(that.item())+'\n')
        f.write('LL1: '+str(LL[0])+'\n')
        f.write('LL5: '+str(LL[4])+'\n')
        f.write('LL10: '+str(LL[9])+'\n')
        f.write('LLtotal: '+str(np.sum(LL))+'\n')
        f.write('MSE1: '+str(MSE[0])+'\n')
        f.write('MSEtotal: '+str(np.sum(MSE))+'\n')
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

        df.to_csv('images/wn_gauss_samples.csv')

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

        df.to_csv('images/wn_gauss_prediction.csv')
        
    @staticmethod
    def denorm(vector):
        return (0.5 * (1+vector))+1e-14

    def _backtest(self, runs = 100, ticks_per_run=200, samples_per_tick=100, view = 5, risk_view=1, epoch=1000, alpha=5):
        # make sure this is configures right
        t1 = time.time()
        self._model._build_model()
        receptive_field = self.get_receptive_field(self._model, self._config)
        self._config.input_seq_length = receptive_field
        self._config.output_seq_length = ticks_per_run+view
        self._config.batch_size = runs

        assert self._config.input_seq_length < (300-ticks_per_run-view) # data is 300 long
        self._dataset.prepare_data(self._config)
        self._model._build_model()
        X, Y = self._dataset.get_validation_set()
        runs = runs if runs else len(X)
        s = np.arange(np.shape(X)[0])
        np.random.shuffle(s)
        print(s[:runs])
        X = X[s[:runs],:,:]
        Y = Y[s[:runs],:,:]

        mask = np.zeros([receptive_field, np.shape(X)[0]])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())

        # initial situation
        have_state = [False for _ in range(runs)]
        actions = np.full((runs, ticks_per_run), 'hold')
        transaction_cost = 0.0
        riskiness = 0.0
        D = (1+transaction_cost)/(1-transaction_cost)
        print('Discount factor :' , D)
        var = np.zeros(np.shape(actions))
        es = np.zeros(np.shape(actions))
        ef = np.zeros(np.shape(actions))
        es_unc = np.zeros(np.shape(actions))
        expected_returns = np.zeros(np.shape(actions))
        observed_returns = np.zeros(np.shape(actions))
        t2 = time.time()
        print('init done in :', str(t2-t1), 'sec')

        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])
        # initial situation
        context = X.copy()
        # start ticks
        for tick in range(ticks_per_run):
            print('tick: ', str(tick) , '/', str(ticks_per_run))
            # fill this with sampled futures
            futures = np.zeros([runs, view, samples_per_tick])
            for sample in range(samples_per_tick):
                context_ = context.copy()
                np.random.seed(sample)
                tf.set_random_seed(sample)
                torch.manual_seed(sample)
                # free run the model to predict multiple steps
                for step in range(view):  
                    _, test_pars = self.evaluate(context_[:, step:, :], context_[:, step:, :], mask_gen)
                    loc = test_pars[0].detach().numpy()[-1:,:, :]
                    locs = Variable(torch.from_numpy(np.einsum('ijk->jik',loc))).float()
                    logvars = Variable(torch.from_numpy(np.exp(np.einsum('ijk->jik', test_pars[1].detach().numpy()[-1:,:, :])))).float()
                    Dist = torch.distributions.normal.Normal(locs, logvars, validate_args=None)
                    pred = Dist.sample()     
                    context_ = np.concatenate([context_, pred], axis = 1)

                # denormalize back to [0,1)
                futures[:, :, sample] = self.denorm(context_[:,receptive_field:,0])
            x0 = self.denorm(context[:, -1])
            if tick==0:
                t3 = time.time()
                print('sampling done in :', str(t3-t2), 'sec')
            # average all samples 
            expected_future = np.mean(futures, axis=-1)
            futures = np.array(futures)
            # collect risk measurements
            print('future: ', np.shape(futures))
            returns = np.divide(futures[:,risk_view-1, :],x0) - 1.0
            print('returns: ', np.shape(futures))
            expected_returns[:, tick] = np.mean(returns, axis=-1)
            var[:, tick] = value_at_risk(returns)
            es[:, tick] = expected_shortfall(returns)
            es_unc[:, tick] = unconditional_expected_shortfall(returns)
            ef[:, tick] = expected_future[:,0]

            nextx = self.denorm(Y[:, tick+receptive_field-1, 0])
            nextx = np.divide(nextx, x0[:,0])-1.0
            observed_returns[:, tick] = nextx

            # for each example in the batch, find trading action
            for i in range(runs):
                if not have_state[i]:
                    # check when and if there is a profitable sell moment
                    t_sell = get_next_or_none(tau for tau, m in enumerate(expected_future[i,:]) if m > x0[i]*D)
                                              # and riskiness > -1 * expected_shortfall(x0, futures[:, tau]))
                    if t_sell is not None:
                        # check if until that sell moment arrives, there is a better buy moment
                        if (t_sell == 0 or expected_future[i,0:t_sell].min() >= x0[i]):  # and riskiness > (-1*ES):
                            have_state[i] = True
                            actions[i,tick] = 'buy'

                else:
                    # check if there is a moment when buying a new share is cheaper than keeping this one
                    t_buy = get_next_or_none(tau for tau, m in enumerate(expected_future[i,:]) if m < x0[i])
                    # check if until that moment arrives, there is a better sell moment
                    if t_buy is not None:
                        if (t_buy == 0 or expected_future[i,0:t_buy].max() <= x0[i]):
                            have_state[i] = False
                            actions[i,tick] = 'sell'


            context = Y[:, tick:tick+receptive_field,:]
            if tick==0:
                t4 = time.time()
                print('trading done in :', str(t4-t3), 'sec')


        observations = self.denorm(Y[:, receptive_field-2:receptive_field+ticks_per_run-1, 0])
        returns = np.zeros(np.shape(actions))
        for i in range(ticks_per_run):
            returns[:,i] = (observations[:,i+1]/observations[:,i])-1 


        violations = observed_returns-var
        np.putmask(violations, violations>=0, np.ones(np.shape(returns)))
        np.putmask(violations, violations<0, 2*np.ones(np.shape(returns)))
        violations -= 1
        T1 = np.sum(violations)
        T0 = (runs*ticks_per_run)-T1
        pihat = T1/(T0+T1)
        lr = likelihood_ratio(pihat, alpha/100., T1, T0)
        z1 = z1_score(returns, violations, es, T1, alpha/100.)
        z2 = z2_score(returns, violations, es_unc, T1+T0, alpha/100.)
        z3 = z2_score(returns, violations, es, T1+T0, alpha/100.)

        R = []
        P = []
        for i in range(runs):
            roi, profit = compute_roi(self.denorm(Y[i, receptive_field-1:receptive_field+ticks_per_run]), actions[i,:], transaction_cost, violations[i,:])
            R.append(roi)
            P.append(profit)

        f = open('backtest_results_'+self._config.model_name+'txt', 'w')
        f.write('runs: '+str(runs)+'\n')
        f.write('ticks_per_run: '+str(ticks_per_run)+'\n')
        f.write('samples_per_tick: '+str(samples_per_tick)+'\n')
        f.write('view: '+str(view)+'\n')
        f.write('risk_view: '+str(risk_view)+'\n')
        f.write('transaction_cost: '+str(transaction_cost)+'\n')
        f.write('riskiness: '+str(riskiness)+'\n')
        f.write('T1: '+str(T1)+'\n')
        f.write('T0: '+str(T0)+'\n')
        f.write('ROI: '+str(np.sum(R))+'\n')
        f.write('profit: '+str(np.sum(P))+'\n')
        f.write('pihat: '+str(pihat)+'\n')
        f.write('lr: '+str(lr)+'\n')
        f.write('z1: '+str(z1)+'\n')
        f.write('z2: '+str(z2)+'\n')
        f.write('z3: '+str(z3)+'\n')
        f.write('es unconditional observed: ' +str(-1*np.mean(violations*returns /(alpha/100.)))+'\n')
        f.write('es unconditional predicted: ' +str(np.mean(es_unc*violations /(alpha/100.)))+'\n')
        f.write('es conditional observed: ' +str(-1*np.sum(violations*returns) /np.sum(violations))+'\n')
        f.write('es conditional predicted: ' +str(np.sum(violations*es) /np.sum(violations))+'\n')
        f.close()

        f = open('backtest_results_'+self._config.model_name+'txt', 'r')
        for line in f:
            print(line)
        f.close()

    
   
    
        

