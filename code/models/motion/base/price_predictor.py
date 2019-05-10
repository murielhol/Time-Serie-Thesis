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
import h5py

import time
import copy

import torch
from torch.autograd import Variable

import timeit


from model import Model


from models.motion.mmd import mix_rbf_mmd2_and_ratio
import models.motion.forward_kinematics as forward_kinematics
import models.motion.viz as viz



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

        # np.random.seed(self._config.seed)
        # tf.set_random_seed(self._config.seed)
        torch.manual_seed(self._config.seed)

            
    @staticmethod
    def adjust_lr(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*0.95
        print('lr: ', param_group['lr'])
    
    @staticmethod
    def get_receptive_field(model, config):
        # make sure that batch norm is turned off
        model.gen.eval()
        model.gen_opt.zero_grad()
        # imagine batch size is 10, seq_len is 1000 and 1 channel
        bs = config.batch_size
        seq_len = config.input_seq_length
        channels = 54
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
        receptive_field = self.get_receptive_field(self._model, self._config)
        print('--------------------------------------------------------------------')
        print('NOTE: the receptive field is ', receptive_field, ' and your input is ', self._config.input_seq_length)
        print('--------------------------------------------------------------------')
        t2 = time.time()
        print('Finished building the model: ' + str(t2-t1) +' sec \n')
        # ################# get data ################################
        self._dataset.prepare_data(self._config, receptive_field)

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
        steps_per_epoch = self._dataset.train_size//(self._config.batch_size)
        ############## initialize all the stuff ##################
 
        t = timeit.default_timer()
        mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        mask[receptive_field-1:, :] = 1
        mask_gen = Variable(torch.from_numpy(mask).float())
                
        test_mask = np.zeros([self._config.input_seq_length, self._config.batch_size])
        test_mask[receptive_field-1:, :] = 1
        test_mask_gen = Variable(torch.from_numpy(test_mask).float())



        output_len = self._config.input_seq_length - receptive_field + 1

        for epoch in range(1, int(self._config.epochs+1)):
            loss_sum = 0
            test_loss_sum = 0
            log = []
            print('--------- Epoch nr ', epoch, ' ------------')
            for train_step in range(1, int(1001)):
                x, y = self._dataset.get_batch(self._dataset.train_set)
                assert np.sum(x[:,1,:] - y[:,0,:])==0

                x = np.einsum('ijk->jik', x)
                y = np.einsum('ijk->jik', y)
                x = Variable(torch.from_numpy(x).float())
                trueY = Variable(torch.from_numpy(y).float())
                
                loss, pred = self._model.gen([x,trueY, mask_gen])
                loss_sum+=loss.item()
                
                self._model.gen.zero_grad()
                loss.backward()
                self._model.gen_opt.step()

                if train_step%100==0:
                    print('step', train_step, ':' ,loss.item())

                ################ occasionally show the (test) performance #############  
            x, y = self._dataset.get_batch(self._dataset.test_set)
            test_cost, _ = self.evaluate(x, y, test_mask_gen)
            test_loss_sum += test_cost

            # === Validation with srnn's seeds ===
            for action in self._dataset.actions:

                # Evaluate the model on the test batches
                x, y , _= self._dataset.get_batch_srnn(self._dataset.test_set, action)
                srnn_loss, srnn_poses = self.evaluate(x, y, mask_gen[:,:8])
                # print(np.sum(x[:, :receptive_field-1, :]), np.shape(x[:receptive_field-1]))
                # boe
                srnn_poses = np.einsum('ijk->jik',srnn_poses[0].detach().numpy()[-output_len:,:, :])
                # Denormalize the output
                srnn_pred_expmap = self._dataset.revert_output_format(srnn_poses)

                # Save the errors here
                mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))

                # Training is done in exponential map, but the error is reported in
                # Euler angles, as in previous work.
                # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
                N_SEQUENCE_TEST = 8
                for i in np.arange(N_SEQUENCE_TEST):
                    eulerchannels_pred = srnn_pred_expmap[i]

                    # Convert from exponential map to Euler angles
                    for j in np.arange( eulerchannels_pred.shape[0] ):
                        for k in np.arange(3,97,3):
                            eulerchannels_pred[j,k:k+3] = self._dataset.rotmat2euler(
                                                          self._dataset.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

                    # The global translation (first 3 entries) and global rotation
                    # (next 3 entries) are also not considered in the error, so the_key
                    # are set to zero.
                    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                    gt_i=np.copy(self._dataset.srnn_gts_euler[action][i])
                    gt_i[:,0:6] = 0

                    # Now compute the l2 error. The following is numpy port of the error
                    # function provided by Ashesh Jain (in matlab), available at
                    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                    idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]

                    euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                    euc_error = np.sum(euc_error, 1)
                    euc_error = np.sqrt( euc_error )
                    mean_errors[i,:] = euc_error

                # This is simply the mean error over the N_SEQUENCE_TEST examples
                mean_mean_errors = np.mean( mean_errors, 0 )

                # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                print("{0: <16} |".format(action), end="")
                line = ""
                for ms in [1,3,7,9,13,24]:
                    if self._config.input_seq_length - receptive_field >= ms+1:
                        line+=" {0:.3f} |".format( mean_mean_errors[ms] )
                    else:
                        line+="   n/a |"

                print(line)


            s = timeit.default_timer()
            log_line = 'total time: [%f], epoch: [%d/%d], step: [%d/%d], loss: %f,  test_loss: %f' % (
               s-t, epoch, self._config.epochs, train_step, steps_per_epoch, loss_sum / train_step, test_loss_sum )
            print(log_line)

            if epoch%10==0:
                self.adjust_lr(self._model.gen_opt)

          
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
        self._dataset.prepare_data(self._config, receptive_field)

        self._model._build_model()
        state = torch.load('saved_models/'+self._config.model_name+'/'+self._config.model_name+str(epoch)+'.pth.tar', map_location='cpu')
        self._model.gen.load_state_dict(state['state_dict_gen'])
        output_len = self._config.output_seq_length
        ERORR = np.zeros(10)
        SAMPLES_FNAME = 'samples.h5'
        try:
            os.remove( SAMPLES_FNAME )
        except OSError:
            pass

        # === Validation with srnn's seeds ===
        for action in self._dataset.actions:

            x, y , _= self._dataset.get_batch_srnn(self._dataset.test_set, action)

            mask = np.zeros([receptive_field, np.shape(x)[0]])
            mask[receptive_field-1:, :] = 1
            mask_gen = Variable(torch.from_numpy(mask).float())

            # Evaluate the model on the test batches
            x, y , _= self._dataset.get_batch_srnn(self._dataset.test_set, action)
            X = x[:,:receptive_field,:].copy()
            Y = y[:,:receptive_field,:].copy()

            for step in range(output_len):
                srnn_loss, pose  = self.evaluate(X[:, step:step+receptive_field,:], Y, mask_gen[:receptive_field,:8])
                pose = np.einsum('ijk->jik',pose[0].detach().cpu().numpy()[-1:,:, :])
                X = np.concatenate([X, pose], axis = 1)

            srnn_poses = X[:,-output_len:,:]

            # Denormalize the output
            srnn_pred_expmap = self._dataset.revert_output_format(srnn_poses)


            # Save the samples
            with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
                for i in np.arange(8):
                # Save conditioning ground truth
                    node_name = 'expmap/gt/{1}_{0}'.format(i, action)
                    hf.create_dataset( node_name, data=self._dataset.srnn_gts_expmap[action][i] )
                    # Save prediction
                    node_name = 'expmap/preds/{1}_{0}'.format(i, action)
                    hf.create_dataset( node_name, data=srnn_pred_expmap[i] )

            # Save the errors here
            mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))
            mean_mmd = np.zeros((len(srnn_pred_expmap)))

            # Training is done in exponential map, but the error is reported in
            # Euler angles, as in previous work.
            # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
            N_SEQUENCE_TEST = 8
            for i in np.arange(N_SEQUENCE_TEST):
                eulerchannels_pred = srnn_pred_expmap[i]

                # Convert from exponential map to Euler angles
                for j in np.arange( eulerchannels_pred.shape[0] ):
                    for k in np.arange(3,97,3):
                        eulerchannels_pred[j,k:k+3] = self._dataset.rotmat2euler(
                                                      self._dataset.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

                # The global translation (first 3 entries) and global rotation
                # (next 3 entries) are also not considered in the error, so the_key
                # are set to zero.
                # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                gt_i=np.copy(self._dataset.srnn_gts_euler[action][i])
                gt_i[:,0:6] = 0
                # gt_i=np.copy(self._dataset.srnn_gts_euler[action][i])
                # eulerchannels_pred[:,0:6] = 0

                # Now compute the l2 error. The following is numpy port of the error
                # function provided by Ashesh Jain (in matlab), available at
                # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
                idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]


                euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
                euc_error = np.sum(euc_error, 1)
                euc_error = np.sqrt( euc_error )
                mean_errors[i,:] = euc_error

                sigma_list = [Variable(torch.from_numpy(np.array(s)).float(), requires_grad=False) for s in np.arange(10, 50, 5)]
                gt = Variable(torch.from_numpy(gt_i[:,idx_to_use])).float()
                pred = Variable(torch.from_numpy(eulerchannels_pred[:,idx_to_use])).float()
                _, mmd, that = mix_rbf_mmd2_and_ratio(gt, pred, sigma_list)
                mean_mmd[i] = mmd

            # This is simply the mean error over the N_SEQUENCE_TEST examples
            mean_mean_errors = np.mean( mean_errors, 0 )
            ERORR+=mean_mean_errors

            print( '------------------', action , '------------------')
            print( ','.join(map(str, mean_mean_errors.tolist() )) )
            print(mmd)

            with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
                node_name = 'mean_{0}_error'.format( action )
                hf.create_dataset( node_name, data=mean_mean_errors )
        print('------------average mean error -------------')
        print(ERORR/15.0)
        print(np.mean(ERORR/15.0))
        print('mmd', np.mean(mean_mmd))

    def _make_figs(self, path = 'samples.h5'):
        forward_kinematics.run(self._dataset)

