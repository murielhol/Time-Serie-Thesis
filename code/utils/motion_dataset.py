from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pandas as pd


from six.moves import xrange # pylint: disable=redefined-builtin

class MotionDataset(object):

    '''
    TODOs:

    '''

    def __init__(self, actions, train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use):
        self.train_set = train_set
        self.test_set = test_set
        self.data_mean = data_mean
        self.data_std = data_std
        self.dim_to_ignore = dim_to_ignore
        self.dim_to_use = dim_to_use
        self.actions = actions
        self.train_size = 0
        for key in train_set.keys():
            self.train_size+= np.shape(train_set[key])[0]




    def get_srnn_gts( self, to_euler=True ):
        """
        Get the ground truths for srnn's sequences, and convert to Euler angles.
        (the error is always computed in Euler angles).

        Args
        actions: a list of actions to get ground truths for.
        model: training model we are using (we only use the "get_batch" method).
        test_set: dictionary with normalized training data.
        data_mean: d-long vector with the mean of the training data.
        data_std: d-long vector with the standard deviation of the training data.
        dim_to_ignore: dimensions that we are not using to train/predict.
        one_hot: whether the data comes with one-hot encoding indicating action.
        to_euler: whether to convert the angles to Euler format or keep thm in exponential map

        Returns
        srnn_gts_euler: a dictionary where the keys are actions, and the values
          are the ground_truth, denormalized expected outputs of srnns's seeds.
        """
        srnn_gts_euler = {}

        for action in self.actions:

            srnn_gt_euler = []
            _, _,  srnn_expmap = self.get_batch_srnn( self.test_set, action )

            # expmap -> rotmat -> euler
            for i in np.arange( srnn_expmap.shape[0] ):
                denormed = self.unNormalizeData(srnn_expmap[i,:,:], self.data_mean, self.data_std, self.dim_to_ignore, self.actions, False )

                if to_euler:
                    for j in np.arange( denormed.shape[0] ):
                        for k in np.arange(3,97,3):
                            denormed[j,k:k+3] = self.rotmat2euler( self.expmap2rotmat( denormed[j,k:k+3] ))
                srnn_gt_euler.append( denormed );

            # Put back in the dictionary
            srnn_gts_euler[action] = srnn_gt_euler

        return srnn_gts_euler

    def prepare_data(self, config, receptive_field):
        self.batch_size = config.batch_size
        if config.validate:
            self.target_seq_len = config.output_seq_length
            self.source_seq_len = receptive_field
        else:
            self.target_seq_len = config.input_seq_length - receptive_field + 1
            self.source_seq_len = receptive_field
        self.input_size = 54
        self.srnn_gts_euler = self.get_srnn_gts(to_euler=True )
        self.srnn_gts_expmap = self.get_srnn_gts(to_euler=False )
        




    def get_batch( self, data ):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys    = list(data.keys())
        chosen_keys = np.random.choice( len(all_keys), self.batch_size )
        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len 
        # encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
        # decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        # decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len+self.target_seq_len-1, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.source_seq_len+self.target_seq_len-1, self.input_size), dtype=float)

        for i in range( self.batch_size ):

            the_key = all_keys[ chosen_keys[i] ]

            # Get the number of frames
            n, _ = data[ the_key ].shape

            # Sample somewherein the middle
            idx = np.random.randint( 16, n-total_frames )

            # Select the data around the sampled points
            data_sel = data[ the_key ][idx:idx+total_frames ,:]

            # Add the data
            # encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
            # decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
            # decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]


            encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len+self.target_seq_len-1, :]
            decoder_outputs[i,:,0:self.input_size] = data_sel[1:self.source_seq_len+self.target_seq_len, 0:self.input_size]


        return encoder_inputs, decoder_outputs

    def get_batch_srnn(self, data, action ):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                  "posing", "purchases", "sitting", "sittingdown", "smoking",
                  "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if not action in actions:
            raise ValueError("Unrecognized action {0}".format(action))

        frames = {}
        frames[ action ] = self.find_indices_srnn( data, action )

        batch_size = 8 # we always evaluate 8 seeds
        subject    = 5 # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

        encoder_inputs  = np.zeros((batch_size, self.source_seq_len+self.target_seq_len-1, self.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, self.source_seq_len+self.target_seq_len-1, self.input_size), dtype=float)

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len 

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in xrange( batch_size ):

            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[ (subject, action, subsequence, 'even') ]

            data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

            encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len+self.target_seq_len-1, :]
            decoder_outputs[i,:,0:self.input_size] = data_sel[1:self.source_seq_len+self.target_seq_len, 0:self.input_size]


            # encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
            # decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
            # decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


        return encoder_inputs, decoder_outputs, decoder_outputs[:, self.source_seq_len-1:,:]


    @staticmethod
    def find_indices_srnn(data, action ):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState( SEED )

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
        T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        return idx


    def rotmat2expmap(self, R):
        return self.quat2expmap( self.rotmat2quat(R) );


    @staticmethod
    def quat2expmap(q):
        """
        Converts a quaternion to an exponential map
        Matlab port to python for evaluation purposes
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

        Args
        q: 1x4 quaternion
        Returns
        r: 1x3 exponential map
        Raises
        ValueError if the l2 norm of the quaternion is not close to 1
        """
        if (np.abs(np.linalg.norm(q)-1)>1e-3):
            raise(ValueError, "quat2expmap: input quaternion is not norm 1")

        sinhalftheta = np.linalg.norm(q[1:])
        coshalftheta = q[0]

        r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
        theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
        theta = np.mod( theta + 2*np.pi, 2*np.pi )

        if theta > np.pi:
            theta =  2 * np.pi - theta
            r0    = -r0

        r = r0 * theta
        return r

    @staticmethod
    def rotmat2quat(R):
        """
        Converts a rotation matrix to a quaternion
        Matlab port to python for evaluation purposes
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

        Args
        R: 3x3 rotation matrix
        Returns
        q: 1x4 quaternion
        """
        rotdiff = R - R.T;

        r = np.zeros(3)
        r[0] = -rotdiff[1,2]
        r[1] =  rotdiff[0,2]
        r[2] = -rotdiff[0,1]
        sintheta = np.linalg.norm(r) / 2;
        r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

        costheta = (np.trace(R)-1) / 2;

        theta = np.arctan2( sintheta, costheta );

        q      = np.zeros(4)
        q[0]   = np.cos(theta/2)
        q[1:] = r0*np.sin(theta/2)
        return q

    @staticmethod
    def rotmat2euler( R ):
        """
        Converts a rotation matrix to Euler angles
        Matlab port to python for evaluation purposes
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

        Args
        R: a 3x3 rotation matrix
        Returns
        eul: a 3x1 Euler angle representation of R
        """
        if R[0,2] == 1 or R[0,2] == -1:
            # special case
            E3   = 0 # set arbitrarily
            dlta = np.arctan2( R[0,1], R[0,2] );

            if R[0,2] == -1:
                E2 = np.pi/2;
                E1 = E3 + dlta;
            else:
                E2 = -np.pi/2;
                E1 = -E3 + dlta;

        else:
            E2 = -np.arcsin( R[0,2] )
            E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
            E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

        eul = np.array([E1, E2, E3]);
        return eul

    @staticmethod
    def expmap2rotmat(r):
        """
        Converts an exponential map angle to a rotation matrix
        Matlab port to python for evaluation purposes
        I believe this is also called Rodrigues' formula
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

        Args
        r: 1x3 exponential map
        Returns
        R: 3x3 rotation matrix
        """
        theta = np.linalg.norm( r )
        r0  = np.divide( r, theta + np.finfo(np.float32).eps )
        r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
        r0x = r0x - r0x.T
        R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
        return R



    @staticmethod
    def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot ):
        """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
        https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

        Args
        normalizedData: nxd matrix with normalized data
        data_mean: vector of mean used to normalize the data
        data_std: vector of standard deviation used to normalize the data
        dimensions_to_ignore: vector with dimensions not used by the model
        actions: list of strings with the encoded actions
        one_hot: whether the data comes with one-hot encoding
        Returns
        origData: data originally used to
        """
        T = normalizedData.shape[0]
        D = data_mean.shape[0]

        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in xrange(D):
            if i in dimensions_to_ignore:
              continue
            dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        if one_hot:
            origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
        else:
            origData[:, dimensions_to_use] = normalizedData

        # potentially ineficient, but only done once per experiment
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat
        return origData

    def revert_output_format(self, poses, one_hot=False):
        """
        Converts the output of the neural network to a format that is more easy to
        manipulate for, e.g. conversion to other format or visualization

        Args
        poses: The output from the TF model. A list with (seq_length) entries,
        each with a (batch_size, dim) output
        Returns
        poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
        batch is an n-by-d sequence of poses.
        """
        seq_len = len(poses)
        if seq_len == 0:
            return []

       
        # batch_size, dim = poses.shape[1], poses.shape[2]

        # poses_out = np.concatenate(poses)
        # poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
        # poses_out = np.transpose(poses_out, [1, 0, 2])
        poses_out = poses

        poses_out_list = []
        for i in xrange(poses_out.shape[0]):
            poses_out_list.append(
                self.unNormalizeData(poses_out[i, :, :], self.data_mean, self.data_std, self.dim_to_ignore, self.actions, one_hot))

        return poses_out_list





                