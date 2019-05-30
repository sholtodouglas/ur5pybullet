#@title Trainer Class
# Think about casting to float explicitly

from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()

# print(tf2.enabled())

import tensorflow_probability as tfp

import tensorflow_datasets as tfds


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


from tensorflow.keras.layers import Dense, Flatten, Conv2D,Bidirectional, LSTM, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import Model
from tensorflow.keras.models import  Sequential
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm, tqdm_notebook
import random
from natsort import natsorted, ns
import imageio
from IPython import display
from PIL import Image

import IPython
import time
import seaborn as sns
import matplotlib.patheffects as PathEffects
import traceback

old = False



#@title Trainer Class
# Think about casting to float explicitly
class Model_Trainer():  
  
    ### CONSTRUCTOR ###
    def __init__(self, dataloader, EPOCHS, BETA, ALPHA, MIN_SEQ_LEN, MAX_SEQ_LEN,   
                 MAX_EVER_SEQ_LEN,  LATENT_DIM, LAYER_SIZE, ACTION_DIM, OBS_DIM ,
                 NUM_DISTRIBUTIONS, NUM_QUANTISATIONS,  DISCRETIZED,    RELATIVE,   
                 P_DROPOUT, BETA_ANNEAL, THRESHOLD, TEST, ARM_IN_GOAL):
        # Defaults
#         self.params = {'EPOCHS':30, 'BETA':0.0, 'ALPHA':0.0, 
#                       'MIN_SEQ_LEN':10, 'MAX_SEQ_LEN':10, 'MAX_EVER_SEQ_LEN':0, 
#                       'LATENT_DIM':60, 'LAYER_SIZE':1024,
#                       'ACTION_DIM':8, 'OBS_DIM':36,
#                       'NUM_DISTRIBUTIONS':3, 'NUM_QUANTISATIONS':256,
#                       'DISCRETIZED' : False, 'RELATIVE' : True,
#                       'P_DROPOUT' : 0.1, 'BETA_ANNEAL':1.0, 'THRESHOLD' : -30.0}
#         self.params.update(kwargs)
#         print(self.params)
        self.EPOCHS = EPOCHS
        self.BETA = BETA
        self.ALPHA  = ALPHA
        self.MIN_SEQ_LEN = MIN_SEQ_LEN
        self.MAX_SEQ_LEN = MAX_SEQ_LEN  
        self.MAX_EVER_SEQ_LEN = MAX_EVER_SEQ_LEN
        self.LATENT_DIM = LATENT_DIM
        self.LAYER_SIZE = LAYER_SIZE
        self.ACTION_DIM = ACTION_DIM
        self.ARM_IN_GOAL = ARM_IN_GOAL
        self.NUM_DISTRIBUTIONS = NUM_DISTRIBUTIONS
        self.NUM_QUANTISATIONS = NUM_QUANTISATIONS
        self.DISCRETIZED = DISCRETIZED
        self.RELATIVE = RELATIVE
        self.P_DROPOUT = P_DROPOUT
        self.BETA_ANNEAL = BETA_ANNEAL
        self.THRESHOLD = THRESHOLD
        self.BATCH_SIZE = 117
        self.ARM_IN_GOAL = ARM_IN_GOAL
        self.OBS_DIM = 14
        
#         self.dataset, self.valid_dataset, self.viz_dataset, self.viz_valid_dataset = dataloader.load_data()

        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        self.actor = ACTOR(self.LAYER_SIZE, self.ACTION_DIM, self.OBS_DIM, self.NUM_DISTRIBUTIONS, self.NUM_QUANTISATIONS, self.DISCRETIZED, self.P_DROPOUT)
        self.encoder = TRAJECTORY_ENCODER_LSTM(self.LAYER_SIZE, self.LATENT_DIM, self.P_DROPOUT)
        

        if TEST:
            pass
        else:
            self.TRAIN_LEN = dataloader.TRAIN_LEN
            self.VALID_LEN = dataloader.VALID_LEN
            self.train_summary_writer, self.test_summary_writer = self.tensorboard_logger()
            self.dataloader = dataloader

############################################ Keep up to here when replacing. #############################################

  
    def training_loop(self, i):

#         BETA = tf.constant(BETA) # dunno if needed

        IMI_val, KL_val, ent_val = np.Inf, np.Inf, np.Inf

        # this is shit make the OO formulation better so data is internal to dataloader class
        dataset = tf.data.Dataset.from_generator(self.dataloader.generator, (tf.float32, tf.float32, tf.float32, tf.int32), args = (self.dataloader.moments,'Train'))
        valid_dataset = tf.data.Dataset.from_generator(self.dataloader.generator, (tf.float32, tf.float32, tf.float32, tf.int32), args = (self.dataloader.moments,'Valid'))


        
        best_val_loss = 0
        
        for epoch in tqdm_notebook(range(self.EPOCHS), 'Epoch '):

            self.BETA *= self.BETA_ANNEAL

 
            train_set = dataset.shuffle(dataloader.TRAIN_LEN).batch(self.BATCH_SIZE)
            valid_set = iter(valid_dataset.shuffle(dataloader.VALID_LEN).batch(self.BATCH_SIZE))
            for obs,acts, mask, lengths in train_set:
                IMI, KL, gripper_loss = self.train_step(obs,acts,self.BETA, mask, lengths)
                  
                  
                print("\r",f"Epoch {epoch}\t| TRAIN | IMI: {float(IMI):.2f}, KL: {float(KL):.2f}\
                TEST | IMI: {float(IMI_val):.2f}, KL: {float(KL_val):.2f}, , entropy: {float(ent_val):.2f}",end="")

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('imi_loss', IMI, step=self.steps)
                    tf.summary.scalar('kl_loss', KL, step=self.steps)
                    tf.summary.scalar('gripper_loss',gripper_loss, step=self.steps)

                if self.steps % 50 == 0:
                    valid_obs, valid_acts, valid_mask, valid_lengths = valid_set.next()
                    IMI_val, KL_val, gripper_loss_val = self.test_step(valid_obs, valid_acts, valid_mask, valid_lengths)
                    
                    with self.test_summary_writer.as_default():
                        tf.summary.scalar('imi_loss', IMI_val, step=self.steps)
                        tf.summary.scalar('kl_loss', KL_val, step=self.steps)
                        tf.summary.scalar('gripper_loss',gripper_loss_val, step=self.steps)
                      


                self.steps+=1

            if IMI < best_val_loss:
                best_val_loss = IMI
                print('Saving Weights, best loss encountered')
                # This should be some test uid
                extension = 'drive/My Drive/Yeetbot_v3'
                self.save_weights(extension)
#                 worksheet.update_acell('S'+str(start_row+i), str(epoch)+', '+str(float(best_val_loss)))
        # Write results to google sheets or similar
        return float(best_val_loss)
        

    # add uid
    def tensorboard_logger(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-Beta:'+str(self.BETA)+'-Latent:'+str(self.LATENT_DIM)+'Drp'+str(self.P_DROPOUT)+'autotest'+'_'+str(self.MAX_SEQ_LEN)
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer
      
    def load_weights(self, extension):
        print('Loading in network weights...')
        # load some sample data to initialise the model
#         load_set = iter(self.dataset.shuffle(self.TRAIN_LEN).batch(self.BATCH_SIZE))
        obs  = tf.zeros((self.BATCH_SIZE,self.MAX_SEQ_LEN,self.OBS_DIM))
        acts = tf.zeros((self.BATCH_SIZE,self.MAX_SEQ_LEN,self.ACTION_DIM))
        mask = acts
        lengths = tf.cast(tf.ones(self.BATCH_SIZE), tf.int32)
    
        _, _, _= self.test_step(obs, acts, mask, lengths)
   
    
#         self.planner.load_weights(extension+'/planner.h5')
        self.encoder.load_weights(extension+'/encoder.h5')
        self.actor.load_weights(extension+'/actor.h5')
        print('Loaded.')
#         return actor, planner, encoder


    def save_weights(self, extension):
        try:
            os.mkdir(extension)
        except Exception as e:
            #print(e)
            pass

        #print(extension)
        self.actor.save_weights(extension+'/actor.h5')
#         self.planner.save_weights(extension+'/planner.h5')
        self.encoder.save_weights(extension+'/encoder.h5')

    # INFOVAE MMD loss
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
      
        
    @tf.function
    def compute_loss(self, normal_enc, z, obs, acts, s_g,BETA, mu_enc, s_enc, mask, lengths, training=False):
        AVG_SEQ_LEN = obs.shape[1]
        CURR_BATCH_SIZE = obs.shape[0]
        # Automatically averaged over batch size i.e. SUM_OVER_BATCH_SIZE

        true_samples = tf.random.normal(tf.stack([CURR_BATCH_SIZE, self.LATENT_DIM]))
        # MMD
        # Need to convert normal_enc from tfp.Normal_distrib to tensor of sampled values
        #std_normal=  tfd.Normal(0,2)
        #batch_avg_mean = tf.reduce_mean(mu_plan, axis = 0) #m_enc will batch_size, latent_dim. We want average mean across the batches so we end up with a latent dim size avg_mean_vector. Each dimension of the latent dim should be mean 0 avg across the batch, but individually can be different.
        #batch_avg_s = tf.reduce_mean(s_plan,axis=0)
        #batch_avg_normal = tfd.Normal(batch_avg_mean, batch_avg_s)
        #info_kl = tf.reduce_sum(tfd.kl_divergence(batch_avg_normal, std_normal))
        info_kl = self.compute_mmd(true_samples, normal_enc.sample() )
        KL = info_kl

        #KL = tf.reduce_sum(tfd.kl_divergence(normal_enc, normal_plan))/CURR_BATCH_SIZE #+ tf.reduce_sum(tfd.kl_divergence(normal_plan, normal_enc)) #KL divergence between encoder and planner distirbs
        #KL_reverse = tf.reduce_sum(tfd.kl_divergence(normal_plan, normal_enc))/CURR_BATCH_SIZE
        IMI = 0
        OBS_pred_loss = 0
        
        
        #s_g_dim= s_g.shape[-1]
        #s_g = tf.tile(s_g, [1,self.MAX_SEQ_LEN])
        #s_g = tf.reshape(s_g, [-1, self.MAX_SEQ_LEN,s_g_dim ]) #subtract arm rel states
        z = tf.tile(z, [1, self.MAX_SEQ_LEN])
        z = tf.reshape(z, [-1, self.MAX_SEQ_LEN, self.LATENT_DIM]) # so that both end up as BATCH, SEQ, DIM
        
        mu, scale, prob_weight, pdf, obs_pred, gripper= self.actor(obs,z,s_g, training = training)
        
        
        
        log_prob_actions = -pdf.log_prob(acts[:,:,:self.ACTION_DIM-1]) # batchsize, Maxseqlen, actions, 
        masked_log_probs = log_prob_actions*mask[:,:,:self.ACTION_DIM-1] # should zero out all masked elements.
        avg_batch_wise_sum = tf.reduce_sum(masked_log_probs, axis = (1,2)) / lengths
        IMI = tf.reduce_mean(avg_batch_wise_sum)
        
        
        #sampled_acts = pdf.sample()
        #MSE_action_loss = tf.losses.MSE(tf.squeeze(acts[:,:,:self.ACTION_DIM-1])*tf.squeeze(mask[:,:,:self.ACTION_DIM-1]),tf.squeeze(sampled_acts)*tf.squeeze(mask[:,:,:self.ACTION_DIM-1]))
        #IMI = tf.reduce_mean(MSE_action_loss)
        
        gripper_loss = tf.losses.MAE(tf.squeeze(acts[:,:,self.ACTION_DIM-1])*tf.squeeze(mask[:,:,0]),tf.squeeze(gripper)*tf.squeeze(mask[:,:,0]))
        gripper_loss = tf.reduce_mean(gripper_loss) # lets go quartic and see if its more responsive

    
        loss = IMI + 50*gripper_loss + self.BETA*KL #+ (self.BETA/10)* info_kl#+ self.BETA*KL_reverse#ALPHA*entropy# + OBS_pred_loss*self.ALPHA

        #print(loss)
        #print(gripper_loss)
        return loss, IMI, KL, gripper_loss



    @tf.function
    def train_step(self, obs, acts, BETA, mask, lengths):
        with tf.GradientTape() as tape:

            # obs and acts are a trajectory, so get intial and goal
            s_i = obs[:,0,:]
            range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
            expanded_lengths = tf.expand_dims(lengths-1,1)# lengths must be subtracted by 1 to become indices.
            s_g = tf.gather_nd(obs, tf.concat((range_lens, expanded_lengths),1)) # get the actual last element of the sequencs.
            if self.ARM_IN_GOAL:
              s_g = s_g[:,:-6] #don't inc rel states
            else:
              s_g = s_g[:,8:-6] #don't inc rel states or arm states in goal

            # Encode the trajectory
            mu_enc, s_enc = self.encoder(obs, acts, training = True)
            encoder_normal = tfd.Normal(mu_enc,s_enc)
            z = encoder_normal.sample()

            #Produce a plan from the inital and goal state
            #mu_plan, s_plan = self.planner(s_i,s_g, training = True)
            #planner_normal = tfd.Normal(mu_plan,s_plan)
            #zp = planner_normal.sample()

            lengths = tf.cast(lengths, tf.float32)
            loss, IMI, KL, gripper_loss = self.compute_loss(encoder_normal, z, obs, acts, s_g, self.BETA, mu_enc, s_enc, mask, lengths, training = True)
            #find and apply gradients with total loss
        gradients = tape.gradient(loss,self.encoder.trainable_variables+self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.encoder.trainable_variables+self.actor.trainable_variables))

        # return values for diagnostics
        return IMI, KL, gripper_loss


    @tf.function
    def test_step(self, obs, acts, mask, lengths):   
        s_i = obs[:,0,:]
        range_lens = tf.expand_dims(tf.range(tf.shape(lengths)[0]), 1)
        expanded_lengths = tf.expand_dims(lengths-1,1)# lengths must be subtracted by 1 to become indices.
        s_g = tf.gather_nd(obs, tf.concat((range_lens, expanded_lengths),1)) # get the actual last element of the sequencs.
        if self.ARM_IN_GOAL:
              s_g = s_g[:,:-6] #don't inc rel states
        else:
              s_g = s_g[:,8:-6] #don't inc rel states or arm states in goal
        # Encode Trajectory 
        mu_enc, s_enc = self.encoder(obs, acts)
        encoder_normal = tfd.Normal(mu_enc,s_enc)
        z = encoder_normal.sample()

        # PLan with si,sg.
        #mu_plan, s_plan = self.planner(s_i,s_g)
        #planner_normal = tfd.Normal(mu_plan,s_plan)
        #zp = planner_normal.sample()

        lengths = tf.cast(lengths, tf.float32)
        _, IMI, KL, gripper_loss = self.compute_loss(encoder_normal, z, obs, acts, s_g, self.BETA, mu_enc, s_enc, mask, lengths)
    
        return IMI, KL, gripper_loss
    


class TRAJECTORY_ENCODER_LSTM(Model):
  def __init__(self, LAYER_SIZE, LATENT_DIM, P_DROPOUT):
    super(TRAJECTORY_ENCODER_LSTM, self).__init__()

    self.bi_lstm = Bidirectional(CuDNNLSTM(LAYER_SIZE, return_sequences=True), merge_mode=None)
    self.mu = Dense(LATENT_DIM)
    self.scale = Dense(LATENT_DIM, activation='softplus')
    self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)

  def call(self, obs, acts, training = False):
    x = tf.concat([obs,acts], axis = 2) # concat observations and actions together.
    x = self.bi_lstm(x)
    x = self.dropout1(x, training=training)
    bottom = x[0][:,-1, :] # Take the last element of the bottom row
    top = x[1][:,0,:] # Take the first elemetn of the top row cause Bidirectional, top row goes backward.
    x = tf.concat([bottom, top], axis = 1)
    mu = self.mu(x)
    s = self.scale(x)

    return mu, s



class ACTOR(Model):
  def __init__(self, LAYER_SIZE, ACTION_DIM, OBS_DIM, NUM_DISTRIBUTIONS, NUM_QUANTISATIONS, DISCRETIZED, P_DROPOUT):
    super(ACTOR, self).__init__()
    self.ACTION_DIM = ACTION_DIM-1
    self.OBS_DIM = OBS_DIM
    self.NUM_DISTRIBUTIONS = NUM_DISTRIBUTIONS
    self.DISCRETIZED = DISCRETIZED
    self.NUM_QUANTISATIONS = NUM_QUANTISATIONS
    
    self.RNN1 = CuDNNLSTM(LAYER_SIZE, return_sequences=True, return_state = True)
    self.RNN2 = CuDNNLSTM(LAYER_SIZE, return_sequences=True, return_state = True)
    self.mu = Dense(self.ACTION_DIM*self.NUM_DISTRIBUTIONS) #means of our logistic distributions
    # softplus activations are to ensure positive values for scale and probability weighting.
    self.scale = Dense(self.ACTION_DIM*self.NUM_DISTRIBUTIONS,activation='softplus') # scales of our logistic distrib
    self.prob_weight = Dense(self.ACTION_DIM*self.NUM_DISTRIBUTIONS,activation='softplus') # weightings on each of the distribs.
#     self.next_obs_pred = Dense(self.OBS_DIM)
    self.gripper = Dense(1)
    self.dropout1 = tf.keras.layers.Dropout(P_DROPOUT)
    

  def call(self, s, z, s_g, training = False, past_state = None):
      B = z.shape[0] #dynamically get batch size
      s_g=tf.zeros((z.shape))
      state_out = None
      if len(s.shape) == 3:
          x = tf.concat([s, z, s_g], axis = 2) # (BATCHSIZE)
          [x, _, _] = self.RNN1(x)
          [x, _, _] = self.RNN2(x)
          x = self.dropout1(x, training=training)

      else:
          x = tf.concat([s, z, s_g], axis = 1) # (BATCHSIZE,  OBS+OBS+LATENT)
          x= tf.expand_dims(x, 1) # make it (BATCHSIZE,  1, OBS+OBS+LATENT) so LSTM is happy.
          [x, s1l1, s2l1] = self.RNN1(x, initial_state = past_state[0])
          [x, s1l2, s2l2] = self.RNN2(x, initial_state = past_state[1])
          state_out = [[s1l1, s2l1], [s1l2, s2l2]]
          x = self.dropout1(x, training=training)


      mu = tf.reshape(self.mu(x), [B,-1,self.ACTION_DIM, self.NUM_DISTRIBUTIONS])
      #mu = tf.concat([mu[:,:,:7,:],tf.keras.activations.sigmoid(tf.expand_dims(mu[:,:,7,:], 2))], axis = 2)
      scale = tf.reshape(self.scale(x), [B,-1,self.ACTION_DIM, self.NUM_DISTRIBUTIONS])
      prob_weight = tf.reshape(self.prob_weight(x), [B,-1,self.ACTION_DIM, self.NUM_DISTRIBUTIONS])

      if self.DISCRETIZED:
          # multiply mean by 64 so that for a neuron between -2 and 2, it covers the full range 
          # between -128 and 128, i.e so the weights don't have to be huge! in the mean layer, but tiny in the s and prob layers.
          mu = mu*self.NUM_QUANTISATIONS/4
          discretized_logistic_dist = tfd.QuantizedDistribution(distribution=tfd.TransformedDistribution(
              distribution=tfd.Logistic(loc=mu,scale=scale),bijector=tfb.AffineScalar(shift=-0.5)),
          low=-self.NUM_QUANTISATIONS/2,
          high=self.NUM_QUANTISATIONS/2)

          # should be batch size by action dim
          distributions = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
                  probs=prob_weight),components_distribution=discretized_logistic_dist)
      else:
          logistic_dist = tfd.Logistic(loc=mu,scale=scale)

          distributions = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
                  probs=prob_weight),components_distribution=logistic_dist)



      obs_pred = None#self.next_obs_pred(x)
      grip = tf.keras.activations.sigmoid(self.gripper(x))

      if state_out == None:
          return mu,scale, prob_weight, distributions, obs_pred, grip
      else:
          return mu,scale, prob_weight, distributions, obs_pred, state_out, grip