import inspect
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear
from utils import SwitchableDropoutWrapper
from ran_cell import RANCell

class RecurrentVariationalAutoencoder(object):

  def __init__(self, batch_size, num_input, num_hidden, layer_depth, rnn_type, seq_length,
               learning_rate, keep_drop=0.5, grad_clip=5.0, is_training=False):
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.seq_length = seq_length
    self.batch_size = batch_size
    self.rnn_type = rnn_type
    self.layer_depth = layer_depth
    self.learning_rate = learning_rate
    self.grad_clip = grad_clip
    self.is_training = is_training
    self.keep_drop = keep_drop
    self.x = tf.placeholder(tf.float32, [batch_size, seq_length, self.num_input])

    # LSTM cells for encoder and decoder
    def create_cell():
      if rnn_type == "GRU":
        cell = rnn.GRUCell(num_hidden)
      elif rnn_type == "RAN":
        cell = RANCell(num_hidden, normalize=tf.constant(self.is_training))
      cell = SwitchableDropoutWrapper(cell, output_keep_prob=self.keep_drop, is_train=tf.constant(self.is_training))
      return cell

    with tf.variable_scope('encoder_cells', initializer=tf.contrib.layers.xavier_initializer()):
      self.enc_cell = rnn.DeviceWrapper(rnn.MultiRNNCell([create_cell() for _ in range(layer_depth)]), device="/gpu:0")

    with tf.variable_scope('decoder_cells', initializer=tf.contrib.layers.xavier_initializer()):
      self.dec_cell = rnn.DeviceWrapper(rnn.MultiRNNCell([create_cell() for _ in range(layer_depth)]), device="/gpu:1")

    with tf.variable_scope('encoder'):
      outputs, _ = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                     inputs=self.x,
                                     time_major=False,
                                     swap_memory=True,
                                     dtype=tf.float32)
      self.enc_output = outputs[:, -1, :]

    with tf.variable_scope('latent'):
      # reparametrization trick
      with tf.name_scope("Z"):
        self.z_mean = tf.contrib.layers.fully_connected(inputs=self.enc_output, num_outputs=num_hidden,
                                                        activation_fn=None, scope="z_mean")
        self.z_stddev = tf.contrib.layers.fully_connected(inputs=self.enc_output, num_outputs=num_hidden,
                                                          activation_fn=tf.nn.softplus, scope="z_ls2")

      # sample z from the latent distribution
      with tf.name_scope("z_samples"):
        with tf.name_scope('random_normal_sample'):
          eps = tf.random_normal((batch_size, num_hidden), 0, 1, dtype=tf.float32) # draw a random number
        with tf.name_scope('z_sample'):
          self.z = self.z_mean + tf.sqrt(self.z_stddev) * eps  # a sample it from Z -> z

    with tf.variable_scope('decoder'):
      reversed_inputs = tf.reverse(self.x, [1])
      flat_targets = tf.reshape(reversed_inputs, [-1])
      dec_first_inp = tf.nn.relu(_linear(self.z, self.num_input, True))

      # [GO, ...inputs]
      dec_inputs = tf.concat((tf.expand_dims(dec_first_inp, 1), reversed_inputs[:, 1:, :]), 1)
      self.w1 = tf.get_variable("w1", shape=[self.num_hidden, self.num_input],
                                initializer=tf.contrib.layers.xavier_initializer())
      self.b1 = tf.get_variable("b1", shape=[self.num_input], initializer=tf.constant_initializer(0.0))
      self.initial_state = self.dec_cell.zero_state(batch_size, dtype=tf.float32)
      dec_outputs, _ = tf.nn.dynamic_rnn(cell=self.dec_cell,
                                         inputs=dec_inputs,
                                         initial_state=self.initial_state,
                                         time_major=False,
                                         swap_memory=True,
                                         dtype=tf.float32)
    logist = tf.matmul(tf.reshape(dec_outputs, [-1, self.num_hidden]), self.w1) + self.b1
    self.reconstruction = tf.reshape(logist, [-1])
    self.reconstruction_loss = 0.5 * tf.reduce_mean(tf.pow(self.reconstruction - flat_targets, 2.0))
    self.latent_loss = -0.5 * (1.0 + tf.log(self.z_stddev) - tf.square(self.z_mean) - self.z_stddev)
    self.latent_loss = tf.reduce_sum(self.latent_loss, 1) / tf.cast(seq_length, tf.float32)
    self.latent_loss = tf.reduce_sum(self.latent_loss) / tf.cast(batch_size, tf.float32)
    self.cost = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.001)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

  def partial_fit(self, sess, X):
    cost, _ = sess.run((self.cost, self.train_op), feed_dict={self.x: X})
    return cost

  def calc_total_cost(self, sess, X):
    return sess.run(self.cost, feed_dict={self.x: X})

  def transform(self, sess, X):
    return sess.run(self.z, feed_dict={self.x: X})

  def generate(self, sess, hidden=None):
    if hidden is None:
      hidden = sess.run(tf.random_normal([1, self.num_hidden]))
    return sess.run(self.reconstruction, feed_dict={self.z: hidden})

  def reconstruct(self, sess, X):
    return sess.run(self.reconstruction, feed_dict={self.x: X})

  def getWeights(self, sess):
    return sess.run(self.w1)

  def getBiases(self, sess):
    return sess.run(self.b1)

if __name__ == "__main__":
  import numpy as np

  np.random.seed(1)
  tf.set_random_seed(1)

  batch_size = 32
  num_input = 200
  num_hidden = 128
  layer_depth = 2
  seq_length = 20
  rnn_type = 'RAN'
  epochs = 1000
  learning_rate = 1e-3

  rva = RecurrentVariationalAutoencoder(batch_size, num_input, num_hidden, layer_depth, rnn_type,
                                        seq_length, learning_rate, 0.5, 5.0, True)
  dataset = np.random.rand(batch_size, seq_length, num_input)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.global_variables_initializer().run()

    for e in range(epochs):
      cost = rva.partial_fit(sess, dataset)
      print('epoch:', e, 'Cost: ', cost)
