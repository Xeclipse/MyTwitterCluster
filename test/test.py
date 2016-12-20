from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
import tensorflow as tf

label=tf.Variable([[1,2,3],[2,3,4]])

norm = tf.reduce_sum(label, 1)
norm = tf.add(norm, tf.Variable([1] * 2))