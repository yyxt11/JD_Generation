
# coding: utf-8

# In[13]:


import helper
import numpy as np
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.contrib import seq2seq


# In[14]:


int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


#     建立NN网络

# In[15]:


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[16]:


def get_inputs():
    '''
    输入初始化
    '''
    input_data = tf.placeholder(tf.int32,[None,None],name='input')
    target_data = tf.placeholder(tf.int32,[None,None],name='target')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')
    return input_data, target_data, learning_rate


# In[17]:


def get_init_cell(batch_size, rnn_size):
    """
    初始化 RNN Cell.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm]*2)
    
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32),name='initial_state')
    
    return cell, initial_state


# In[18]:


def get_embed(input_data, vocab_size, embed_dim):
    """
    word embedding 输入.
    :param input_data:  输入.
    :param vocab_size: 总词语数.
    :param embed_dim: w2v 维数
    :return: Embedded input.
    """
    #embedding 初始化，这边不采用预先训练的embeding,边训练边调参数
    embedding = tf.Variable(tf.random_uniform((vocab_size,embed_dim),-1,1))
    embed = tf.nn.embedding_lookup(embedding,input_data)
    return embed


# In[19]:


def build_rnn(cell, inputs):
    """
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    Outputs,Finalstate = tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32)
    Final_state = tf.identity(Finalstate,"final_state")
    return Outputs,Final_state


# In[20]:


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embedding = get_embed(input_data,vocab_size,embed_dim)
    lstm_output,final_state = build_rnn(cell,embedding)
    #seq_output = tf.concat(lstm_output, axis=1)
    #x = tf.reshape(seq_output,[-1,rnn_size])
    #print(embedding.get_shape())
    #print(lstm_output.get_shape())
    
    #weights = tf.Variable(tf.truncated_normal([lstm_output.get_shape()[0].value,lstm_output.get_shape()[2].value,vocab_size], stddev=0.1))
    #bias = tf.Variable(tf.zeros(vocab_size))
    
    #print(weights.get_shape())
    #logits = tf.matmul(lstm_output,weights)+ bias
    logits = tf.contrib.layers.fully_connected(lstm_output,vocab_size,activation_fn=None)
    return logits,final_state


# In[21]:


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    batch_output = []
    characters_per_batch = seq_length*batch_size
    
    #print(characters_per_batch)
    batch_num = len(int_text)//characters_per_batch
    
    x_full_seqs = np.array(int_text[:batch_num*characters_per_batch])
    
    y_full_seqs = np.zeros_like(x_full_seqs)
      
    #bound limit
    if len(int_text) > batch_num*characters_per_batch: 
        y_full_seqs = int_text[1:batch_num*characters_per_batch + 1]
    else:
        y_full_seqs[:-1],y_full_seqs[-1] = int_text[1:batch_num*characters_per_batch],int_text[0]
    
    #reshape
    x_reshape = np.reshape(x_full_seqs,(batch_size,-1))
    y_reshape = np.reshape(y_full_seqs,(batch_size,-1))
    
   # print(x_reshape)
   # print(batch_num)
    #individual batches
    x_bathes = np.split(x_reshape,batch_num,1)
    y_bathes = np.split(y_reshape,batch_num,1)
                           
   # print(x_bathes[0])
   # print(y_bathes[0])
   
    for i in range(batch_num):  
        batch_output.append(np.stack((x_bathes[i],y_bathes[i])))
        
    return np.array(batch_output)


# In[22]:


#设置各种超参数
num_epochs = 30
batch_size = 256
rnn_size = 512
embed_dim = 400
seq_length = 20
learning_rate = 0.002

#打印间隔
show_every_n_batches = 30

#保存路径
save_dir = './save'


# In[23]:


#build the graph
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# In[24]:


#训练
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# In[25]:


#参数保留
helper.save_params((seq_length, save_dir))


# In[ ]:




