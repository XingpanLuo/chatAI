import sys
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import warnings
warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码

def download_nmt():
    path_to_file="/home/lumos/.keras/datasets/spa-eng/spa.txt"
    return path_to_file

class NMTDataset:
    def __init__(self, problem_type='en-spa'):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None


    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2 
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w)
        w = w.strip()
        w = '<start> ' + w + ' <end>'
        #print(w)
        return w

    def create_dataset(self, path, num_examples):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        #print(word_pairs)
        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn) 
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang) 

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences 
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)
        
        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
        warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
        file_path = download_nmt()
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer

class Encoder(tf.keras.Model):
  warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state = hidden)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, max_length_input,max_length_output,attention_type='luong'):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.attention_type = attention_type
    self.max_length_input=max_length_input
    self.max_length_output=max_length_output
    # Embedding Layer
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    #Final Dense layer on which softmax will be applied
    self.fc = tf.keras.layers.Dense(vocab_size)

    # Define the fundamental cell for decoder recurrent structure
    self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)



    # Sampler
    self.sampler = tfa.seq2seq.sampler.TrainingSampler()

    # Create attention mechanism with memory = None
    self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                              None, self.batch_sz*[self.max_length_input], self.attention_type)

    # Wrap attention mechanism with the fundamental rnn cell of decoder
    self.rnn_cell = self.build_rnn_cell(batch_sz)

    # Define the decoder with respect to fundamental rnn cell
    self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)


  def build_rnn_cell(self, batch_sz):
    rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                  self.attention_mechanism, attention_layer_size=self.dec_units)
    return rnn_cell

  def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
    if(attention_type=='bahdanau'):
      return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    else:
      return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

  def build_initial_state(self, batch_sz, encoder_state, Dtype):
    decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state


  def call(self, inputs, initial_state):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[self.max_length_output-1])
    return outputs


class Seq2seq(object):
  def __init__(self,EPOCHS,re_model=False) -> None:
    if(re_model):
      self.make_model(EPOCHS)
    else:
      self.make_model(0)
  def make_model(self,EPOCHS):
    self.BUFFER_SIZE = 32000
    BATCH_SIZE = 64
    self.BATCH_SIZE=BATCH_SIZE
    num_examples=30000
    dataset_creator = NMTDataset('zh-zh')
    self.dataset_creator=dataset_creator
    train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, self.BUFFER_SIZE, BATCH_SIZE)
    self.targ_lang=targ_lang
    self.inp_lang=inp_lang
    example_input_batch, example_target_batch = next(iter(train_dataset))
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    self.max_length_input = example_input_batch.shape[1]
    self.max_length_output = example_target_batch.shape[1]
    embedding_dim = 256
    units = 1024
    self.units=units
    steps_per_epoch = num_examples//BATCH_SIZE
    self.encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    sample_hidden = self.encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = self.encoder(example_input_batch, sample_hidden)
    self.decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, self.max_length_input,self.max_length_output,'luong')
    sample_x = tf.random.uniform((BATCH_SIZE, self.max_length_output))
    self.decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = self.decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)
    sample_decoder_outputs = self.decoder(sample_x, initial_state)
    optimizer = tf.keras.optimizers.Adam()
    self.optimizer=optimizer
    self.checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=self.encoder,
                                    decoder=self.decoder)
    if(EPOCHS):
      for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = self.encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
          batch_loss = self.train_step(inp, targ, enc_hidden)
          total_loss += batch_loss
          if batch % 100 == 0:
              print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
          self.checkpoint.save(file_prefix = checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  def translate(self,sentence):
    print("translate test in")
    self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
    result = self.evaluate_sentence(sentence)
    print("translate test out")
    result = self.targ_lang.sequences_to_texts(result)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
  
  def loss_function(self,real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)  
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

  #@tf.function
  def train_step(self,inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
      enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)


      dec_input = targ[ : , :-1 ] # Ignore <end> token
      real = targ[ : , 1: ]         # ignore <start> token

      # Set the AttentionMechanism object with encoder_outputs
      self.decoder.attention_mechanism.setup_memory(enc_output)

      # Create AttentionWrapperState as initial_state for decoder
      decoder_initial_state = self.decoder.build_initial_state(self.BATCH_SIZE, [enc_h, enc_c], tf.float32)
      pred = self.decoder(dec_input, decoder_initial_state)
      logits = pred.rnn_output
      loss = self.loss_function(real, logits)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return loss
  def evaluate_sentence(self,sentence):
    sentence = self.dataset_creator.preprocess_sentence(sentence)
    #print(sentence)
    inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                            maxlen=self.max_length_input,
                                                            padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
    enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index['<start>'])
    end_token = self.targ_lang.word_index['<end>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)
    # Setup Memory in decoder stack
    self.decoder.attention_mechanism.setup_memory(enc_out)

    # set decoder_initial_state
    decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)


    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder 
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this. 
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = self.decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token= end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()