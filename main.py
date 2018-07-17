from __future__ import print_function, division, absolute_import
import os
from shutil import copyfile
import numpy as np
from datetime import datetime
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, TensorBoard, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback, CSVLogger, Callback

import data_helper
import RENLayer
import clr


optimizer = Adam(lr=.01, clipnorm=40)

start = datetime.now()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'source_path',
    'data/tasks_1-20_v1-2.tar.gz',
    'Tar containing bAbI sources.')
tf.app.flags.DEFINE_string('dataset_id', 'qa6', 'Dataset destination.')
tf.app.flags.DEFINE_boolean('only_1k', False, 'Whether to use bAbI 1k or bAbI 10k (default).')
tf.app.flags.DEFINE_string('filename', '0', 'Filename of log: ID GPU Description')

logdir = 'logs/'+FLAGS.filename+'/'
if not os.path.exists(logdir): os.makedirs(logdir)
copyfile('main.py', logdir+'main.py')
copyfile('data_helper.py', logdir+'data_helper.py')
copyfile('RENLayer.py', logdir+'RENLayer.py')


# Model parameters
EMBED_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS     = 200
NUM_BLOCKS = 20

PATIENCE   = 200
LRPATIENCE = 30

# Cyclic Learning Rate
# clr = clr.CyclicLR(base_lr=2e-4, max_lr=1e-2, step_size=3000., mode='triangular')
# clr = clr.CyclicLR(base_lr=0.001, max_lr=0.01, step_size=3000., mode='triangular')
# options: [triangular, triangular2, exp_range]

# --------------------------------- PREPARE DATA ---------------------------------------------------
train, test, params = data_helper.get_data(FLAGS, batch_size=BATCH_SIZE)
x, xq, y = train[0], np.expand_dims(train[1], 1), train[2]
tx, txq, ty = test[0], np.expand_dims(test[1], 1), test[2]

max_sentence_length = params["max_sentence_length"]
story_maxlen = params["story_maxlen"]
query_maxlen = params["query_maxlen"]
vocab_size = params["vocab_size"]

vocab_size += NUM_BLOCKS
# ========================================== BUILD KERAS MODEL ======================================
print('Build model...')

# ------------------------------------------ STORY INIT ---------------------------------------------
# Story
# Define input
sentence = layers.Input(shape=(story_maxlen, max_sentence_length,), dtype='int32')
# ------------------------------------------ QUERY INIT ---------------------------------------------
# Query
question = layers.Input(shape=(1, query_maxlen,), dtype='int32')

# ------------------------------------------- EMBEDDINGS --------------------------------------------

# create embedding and masking layers
embed_1 = RENLayer.RENEmbed(vocab_size=vocab_size, embedding_size=EMBED_HIDDEN_SIZE, sentence_len=max_sentence_length)
embed_2 = RENLayer.RENEmbed(vocab_size=vocab_size, embedding_size=EMBED_HIDDEN_SIZE, sentence_len=query_maxlen)

mask_1 = RENLayer.RENMask(vocab_size=vocab_size, embedding_size=EMBED_HIDDEN_SIZE, sentence_len=max_sentence_length)
mask_2 = RENLayer.RENMask(vocab_size=vocab_size, embedding_size=EMBED_HIDDEN_SIZE, sentence_len=query_maxlen)

activation = PReLU(alpha_initializer='ones')

# embed sentence and question
encoded_sentence = embed_1(sentence)
encoded_question = embed_1(question)

print('encoded_sentence', encoded_sentence)
print('encoded_sentence', encoded_question)

encoded_sentence = mask_1(encoded_sentence)
encoded_question = mask_2(encoded_question)

print('masked encoded_sentence', encoded_sentence)
print('masked encoded_question', encoded_question)

# initialize keys
# keys = [tf.get_variable("Key_%d" % i, [EMBED_HIDDEN_SIZE], initializer=tf.random_normal_initializer(stddev=0.1))
#         for i in range(NUM_BLOCKS)]


def get_keys(x):
    keys = [key for key in range(vocab_size - NUM_BLOCKS, vocab_size)]
    return tf.squeeze(tf.reshape(keys, [1, -1]))


def get_keys_shape(input_shape):
    return NUM_BLOCKS,

# keys = get_keys(None)
keys = layers.Lambda(get_keys, output_shape=get_keys_shape)(encoded_sentence)
keys = embed_1(keys)
print('embedded_keys', keys)

keys = tf.split(keys, NUM_BLOCKS, axis=0)
keys = [tf.squeeze(key, axis=0) for key in keys]

# create the main Recurrent Entity Network cells
last_state = RENLayer.REN(initial_batch_size=BATCH_SIZE,
                          units=EMBED_HIDDEN_SIZE,
                          num_blocks=NUM_BLOCKS,
                          num_units_per_block=EMBED_HIDDEN_SIZE,
                          vocab_size=vocab_size,
                          keys=keys,
                          activation=activation,
                          initializer='normal')(encoded_sentence)

print('last_state', last_state)

# create output layer and get predictions
preds = RENLayer.RENL(embedding_size=EMBED_HIDDEN_SIZE,
                      vocab_size=vocab_size,
                      num_blocks=NUM_BLOCKS, activation=activation)([last_state, encoded_question])

print('logits', preds)
# apply activation
# preds = layers.Dropout(0.3)(preds) # uncomment this in case the model is over fitting
preds = layers.Activation('softmax')(preds)

model = Model([sentence, question], preds)

# compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print out model summary
model.summary()

# fit the model with story, query and answers
print('Training the model..')

def callbacks():
		checkpoint    = ModelCheckpoint(logdir + 'bestcheckpoint' +'.hdf5', monitor='val_loss', verbose=1, save_best_only=True) # {epoch:02d}-{val_acc:.2f}
		tensorboard   = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False, write_images=True)
		earlystopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
		history       = History()
		lrplateau     = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LRPATIENCE, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-9)
		# plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))
		csvlogger     = CSVLogger(logdir+'csvlogger.csv', separator=',', append=True)
		class prediction_history(Callback):
		    def __init__(self):
		        self.predhis = []
		    def on_epoch_end(self, epoch, logs={}):
		    	pred = model.predict([tx, txq])
		        self.predhis.append(pred)
		        print(pred)
		        # if CLASSIFICATION:
		        # 	print(pred[:,1]) # np.array_str(    precision=2, suppress_small=True).replace('\n','')
		        # else:
		        # 	print(pred[:,0])
		prediction_history = prediction_history()
		return [checkpoint, tensorboard, earlystopping, history, lrplateau, csvlogger, prediction_history]


# clr applied and fitting the model
hist = model.fit([x, xq], y,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 # callbacks=[clr],
                 callbacks=callbacks(),
                 validation_data=([tx, txq], ty))


# ----------------------------------------- EVALUATE MODEL -------------------------------------
# eval
print('Evaluate model...')
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)

# get model accuracy
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


# -------------------------------------- LOG SUMMARY ------------------------------------------
time_taken = datetime.now() - start

if FLAGS.only_1k:
    dataset_size = '1k'
else:
    dataset_size = '10k'

with open("logs.txt", "a") as file:
    data = "Dataset: " + str(FLAGS.dataset_id) + "_" + str(dataset_size) + ", Loss:" + str(loss) + ", Accuracy:" + str(acc) \
           + ", Epochs:" + str(EPOCHS) + ", Time Taken To Train: " + str(time_taken) + "\n"
    file.write(data)
