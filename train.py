#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import nltk


def load_word2vec(file_name, initW=None, vocab_processor=None):
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(file_name,
                                              binary=True)
    for word in model.wv.vocab:
        vector = model[word]
        if initW is None or vocab_processor is None:
            print word
            print "vocab processor and initW not initialized"
            break
        idx = vocab_processor.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = vector
    return initW, vocab_processor

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .01, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("dev_data_file", "", "Data source for the training data.")
#tf.flags.DEFINE_string("training_data_file", "/home/giuliano.tortoreto/slu/switchboard_data/switchboard_train_set_paper", "Data source for the training data.")
tf.flags.DEFINE_string("training_data_file", "/Users/gt/data/switchboard_train_set_paper", "Data source for the training data.")
##tf.flags.DEFINE_string("positive_data_file", "./complete_learn2.txt", "Data source for the training data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
#tf.flags.DEFINE_string("word2vec","/home/giuliano.tortoreto/GoogleNews-vectors-negative300.bin", "The path to the file containing word2vec vectors")
tf.flags.DEFINE_string("word2vec","/Users/gt/Projects/GoogleNews-vectors-negative300.bin", "The path to the file containing word2vec vectors")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, z, y = data_helpers.load_data_and_labels_dialog_act(FLAGS.training_data_file)

print("Features length (x,z,y)",len(x_text), len(z), len(y))

# Build vocabulary
tokenized_list_lens = [len(nltk.word_tokenize(x)) for x in x_text]
max_document_length = max(tokenized_list_lens)
avg_doc = np.mean(tokenized_list_lens)
stdev_doc = np.var(tokenized_list_lens)
print ("Document length (max,avg,stdev): ",max_document_length, avg_doc, stdev_doc)
#max_document_length = min(max_document_length, int(avg_doc+stdev_doc))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=data_helpers.tokenize)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
z_shuffled = z[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
if FLAGS.dev_data_file == "":
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    z_train, z_dev = z_shuffled[:dev_sample_index], z_shuffled[dev_sample_index:]
else:
    x_train = x_shuffled
    y_train = y_shuffled
    z_train = z_shuffled
    x_dev, z_dev, y_dev = data_helpers.load_data_and_labels_dialog_act(FLAGS.dev_data_file)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    #session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-5)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        #sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

        if FLAGS.word2vec:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
            # load any vectors from the word2vec
            print("Load word2vec file {}\n".format(FLAGS.word2vec))
            #data_helpers.append_to_additional_file("Load word2vec file {}\n".format(FLAGS.word2vec))
            initW, vocab_processor = load_word2vec(FLAGS.word2vec, initW, vocab_processor)
            """with open(FLAGS.word2vec, "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                data_helpers.append_to_additional_file("vocab size " + str(vocab_size))
                data_helpers.append_to_additional_file("layer1_size " + str(layer1_size))
                for line in range(vocab_size):
                    word = []
                    while isinstance(word, list):
                        ch = f.read(1)
                        #data_helpers.append_to_additional_file("Found character " + ch)
                        if ch == ' ':
                            word = ''.join(word)
                            #data_helpers.append_to_additional_file("Found word "+ word)
                        else:
                            if ch != '\n':
                                word.append(ch)
                    idx = vocab_processor.vocabulary_.get(word)
                    if idx != 0:
                        initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                    else:
                        f.read(binary_len)
            """

            sess.run(cnn.W.assign(initW))


        def train_step(x_batch, z_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_z: z_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, z_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_z: z_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter_da(
            list(zip(x_train, z_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, z_batch, y_batch = zip(*batch)
            train_step(x_batch, z_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, z_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))




def read_file(file_name):
    import numpy as np
    import sys
    if sys.version_info[0] < 3:
        with open(file_name, "rb",) as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            print("vocab size " + str(vocab_size))
            print("layer1_size " + str(layer1_size))
            for line in range(vocab_size):
                word = []
                while isinstance(word, list):
                    ch = f.read(1)
                    #print("Found character " + ch)
                    if ch == ' ':
                        word = ''.join(word)
                        print("Found word " + word)
                    else:
                        if ch != '\n':
                            word.append(ch)
    else:
        with open(file_name, "r",encoding='ISO-8859-1') as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            print("vocab size " + str(vocab_size))
            print("layer1_size " + str(layer1_size))
            for line in range(vocab_size):
                word = []
                while isinstance(word, list):
                    ch = f.read(1)
                    #print("Found character " + ch)
                    if ch == ' ':
                        word = ''.join(word)
                        print("Found word " + word)
                    else:
                        if ch != '\n':
                            word.append(ch)