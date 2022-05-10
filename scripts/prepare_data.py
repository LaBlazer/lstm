import os
import itertools
import functools
import tensorflow as tf
import numpy as np
import array

from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

tf.compat.v1.flags.DEFINE_string('f','','')

tf.flags.DEFINE_integer(
  "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original data files (default = './data')")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data"),
  "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.txt")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.txt")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.txt")

def tokenizer_fn(iterator):
  return (tokenizer(x) if x != '<EOS>' else ['<EOS>'] for x in iterator)


def read_support_dialogues(f_path):
    dialogues = []
    with open(f_path, encoding='utf-8') as f:
        for line in f.readlines():
            
            
            item_arr = line.strip().split('\t')
            
            if len(item_arr) < 2:
                pass
            
            context = [u.strip() for u in item_arr[1:-1]]
            response = item_arr[-1].strip()
            label = int(item_arr[0].strip())
            
            dialogues.append((context, response, label))

    return dialogues


def create_vocab(input_iter, min_frequency):
  """
  Creates and returns a VocabularyProcessor object with the vocabulary
  for the input iterator.
  """
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      FLAGS.max_sentence_len,
      min_frequency=min_frequency,
      tokenizer_fn=tokenizer_fn)
  vocab_processor.fit(input_iter)
  return vocab_processor


def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary. Returns a python array.
  """
  return next(vocab_processor.transform(sequence)).tolist()

def insert_eos(utterances):
  tr = []
  for u in utterances:
    tr.append(u)
    tr.append('<EOS>')
  return tr


def create_train(row, vocab):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, response, label = row
  label = int(float(label))
  #context = insert_eos(context)
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(response, vocab)
  context_len = len(next(vocab._tokenizer(context)))
  utterance_len = len(next(vocab._tokenizer(response)))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example



def create_tfrecords_train_file(rows, output_filename, vocab):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for row in rows:
    x = create_train(row, vocab)
    writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def create_test(rows, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """

  # New Example
  example = tf.train.Example()

  for i, r in enumerate(rows):
    context, response, label = r
    #context = insert_eos(context)
    label = int(float(label))

    context_transformed = transform_sentence(context, vocab)
    utterance_transformed = transform_sentence(response, vocab)
    context_len = len(next(vocab._tokenizer(context)))
    utterance_len = len(next(vocab._tokenizer(response)))

    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])

    if label == 1:
      example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
      example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
    else:
      dis_key = "distractor_{}".format(i)
      dis_len_key = "distractor_{}_len".format(i)
      example.features.feature[dis_len_key].int64_list.value.extend([utterance_len])
      example.features.feature[dis_key].int64_list.value.extend(utterance_transformed)
  return example

def create_tfrecords_test_file(rows, output_filename, vocab, n=10):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i in range(0, len(rows), n):
    batch = rows[i:i+n]
    x = create_test(batch, vocab)
    writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
  """
  Writes the vocabulary to a file, one word per line.
  """
  vocab_size = len(vocab_processor.vocabulary_)
  with open(outfile, "w") as vocabfile:
    for id in range(vocab_size):
      word = vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
  print("Creating vocabulary...")
  train_dialogues = read_support_dialogues(TRAIN_PATH)
  valid_dialogues = read_support_dialogues(VALIDATION_PATH)
  test_dialogues = read_support_dialogues(TEST_PATH)
  all_utterances = [' '.join(c) + ' ' + r for c, r, _ in train_dialogues + valid_dialogues + test_dialogues]
  #all_utterances.extend(['<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>', '<EOS>']) # nepozeraj sa sem

  vocab = create_vocab(all_utterances, min_frequency=FLAGS.min_word_frequency)
  print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))


  # Create vocabulary.txt file
  write_vocabulary(
    vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

  # Save vocab processor
  vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

  # Create validation.tfrecords
  create_tfrecords_test_file(
      valid_dialogues,
      output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"), vocab=vocab)

  # Create test.tfrecords
  create_tfrecords_test_file(
      test_dialogues,
      output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"), vocab=vocab)

  # Create train.tfrecords
  create_tfrecords_train_file(
      train_dialogues,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"), vocab=vocab)
