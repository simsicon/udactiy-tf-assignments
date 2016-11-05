import tensorflow as tf
import numpy as np
import zipfile
import os
import pdb

flags = tf.app.flags

flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g text8.zip")

flags.DEFINE_boolean(
    "decompress", False,
    "If training data is zip compressed, pass this flag.")

FLAGS = flags.FLAGS

class Bigrams2Vec(object):
    def __init__(self, train_data_file, decompress=False, batch_size=128, window=4, emb_dim=256):
        _text = self.read_train_data_file(train_data_file, decompress)
        self.text = _text
        self.bigrams = [_text[2 * i] + _text[2 * i + 1] for i in range(len(_text) / 2)]
        self._bigrams_len = len(self.bigrams)
        self.vocabulary = np.array(sorted(list(set(self.bigrams))))
        self._vocabulary_size = len(self.vocabulary)
        self.dictionary = dict(zip(self.vocabulary, range(self._vocabulary_size)))
        self._batch_size = batch_size
        self._window = window
        self._batch_cursor = 0
        self._emb_dim = emb_dim
        self._num_steps = 10000001
        self.build_graph()

    def train(self):
        graph = self._graph
        learning_rate = self._learning_rate

        ckpt_dir = "checkpoints/"
        ckpt_filename = "bigrams2vec.ckpt"

        with tf.Session(graph=graph) as sess:
            init_op = tf.initialize_all_variables()
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            saver = tf.train.Saver()

            sess.run(init_op)

            def train_loop(start_at=0):
                average_loss = 0
                for step in range(start_at, self._num_steps, 1):
                    batch_data, batch_labels = self.batch_bigrams()
                    batch_labels = np.reshape(batch_labels, (self._batch_size, 1))
                    feed_dict = {self._train_dataset : batch_data, self._train_labels : batch_labels}
                    _, _l, _lr = sess.run([self._optimizer, self._loss, learning_rate], feed_dict=feed_dict)
                    average_loss += _l

                    if step % 1000 == 0:
                        if step > 0:
                            average_loss = average_loss / 1000
                            print('Average loss at step %d: %f with learning rate %f' %
                                  (step, average_loss, _lr))
                            average_loss = 0

                    if step % 100000 == 0:
                        if step > 0:
                            save_path = saver.save(sess, ckpt_dir + ckpt_filename,
                                                   global_step=self._global_step)
                            print "Bigrams2Vec Model saved in file: %s" % save_path

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print "Bigrams2Vec Model load from file: %s" % ckpt.model_checkpoint_path
                _current_step = self._global_step.eval()
                if _current_step < self._num_steps:
                    train_loop(start_at=_current_step)
            else:

                print("bigrams len: %d" % self._bigrams_len)
                train_loop()

            save_path = saver.save(sess, ckpt_dir + ckpt_filename)
            print "Bigrams2Vec Model saved in file: %s" % save_path

            _embs_table = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope='bigrams/embeddings/table')[0]
            self.embeddings = _embs_table.eval()

    def evaluate(self):
        pass

    def build_graph(self):
        self._graph = tf.Graph()
        batch_size, voc_size, emb_dim = self._batch_size, self._vocabulary_size, self._emb_dim
        num_sampled = 128
        with self._graph.as_default():
            with tf.variable_scope("bigrams"):
                self._train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
                self._train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

                with tf.variable_scope("embeddings"):
                    embeddings = tf.get_variable("table", [voc_size, emb_dim],
                                                initializer=tf.random_uniform_initializer(-1.0, 1.0))
                with tf.variable_scope("softmax"):
                    _initializer=tf.truncated_normal_initializer(stddev=1.0/np.sqrt(emb_dim))
                    weights = tf.get_variable("weights", [voc_size, emb_dim],
                                             initializer=_initializer)
                    biases = tf.get_variable("biases",
                                            initializer=tf.zeros_initializer([voc_size]))
            emb = tf.nn.embedding_lookup(embeddings, self._train_dataset)
            self._loss = tf.reduce_mean(
                        tf.nn.sampled_softmax_loss(
                            weights, biases, emb, self._train_labels, num_sampled, voc_size))
            self._global_step = tf.Variable(0, trainable=False)
            self._learning_rate = tf.train.exponential_decay(10.0, self._global_step, 50000, 0.95, staircase=True)
            self._optimizer = tf.train.AdagradOptimizer(self._learning_rate).minimize(self._loss, global_step=self._global_step)

    def batch_bigrams(self):
        batch_size, window, d = self._batch_size, self._window, self.dictionary
        span = 2 * window + 1
        batches = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)

        for i in range(batch_size // (2 * window)):
            chunk = self.bigrams[self._batch_cursor:(self._batch_cursor + span)]
            target = d[chunk[window]]
            for j in range(window):
                try:
                    batch_num = i * 2 * window + j

                    batches[batch_num] = d[chunk[j]]
                    labels[batch_num] = target

                    batches[batch_num + window] = d[chunk[span - j - 1]]
                    labels[batch_num + window] = target
                except IndexError:
                    pdb.set_trace()
                    print "oh man!"

            self._batch_cursor = (self._batch_cursor + 1) % self._bigrams_len
            if (self._batch_cursor + span) > self._bigrams_len:
                self._batch_cursor = 0

        return batches, labels

    def read_train_data_file(self, train_data_file, decompress=False):
        if decompress:
            with zipfile.ZipFile(train_data_file) as f:
                data = tf.compat.as_str(f.read(f.namelist()[0]))
        else:
            with open(train_data_file, "rb") as f:
                data = tf.compat.as_str(f.read())
        return data


class BatchGenerator(object):
    def __init__(self, embeddings, vocabulary, dictionary, text, batch_size, num_unrollings):
        self._text = text
        self._embeddings = embeddings
        self._emb_dim = embeddings.shape[1]
        self._vocabulary = vocabulary
        self._dictionary = dictionary
        self._bigrams_size = len(text) - 1
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._bigrams_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch, self._last_batch_ids = self._next_batch()

    def _next_batch(self):
        _bigrams = [self._text[c:c+2] for c in self._cursor]
        ids = [self._dictionary[bg] for bg in _bigrams]
        batch = self._embeddings[ids]
        for b in range(self._batch_size):
            self._cursor[b] = (self._cursor[b] + 2) % self._bigrams_size
        return batch, ids

    def next(self):
        batches = [self._last_batch]
        ids = [self._last_batch_ids]
        for step in range(self._num_unrollings):
            _bs, _is = self._next_batch()
            batches.append(_bs)
            ids.append(_is)
        self._last_batch = batches[-1]
        self._last_batch_ids = ids[-1]
        return batches, ids

class LSTM(object):
    def __init__(self, embeddings, vocabulary, dictionary, text, num_nodes=256, batch_size=128):
        self._embeddings = embeddings
        self._emb_dim = embeddings.shape[1]
        self._vocabulary = vocabulary
        self._dictionary = dictionary
        self._text = text
        self._vocabulary_size = len(vocabulary)
        self._num_nodes = num_nodes
        self._batch_size = batch_size
        self._num_unrollings = 4
        self._num_steps = 2000001
        self.build_graph()
        self.batch_generator = BatchGenerator(self._embeddings, self._vocabulary, self._dictionary,
                                              self._text, self._batch_size, self._num_unrollings)

    def train(self):
        num_steps = self._num_steps
        summary_frequency = 100
        num_unrollings = self._num_unrollings

        graph = self._graph
        with tf.Session(graph=graph) as sess:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            average_loss = 0
            for step in range(num_steps):
                batches, labels = self.batch_generator.next()
                feed_dict = dict()

                for i in range(num_unrollings):
                    feed_dict[self.train_inputs[i]] = batches[i]
                    feed_dict[self.train_labels[i]] = self.one_hot_encoding(labels[i + 1], self._vocabulary_size)

                _, _l, predictions, lr = sess.run(
                    [self.optimizer, self.loss, self.train_prediction, self.learning_rate], feed_dict=feed_dict)

                average_loss += _l
                if step % summary_frequency == 0:
                    if step > 0:
                        average_loss = average_loss / summary_frequency
                    print('Average loss at step %d: %f learning rate: %f' % (step, average_loss, lr))
                    average_loss = 0
                    labels = self.one_hot_encoding(np.concatenate(list(labels)[1:]), self._vocabulary_size)
                    print('Minibatch perplexity: %.2f' % float(
                        np.exp(self.logprob(predictions, labels))))

                if step % (summary_frequency * 10) == 0:
                    print('=' * 80)
                    for _ in range(5):
                        bigram_id = self.sample(self.random_uniform_distribution())
                        feed = self._embeddings[bigram_id]
                        sentence = self._vocabulary[bigram_id][0]
                        self._reset_sample_state.run()
                        for _ in range(79):
                            prediction = self._sample_prediction.eval({self._sample_input: feed})
                            bigram_id = self.sample(prediction)
                            feed = self._embeddings[bigram_id]
                            sentence += self._vocabulary[bigram_id][0]
                        print sentence
                    print('=' * 80)


    def evaluate(self):
        pass

    def build_graph(self):
        graph = tf.Graph()
        emb_dim, num_nodes, num_unrollings = self._emb_dim, self._num_nodes, self._num_unrollings
        batch_size, num_nodes, vocabulary_size = self._batch_size, self._num_nodes, self._vocabulary_size

        with graph.as_default():
            with tf.variable_scope("lstm"):
                with tf.variable_scope("cell"):
                    W = tf.get_variable("input_weights", [emb_dim, num_nodes * 4],
                                       initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                    U = tf.get_variable("previous_hidden_weights", [num_nodes, num_nodes * 4],
                                       initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                    b = tf.get_variable("biases",
                                       initializer=tf.zeros_initializer([1, num_nodes * 4]))

                previous_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
                previous_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

                with tf.variable_scope("classifier"):
                    weights = tf.get_variable("weights", [num_nodes, vocabulary_size],
                                             initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                    biases = tf.get_variable("biases",
                                            initializer=tf.zeros_initializer([vocabulary_size]))

            def lstm_cell(i, o, state):
                prods = tf.split(1, 4, tf.matmul(i, W) + tf.matmul(o, U) + b)
                input_gate = tf.sigmoid(prods[0])
                forget_gate = tf.sigmoid(prods[1])
                state = forget_gate * state + input_gate * tf.tanh(prods[2])
                output_gate = tf.sigmoid(prods[3])
                output = output_gate * tf.tanh(state)
                return output, state

            self.train_inputs = list()
            self.train_labels = list()

            for _ in range(num_unrollings):
                self.train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, emb_dim]))
                self.train_labels.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))

            outputs = list()
            o = previous_output
            state = previous_state

            for i in self.train_inputs:
                output, state = lstm_cell(i, o, state)
                outputs.append(output)

            with tf.control_dependencies([previous_output.assign(output),
                                         previous_state.assign(state)]):
                logits = tf.matmul(tf.concat(0, outputs), weights) + biases
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, self.train_labels)))

            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(10.0, global_step, 20000, 0.95, staircase=True)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients, v = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self.optimizer = self.optimizer.apply_gradients(
                zip(gradients, v), global_step=global_step)

            self.train_prediction = tf.nn.softmax(logits)

            with tf.variable_scope("sample"):
                self._sample_input = tf.placeholder(tf.float32, shape=[1, emb_dim])
                previous_sample_output = tf.get_variable("previous_output",
                                                        initializer=tf.zeros_initializer([1, num_nodes]))
                previous_sample_state = tf.get_variable("previous_state",
                                                       initializer=tf.zeros_initializer([1, num_nodes]))
                self._reset_sample_state = tf.group(
                    previous_sample_output.assign(tf.zeros([1, num_nodes])),
                    previous_sample_state.assign(tf.zeros([1, num_nodes])))

                sample_output, sample_state = lstm_cell(
                    self._sample_input, previous_sample_output, previous_sample_state)

                with tf.control_dependencies([previous_sample_output.assign(sample_output),
                                             previous_sample_output.assign(sample_state)]):
                    self._sample_prediction = tf.nn.softmax(tf.matmul(sample_output, weights) + biases)
        self._graph = graph

    def one_hot_encoding(self, ids, size):
        enc = np.zeros((len(ids), size))
        for i, _id in enumerate(ids):
            enc[i][_id] = 1
        return enc

    def random_uniform_distribution(self):
        b = np.random.uniform(0.0, 1.0, size=[1, self._vocabulary_size])
        return b / np.sum(b, 1)

    def sample(self, prediction):
        return np.random.choice(range(self._vocabulary_size), 1, p=prediction[0])

    def logprob(self, predictions, labels):
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, - np.log(predictions))) / labels.shape[0]

def main(_):
    if not FLAGS.train_data:
        print "--train_data must be specified."
        exit(-1)

    b2v = Bigrams2Vec(
        FLAGS.train_data,
        decompress=FLAGS.decompress)
    b2v.train()

    lstm = LSTM(b2v.embeddings, b2v.vocabulary, b2v.dictionary, b2v.text)
    del b2v
    lstm.train()
    lstm.evaluate()

if __name__ == "__main__":
    tf.app.run()
