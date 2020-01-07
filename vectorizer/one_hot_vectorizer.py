from utils.io_utils import IOUtils
from utils.string_utils import StringUtils
from multiprocessing.pool import Pool
import os


class OneHotVectoriser:

    def __init__(self):
        self.voc = dict()
        self.input_dir = ''
        self.vocab_path = ''
        self.seq_dir = ''
        self.next_dir = ''
        self.out_dir = ''
        self.seq_length = 4
        self.sequences_step = 1

    def clean_files_for_full_stops(self):
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        p = Pool(5)
        for in_file in in_files:
            in_json = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            clean_data = list(p.map(StringUtils.clean_full_stops, in_json))
            IOUtils.write_json_data_to_file("{}/{}".format(self.input_dir, in_file), clean_data)

    def remove_duplicates(self):
        import os
        duplicate_map = dict()
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        for in_file in in_files:
            in_json = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            clean_data = list(filter(lambda x: StringUtils.is_not_duplicate(x, duplicate_map), in_json))
            IOUtils.write_json_data_to_file("{}/{}".format(self.input_dir, in_file), clean_data)

    def clean_data_to_remove_smaller_ones(self):
        import json
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        for in_file in in_files:
            in_json_list = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            out_json = []
            for in_json in in_json_list:
                k = list(map(lambda x: x.strip().lower(), in_json.split(" ")))
                if len(k) > 10:
                    list(map(self.add_to_dict, k))
                    out_json.append(" ".join(k))

            IOUtils.write_json_data_to_file("{}/{}".format(self.input_dir, in_file), out_json, indent=2)
        with open("{}/{}".format(self.out_dir, "voc.json"), "w+") as f:
            json.dump(self.voc, f)

    def clean_data_to_remove_un_frequent_words(self, median):
        import json
        with open("{}/{}".format(self.out_dir, "voc.json"), "r") as f:
            voc_local = json.load(f)
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        for in_file in in_files:
            in_json_list = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            out_json = []
            for in_json in in_json_list:
                k = list(map(lambda x: x.strip().lower(), in_json.split(" ")))
                fi = list(filter(lambda y: not (len(y) == 1 and y != "a" and y != "i" and y != "*"), k))
                fi = list(map(lambda y: y if voc_local[y] >= median else "#ner", fi))
                out_json.append(" ".join(fi))
            IOUtils.write_json_data_to_file("{}/{}".format(self.input_dir, in_file), out_json, indent=2)

    def remove_multiple_ners(self):
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        for in_file in in_files:
            in_json = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            clean_data = list(filter(lambda x: x.count("#ner") <= 3 and len(x) > 50, in_json))
            clean_data = list(map(StringUtils.work_to_clean_up_ners, clean_data))
            IOUtils.write_json_data_to_file("{}/{}".format(self.input_dir, in_file), clean_data)

    def create_sequences(self, word_list):
        sequences = []
        next_words = []
        for i in range(0, len(word_list) - self.seq_length, self.sequences_step):
            sequences.append(word_list[i: i + self.seq_length])
            next_words.append(word_list[i + self.seq_length])
        return sequences, next_words

    def create_sentence_corpus(self):
        total_seq = []
        total_next = []
        input_count = 1
        import random
        in_files = IOUtils.get_all_file_names_in_dir(self.input_dir, 'json')
        for in_file in in_files:
            in_json = IOUtils.get_json_from_json_file(os.path.join(self.input_dir, in_file))
            sentence_list = list(map(StringUtils.get_word_list, in_json))
            random.shuffle(sentence_list)
            sequences_list = list()
            next_words_list = list()
            input_count += len(sentence_list)

            for word_list in sentence_list:
                sequences, next_words = self.create_sequences(word_list)
                sequences_list += sequences
                next_words_list += next_words

            assert len(sequences_list) == len(next_words_list)
            IOUtils.write_json_data_to_file("{}/{}".format(self.seq_dir, in_file), sequences_list, indent=2)
            IOUtils.write_json_data_to_file("{}/{}".format(self.next_dir, in_file), next_words_list, indent=2)
            total_next += next_words_list
            total_seq += sequences_list
        print("Total input count is: {}".format(input_count))
        return total_seq, total_next

    def create_vocabulary(self, word_list):
        import collections
        # count the number of words
        word_counts = collections.Counter(word_list)

        # Mapping from index to word : that's the vocabulary
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))

        # Mapping from word to index
        vocab = {x: i for i, x in enumerate(vocabulary_inv)}
        words = [x[0] for x in word_counts.most_common()]

        # size of the vocabulary
        vocab_size = len(words)
        print("vocab size: ", vocab_size)

        from six.moves import cPickle
        # save the words and vocabulary
        with open(os.path.join(self.vocab_path), 'wb') as f:
            cPickle.dump((words, vocab, vocabulary_inv), f)

        return vocab, vocab_size

    def create_vectors(self, vocab, vocab_size, sequences, next_words, seq_length=4):
        import numpy as np
        import pickle
        X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
        y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
        for i, sentence in enumerate(sequences):
            for t, word in enumerate(sentence):
                X[i, t, vocab[word]] = 1
            y[i, vocab[next_words[i]]] = 1
        pickle.dump(X, self.out_dir + "X.pk")
        pickle.dump(y, self.out_dir + "y.pk")
        return X, y

    def add_to_dict(self, wd):
        if wd in self.voc:
            self.voc[wd] += 1
        else:
            self.voc[wd] = 1

    def median(self):
        import json
        import statistics
        with open("{}/{}".format(self.input_dir, "voc.json"), "r") as f:
            d = json.load(f)
        ls = d.values()
        ls = list(filter(lambda x: x > 9, ls))
        print(statistics.median(ls))
