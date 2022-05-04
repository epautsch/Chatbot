import re

import unicodedata

padding = 0  # Used for padding short sentences
start_sentence = 1  # Start-of-sentence token
end_token = 2  # End-of-sentence token


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {padding: "PAD", start_sentence: "SOS", end_token: "EOS"}
        self.num_words = 3

    def append_full_sentence(self, sentence):
        for word in sentence.split(' '):
            self.append_word(word)

    def append_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return

        self.trimmed = True
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # dictionaries must be reset here
        # that's what was causing error earlier
        self.word2index = {}
        self.word2count = {}
        self.index2word = {padding: "PAD", start_sentence: "SOS", end_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.append_word(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# check textbook chapter on RE
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()

    return s


def readVocs(datafile, corpus_name):
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    voc = Vocabulary(corpus_name)

    return voc, pairs


# checks for under max_length
def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


# this creates Voc obj
def loadPrepareData(corpus, corpus_name, datafile, save_dir, max_length):
    voc, pairs = readVocs(datafile, corpus_name)
    # Keep print statements for now
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")

    for pair in pairs:
        voc.append_full_sentence(pair[0])
        voc.append_full_sentence(pair[1])

    print("Counted words:", voc.num_words)

    return voc, pairs