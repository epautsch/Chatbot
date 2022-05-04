import itertools

import torch
padding = 0  # Used for padding short sentences
start_sentence = 1  # Start-of-sentence token
end_sentence = 2  # End-of-sentence token


def indexFromSentences(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [end_sentence]


def zeroPad(l, fillvalue=padding):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryArray(l, value=padding):
    m = []

    for i, seq in enumerate(l):
        m.append([])

        for token in seq:
            if token == padding:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


def inputVar(l, voc):
    indexes_batch = [indexFromSentences(voc, sentence) for sentence in l]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    padList = zeroPad(indexes_batch)
    padVar = torch.LongTensor(padList)

    return padVar, lengths


def outputVar(l, voc):
    indexes_batch = [indexFromSentences(voc, sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])

    padList = zeroPad(indexes_batch)
    mask = binaryArray(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)

    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)

    return inp, lengths, output, mask, max_target_len


def trimRareWords(voc, pairs, min_count):
    voc.trim(min_count)
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    return keep_pairs