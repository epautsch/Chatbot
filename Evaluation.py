import torch
from torch import nn

from DataExtraction import normalizeString
from DataPreparation import indexFromSentences

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
start_sentence = 1

# Greedy Search Decoder
# uses inheritance from torch
class GSD(nn.Module):
    def __init__(self, encoder, decoder):
        super(GSD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        e_outputs, e_hidden = self.encoder(input_seq, input_length)
        d_hidden = e_hidden[:self.decoder.n_layers]
        d_input = torch.ones(1, 1, device=device, dtype=torch.long) * start_sentence
        tokens = torch.zeros([0], device=device, dtype=torch.long)
        scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            d_output, d_hidden = self.decoder(d_input, d_hidden, e_outputs)
            d_scores, d_input = torch.max(d_output, dim=1)
            tokens = torch.cat((tokens, d_input), dim=0)
            scores = torch.cat((scores, d_scores), dim=0)
            d_input = torch.unsqueeze(d_input, 0)
        return tokens, scores


def eval(encoder, decoder, searcher, voc, sentence, max_length):
    indexes_batch = [indexFromSentences(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # cpu or gpu
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded = [voc.index2word[token.item()] for token in tokens]
    return decoded


def evalInput(encoder, decoder, searcher, voc, max_length):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            # Exit case
            if input_sentence == 'exit': break
            input_sentence = normalizeString(input_sentence)
            output_words = eval(encoder, decoder, searcher, voc, input_sentence, max_length)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Jeff:', ' '.join(output_words))

        except KeyError:
            print("unknown word")