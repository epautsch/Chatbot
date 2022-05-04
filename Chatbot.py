from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import optim
import csv
import random
import os
import codecs
from io import open

from DataExtraction import *
from DataPreparation import *
from FilePreparation import *
from Evaluation import GSD, evalInput
from Seq2Seq import *
from Training import trainAllIterations

# setup
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

datafile = os.path.join(corpus, "formatted_movie_lines.txt")
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initial data structures
lines_dict = {}
convos = []
line_fields = ["lineID", "characterID", "movieID", "character", "text"]
convo_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# process convos
lines_dict = load_by_line(os.path.join(corpus, "movie_lines.txt"), line_fields)
convos = organizeConvos(os.path.join(corpus, "movie_conversations.txt"),
                        lines_dict, convo_fields)
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in makePairs(convos):
        writer.writerow(pair)

padding = 0  # short sentence padding
start_sentence = 1  # sentence start
end_sentence = 2  # sentence end
maximum_sentence_length = 10  # max length for sentence consideration
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir, maximum_sentence_length)
minimum_count = 3
pairs = trimRareWords(voc, pairs, minimum_count)
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

# model configuration
model_name = 'cb_model'
# uncomment the specific attention model desired for training
#attention_model = 'dot'
#attention_model = 'general'
attention_model = 'concat'
HIDDEN_SIZE = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# load checkpoint. Uncomment loadFilename (line 72) to load in trained model.
loadFilename = None
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, HIDDEN_SIZE),
                           '{}_checkpoint.tar'.format(checkpoint_iter))
if loadFilename:
    # !IMPORTANT!
    # If loading on same machine the model was trained on
    checkPoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkPoint['en']
    decoder_sd = checkPoint['de']
    encoder_optimizer_sd = checkPoint['en_opt']
    decoder_optimizer_sd = checkPoint['de_opt']
    embedding_sd = checkPoint['embedding']
    voc.__dict__ = checkPoint['voc_dict']

embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
encoder = EncoderRNN(HIDDEN_SIZE, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attention_model, embedding, HIDDEN_SIZE, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Master, models built and ready to go!')

clip = 50.0
TEACHER_FORCING_RATIO = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# need to be in train mode
encoder.train()
decoder.train()

# Optimizers is available
print('Building optimizers, master ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# cuda config
# for state in encoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
# for state in decoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()

print("Erik's computer! Starting Training! Ikimas!")
trainAllIterations(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   HIDDEN_SIZE, print_every, save_every, clip, corpus_name, loadFilename,
                   maximum_sentence_length, TEACHER_FORCING_RATIO, checkPoint) # <-- add 'checkPoint' if loading in
encoder.eval()
decoder.eval()
searcher = GSD(encoder, decoder)
evalInput(encoder, decoder, searcher, voc, maximum_sentence_length)
















