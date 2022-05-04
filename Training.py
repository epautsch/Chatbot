import os
import random

import torch
from torch import nn

from DataPreparation import batch2TrainData

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# Keep these token names
padding = 0
start_sentence = 1
end_sentence = 2


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)

    return loss, nTotal.item()


def train_actual(input_var, lengths, target_var, mask, max_target_len, encoder, decoder, embedding,
                 encoder_optimizer, decoder_optimizer, batch_size, clip, max_length, teacher_forcing_ratio):
    # zero gradients.. check documentation!
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_var = input_var.to(device)
    target_var = target_var.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu") # needs to be on cpu, let's read why
    loss = 0
    loss_list = []
    n_totals = 0
    e_outputs, e_hidden = encoder(input_var, lengths)
    d_input = torch.LongTensor([[start_sentence for _ in range(batch_size)]])
    d_input = d_input.to(device)
    d_hidden = e_hidden[:decoder.n_layers]

    # check for teacher forcing
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, d_hidden = decoder(
                d_input, d_hidden, e_outputs
            )
            d_input = target_var[t].view(1, -1)
            # loss accumulator. repeat below too
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_var[t], mask[t])
            loss += mask_loss
            loss_list.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, d_hidden = decoder(
                d_input, d_hidden, e_outputs
            )
            _, topi = decoder_output.topk(1)
            d_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            d_input = d_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_var[t], mask[t])
            loss += mask_loss
            loss_list.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # backprop call here
    loss.backward()
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    _sum = sum(loss_list)

    return _sum / n_totals

# make sure all parameters are given in Chatbot.py when calling
def trainAllIterations(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
                       decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers,
                       save_dir, n_iteration, batch_size, hidden_size, print_every, save_every, clip,
                       corpus_name, loadFilename, max_length, teacher_forcing_ratio, checkpoint=None):

    # make batch list to iterate through
    # use list comprehension like below
    t_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    iter_start = 1
    print_loss = 0

    if loadFilename:
        iter_start = checkpoint['iteration'] + 1

    # training loop's start
    for iteration in range(iter_start, n_iteration + 1):
        training_batch = t_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train_actual(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                            decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip,
                            max_length, teacher_forcing_ratio)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iter: {}; percentage finished: {:.1f}%; Avg loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # save point, just in case, and to separate training time
        # check that ordering doesn't matter
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers,
                                                                                          decoder_n_layers,
                                                                                          hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))