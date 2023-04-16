#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        super(CharDecoder, self).__init__()

        self.vocab_size = len(target_vocab.char2id)

        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size)
        self.decoderCharEmb = nn.Embedding(self.vocab_size, char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

        ### END YOUR CODE



    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        X = self.decoderCharEmb(input)  # (length, batch, e_char)
        h, dec_hidden = self.charDecoder(X, dec_hidden)   # scores.shape: (length, batch, h)
        scores = self.char_output_projection(h)    # (length, batch, V_char)

        return  scores, dec_hidden

        ### END YOUR CODE


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        X_input = char_sequence[:-1]    # chop off the char vocab's <END> token
        scores, dec_hidden = self.forward(X_input, dec_hidden) # scores.shape: (length, batch, V_char)
        scores_reshaped = scores.view(-1, scores.shape[2])  # (length*batch, V_char)

        X_target = char_sequence[1:]    # chop off the char vocab's <START> token
        char_seq_reshaped = X_target.reshape(-1)  # (length*batch, )

        char_pad_token_id = self.target_vocab.char2id['<pad>']
        loss = F.cross_entropy(scores_reshaped, char_seq_reshaped, ignore_index=char_pad_token_id, reduction='sum')

        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        dec_hidden = initialStates
        batch_size = dec_hidden[0].shape[1]
        char_start_token_id = self.target_vocab.start_of_word
        current = torch.tensor([char_start_token_id] * batch_size, device=device).unsqueeze(0)   # (L=1, batch=batch_size)

        output_word = torch.empty(0, batch_size, device=device, dtype=torch.long)

        for _ in range(max_length + 1):
            scores, dec_hidden = self.forward(current, dec_hidden)  # scores: (L=1, b, V_char)
            assert scores.shape == (1, batch_size, self.vocab_size)

            current = torch.argmax(scores, dim=2)   # (1, b)
            output_word = torch.cat((output_word, current))  # after current loop: (max_length, b)

        decodedWords = []
        char_end_token_id = self.target_vocab.end_of_word

        for char_ids in output_word.transpose(0, 1).tolist():
            char_ids.append(char_end_token_id)  # b/c there can be a word with no <END> token even we went through until max_length + 1
            word = "".join(self.target_vocab.id2char[c_id] for c_id in char_ids[:char_ids.index(char_end_token_id)])
            decodedWords.append(word)

        return decodedWords

        return decodedWords

        ### END YOUR CODE

