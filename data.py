#!/usr/bin/env python

import numpy as np

default_vocab = '\x00abcdefghijklmnopqrstuvwxyz'

def arg_close_not_under(x, reference_values):
    ''' Return the index in reference_values that is closest
    to x without going under x'''
    return np.argmin(np.where(x < reference_values, reference_values, np.inf) - x).astype(np.uint32)


def data_pair(prompts=('hello', 'goodbye', 'talk to me'),
              probs=(0.6, 0.3, 0.1),
              responses=('patrick', 'little kitty', ''),
              noise=0.3):

    cumulative_probs = np.cumsum(probs)

    # choose prompt
    random_number = np.random.random()
    prompt_idx = arg_close_not_under(random_number, cumulative_probs)
    prompt = prompts[prompt_idx]

    # choose response
    if np.random.random() > noise:
        response = responses[prompt_idx]
    else:
        response = np.random.choice(np.concatenate((responses[:prompt_idx], responses[(prompt_idx + 1):])))

    return prompt, response

def string_to_onehot(input, vocab, add_end_token=True):
    vocab = list(vocab)
    if add_end_token:
        input = input + '\x00'
    indices, vocables = zip(*enumerate(vocab))
    reverse_vocab = dict(zip(vocables, indices))

    input_indices = [reverse_vocab[word] for word in input
                     if word in reverse_vocab]

    input_onehot = np.zeros((len(vocab), len(input)),dtype=np.uint8)
    for position, index in enumerate(input_indices):
        input_onehot[index, position] = 1
    return input_onehot


if __name__ == "__main__":
    for i in range(50):
        print "{} {}".format(*data_pair())
