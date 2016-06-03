#!/usr/bin/env python

import numpy as np

def arg_close_not_under(x, reference_values):
    ''' Return the index in reference_values that is closest
    to x without going under x'''
    return np.argmin(np.where(x < reference_values, reference_values, np.inf) - x).astype(np.uint32)

def data_pair(prompts=['hello','goodbye','talk to me'],probs=[0.6, 0.3, 0.1], responses=['Patrick','cruel world',''], noise=0.3):
    # unpack prompts and responses
    #prompts, probs_responses = zip(*prompts_probs.items())
    #probs, responses = zip(*probs_responses)
    cumulative_probs = np.cumsum(probs)
    
    # choose prompt
    random_number = np.random.random()
    prompt_idx = arg_close_not_under(random_number, cumulative_probs)      
    prompt = prompts[prompt_idx]

    # choose response
    if np.random.random() > noise:
        response = responses[prompt_idx]
    else:
        response=np.random.choice(np.concatenate((responses[:prompt_idx], responses[(prompt_idx+1):])))
    
    return prompt, response

if __name__=="__main__":
    for i in range(50):
         print "{} {}".format(*data_pair())

