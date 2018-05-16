import subprocess
import json
import os

def get_lm_probas(inp,inp_type):
    # text is a string. ' ' between tokens, \n between sents? \n\n between sent groups
    # inp_type - 'text' or 'file'
    # the \n ending should be added before this func
    if inp_type == 'text':
        with open('tmp_for_lm.txt','w',encoding='utf-8') as f:
            f.write(inp)
        file = 'tmp_for_lm.txt'
    else:
        file = inp
    f = open(file,'r',encoding='utf-8')
    command = ['/home/egerasimenko/kenlm/build/bin/query',
               '/home/egerasimenko/kenlm/BNC.binary']
    output = subprocess.check_output(command, stdin=f, universal_newlines=True)
    scores = []
    curr_scores = []
    for sent in output.split('\n')[:-5]:
        split_sent = sent.split('\t')
        if len(split_sent) <= 2:
            scores.append(curr_scores)
            curr_scores = []
        else:
            curr_scores.append(float(split_sent[-1].split(' ')[1]))
    scores.append(curr_scores)
    if inp_type == 'text':
        os.remove(file)
    return scores

if __name__ == '__main__':
    with open('lm_preds.json','w',encoding='utf-8') as f:
        f.write(json.dumps(get_lm_probas('sents.txt',inp_type='file')))
