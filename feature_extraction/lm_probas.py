import subprocess
import json
import os

def get_lm_probas(inp,inp_type):
    # text is a string. ' ' between tokens, \n between sents
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
    preds = [float(x.split('\t')[-1].split(' ')[1]) if len(x.split('\t')) > 2 else None
             for x in output.split('\n')][:-5]
    if inp_type == 'text':
        os.remove(file)
    return preds

if __name__ == '__main__':
    with open('lm_preds.json','w',encoding='utf-8') as f:
        f.write(json.dumps(get_lm_probas('sents.txt',inp_type='file')))
