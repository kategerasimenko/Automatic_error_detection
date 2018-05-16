import subprocess
import numpy as np
import sys, os
import json
sys.path.insert(0, os.path.abspath('..'))
from feature_extraction.lm_probas import get_lm_probas

#ABSOLUTE PATH TO FILENAME

def spellchecker(sentence):
    checked = subprocess.run("echo " + filename + " | aspell -l en -a",
                             shell=True, check=True,encoding='utf-8') # not tested
    checked = checked.split('\n')
    parsed_checked = []
    for i,item in enumerate(checked):
        if not i:
            continue
        if item:
            if item == '*':
                parsed_checked.append(None)
            else:
                info,str_suggestions = item.split(': ')
                initial = info.split(' ')[1]
                suggestions = [initial] + str_suggestions.split(', ')[:10]
                final_suggestion = suggestions[lm_decision(sentence,initial,suggestions,
                                               int(info.split(' ')[-1]))]
                parsed_checked.append((initial,final_suggestion))
    return parsed_checked

def dummy_spellchecker():
    with open('reported_errors_sents.txt','r',encoding='utf-8') as f:
        checked_sents = f.read().split('\n\n')
    with open('initial_sents.txt','r',encoding='utf-8') as f:
        sents = f.read().split('\n')
    parsed_checked = []
    first = True
    for item_group,sent in zip(checked_sents,sents):
        for item in item_group.split('\n'):
            if first:
                first = False
                continue
            if item:
                if item == '*':
                    parsed_checked.append(None)
                else:
                    info,str_suggestions = item.split(': ')
                    initial = info.split(' ')[1]
                    suggestions = str_suggestions.split(', ')[:10]
                    final_suggestion = suggestions[lm_decision(sent,initial,suggestions,
                                                   int(info.split(' ')[-1]))]
                    parsed_checked.append((initial,final_suggestion))
    return parsed_checked       

    
def lm_decision(sent,initial,suggestions,idx):
    options = []
    for s in suggestions:
        options.append(sent[:idx]+s+sent[idx+len(initial):])
    probs = get_lm_probas('\n'.join(options)+'\n','text')
    return np.argmax(probs)
    
    
def very_dummy_spellchecker():
    with open('spellcheck_res.json','r',encoding='utf-8') as f:
        parsed_checked = json.loads(f.read())
    return parsed_checked
    
    
if __name__ == '__main__':
    with open('spellcheck_res.json','w',encoding='utf-8') as f:
        f.write(json.dumps(dummy_spellchecker(),ensure_ascii=False))
    