import subprocess
import numpy as np
from lm_probas import get_lm_probas
from nltk.metrics.distance import edit_distance

with open('../irregular_verbs.json','r',encoding='utf-8') as f:
    irregular_verbs = json.loads(f.read())

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
                suggestions = str_suggestions.split(', ')[:10]
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
                    repl = edit_distance(initial,suggestions[0],substitution_cost=0.5)
                    if repl == 0.5 and initial.lower() != suggestions[0].lower():
                        print(initial,suggestions[0])
                        parsed_checked.append((initial,suggestions[0]))
                    else:
                        final_suggestion = suggestions[lm_decision(sent,initial,suggestions,
                                                       int(info.split(' ')[-1]))]
                        parsed_checked.append((initial,final_suggestion))
    return parsed_checked        

def check_overreg_verb(sent,item,idx):
    if item[:-2] in irregular_verbs:
        forms = irregular_verbs[item[:-2]]
    elif item[:-1] in irregular_verbs:
        forms = irregular_verbs[item[:-1]]
    elif len(item) > 4 and item[-3] == item[-4] and \
         item[:-3] in irregular_verbs:
        forms = irregular_verbs[item[:-3]]
    elif len(item) > 3 and item[-3] == 'i' and \
         item[:-3]+'y' in irregular_verbs:
        forms = irregular_verbs[item[:-3]+'y']
    else:
        return None
    if sent[idx-5:idx-1] == 'have' or sent[idx-4:idx-1] == 'has' or \
        sent[idx-4:idx-1] == 'had':
        corr = forms[1]
    else:
        corr = forms[0]
    return corr
    
def lm_decision(sent,initial,suggestions,idx):
    options = []
    for s in suggestions:
        options.append(sent[:idx]+s+sent[idx+len(initial):])
    probs = get_lm_probas('\n'.join(options)+'\n','text')
    return np.argmax(probs)
    
if __name__ == '__main__':
    import json
    with open('spellcheck_res.json','w',encoding='utf-8') as f:
        f.write(json.dumps(dummy_spellchecker(),ensure_ascii=False))
    