import subprocess

#ABSOLUTE PATH TO FILENAME

def spellchecker(filename):
    checked = subprocess.run("cat " + filename + " | aspell -l en -a",
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
                info,suggestions = item.split(': ')
                parsed_checked.append((info.split(' ')[1],
                                       suggestions.split(', ')[0]))
    return parsed_checked

def dummy_spellchecker():
    with open('reported_errors.txt','r',encoding='utf-8') as f:
        checked = f.read().split('\n')
        parsed_checked = []
        for i,item in enumerate(checked):
            if not i:
                continue
            if item:
                if item == '*':
                    parsed_checked.append(None)
                else:
                    info,suggestions = item.split(': ')
                    parsed_checked.append((info.split(' ')[1],
                                       suggestions.split(', ')[0]))
    return parsed_checked        
