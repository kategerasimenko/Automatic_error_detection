import os
from collections import OrderedDict, defaultdict
import re


class RealecExtractor:
    def __init__(self,error_needed,errors_to_correct,path_to_corpus='../REALEC/'):
        self.error = error_needed
        self.errors_to_correct = errors_to_correct
        self.path = path_to_corpus
        self.path_new = './processed_texts/'
        self.current_doc_errors = OrderedDict()
        os.makedirs(self.path_new, exist_ok=True)

    def load_texts(self):
        for root,dirs,files in os.walk(self.path):
            pass

    # ========= COPYPASTE (with changes, though) FROM REALEC EXERCISES - START ============
    def find_errors_indoc(self, line):
        """
        Find all T... marks and save in dictionary.
        Format: {"T1":{'Error':err, 'Index':(index1, index2), "Wrong":text_mistake}}
        """
        if re.search('^T', line) is not None and 'pos_' not in line:
            try:
                t, span, text_mistake = line.strip().split('\t')
                err, index1, index2 = span.split()
                if err == self.error or err in self.errors_to_correct:
                    self.current_doc_errors[t] = {'Error':err, 'Index':(int(index1), int(index2)), "Wrong":text_mistake}
                    return (int(index1), int(index2))
            except:
                #print (''.join(traceback.format_exception(*sys.exc_info())))
                print("Errors: Something wrong! No notes or a double span", line)

    def validate_answers(self, answer):
        # TO DO: multiple variants?
        if answer.upper() == answer:
            answer = answer.lower()
        answer = answer.strip('\'"')
        answer = re.sub(' ?\(.*?\) ?','',answer)
        if '/' in answer:
            answer = answer.split('/')[0]
        if '\\' in answer:
            answer = answer.split('\\')[0]
        if ' OR ' in answer:
            answer = answer.split(' OR ')[0]
        if ' или ' in answer:
            answer = answer.split(' или ')[0]
        if answer.strip('? ') == '' or '???' in answer:
            return None
        return answer

    def find_answers_indoc(self, line):
        if re.search('^#', line) is not None and 'lemma =' not in line:
            try:
                number, annotation, correction = line.strip().split('\t')
                t_error = annotation.split()[1]
                if self.current_doc_errors.get(t_error):
                    validated = self.validate_answers(correction)
                    if validated is not None:
                        self.current_doc_errors[annotation.split()[1]]['Right'] = validated
            except:
                pass
                #print (''.join(traceback.format_exception(*sys.exc_info())))
                #print("Answers: Something wrong! No Notes probably", line)

    def find_delete_seqs(self, line):
        if re.search('^A', line) is not None and 'Delete' in line:
            t = line.strip().split('\t')[1].split()[1]
            if self.current_doc_errors.get(t):
                self.current_doc_errors[t]['Delete'] = 'True'

    def collect_errors_info(self):
        """ Collect errors info """
        print('collecting errors info...')
        anns = []
        for root, dire, files in os.walk(self.path):
            anns += [root+'/'+f for f in files if f.endswith('.ann')]
        i = 0
        for ann in anns:
            self.error_intersects = set()
            with open(ann, 'r', encoding='utf-8') as ann_file:
                if not ann_file.read().strip():
                    continue
            with open(ann, 'r', encoding='utf-8') as ann_file:
                for line in ann_file.readlines():
                    ind = self.find_errors_indoc(line)
                    self.find_answers_indoc(line)
                    self.find_delete_seqs(line)
                    
            new_errors = OrderedDict()
            for x in sorted(self.current_doc_errors.items(),key=lambda x: (x[1]['Index'][0],x[1]['Index'][1],int(x[0][1:]))):
                if 'Right' in x[1] or 'Delete' in x[1]:
                    new_errors[x[0]] = x[1]
            self.current_doc_errors = new_errors
            
            unique_error_ind = []
            error_ind = [self.current_doc_errors[x]['Index'] for x in self.current_doc_errors]
            for ind in error_ind:
                if ind in unique_error_ind:
                    self.error_intersects.add(ind)
                else:
                    unique_error_ind.append(ind)                
            self.embedded,self.overlap1,self.overlap2 = self.find_embeddings(unique_error_ind)
            self.make_one_file(ann[:ann.find('.ann')],str(i))
            i += 1
            self.current_doc_errors.clear()
        

    def find_embeddings(self,indices):
        indices.sort(key=lambda x: (x[0],-x[1]))
        embedded = []
        overlap1, overlap2 = [],[]
        self.embedding = defaultdict(list)
        for i in range(1,len(indices)):
            find_emb = [x for x in indices if (x[0] <= indices[i][0] and x[1] > indices[i][1]) or \
                                              (x[0] < indices[i][0] and x[1] >= indices[i][1])]
            if find_emb:
                for j in find_emb:
                    self.embedding[str(j)].append(indices[i])
                embedded.append(indices[i])
            else:
                overlaps = [x for x in indices if x[0] < indices[i][0] and (x[1] > indices[i][0] and
                                                                            x[1] < indices[i][1])]
                if overlaps:
                    overlap1.append(overlaps[0])
                    overlap2.append(indices[i])
        return embedded, overlap1, overlap2
        
    def tackle_embeddings(self,dic):
        b = dic.get('Index')[0]
        emb_errors = [x for x in self.current_doc_errors.items() if x[1]['Index'] in self.embedding[str(dic.get('Index'))]]
        new_wrong = ''
        nw = 0
        ignore = []
        for j,ws in enumerate(dic['Wrong']):
            emb_intersects = []
            for t,e in emb_errors:
                if e['Index'][0]-b == j:
                    if 'Right' in e and 'Right' in dic and e['Right'] == dic['Right']:
                        break
                    if str(e['Index']) in self.embedding:
                        ignore += self.embedding[str(e['Index'])]
                    if e['Index'] in self.error_intersects:
                        emb_intersects.append((int(t[1:]),e))
                        continue
                    if e['Index'] not in ignore:
                        if 'Right' in e:
                            new_wrong += e['Right']
                            nw = len(e['Wrong'])
                        elif 'Delete' in e:
                            nw = len(e['Wrong'])
            if emb_intersects:
                emb_intersects = sorted(emb_intersects,key=lambda x: x[0])
                last = emb_intersects[-1][1]
                L = -1
                while 'Right' not in last:
                    L -= 1
                    last = emb_intersects[L][1]
                new_wrong += last['Right']
                nw = len(last['Wrong'])
            if not nw:
                new_wrong += ws
            else:
                nw -= 1
        return new_wrong

    def find_overlap(self,s1,s2):
        m = difflib.SequenceMatcher(None, s1, s2).get_matching_blocks()
        if len(m) > 1:
            for x in m[:-1]:
                if x.b == 0:
                    return x.size
        return 0
            

    def make_one_file(self, filename, new_filename):
        """
        Makes file with current types of errors. all other errors checked.
        :param filename: name of the textfile
        return: nothing. just write files in dir <<processed_texts>>
        """
        with open(self.path_new+new_filename+'.txt', 'w', encoding='utf-8') as new_file:
            with open(filename+'.txt', 'r', encoding='utf-8', newline='') as text_file:
                one_text = text_file.read()
                not_to_write_sym = 0
                for i, sym in enumerate(one_text):
                    intersects = []
                    for t_key, dic in self.current_doc_errors.items():
                        if dic.get('Index')[0] == i:
                            if dic.get('Error') == 'Punctuation' and 'Right' in dic and \
                               not dic.get('Right').startswith(','):
                                dic['Right'] = ' '+dic['Right']
                            if dic.get('Index') in self.embedded:
                                continue
                            if str(dic.get('Index')) in self.embedding:
                                if dic.get('Error') == self.error:
                                    new_wrong = self.tackle_embeddings(dic)
                                    new_file.write('**'+str(dic.get('Right'))+'**'+str(len(new_wrong))+'**'+new_wrong)
                                    not_to_write_sym = len(dic['Wrong'])
                                    break

                            if dic.get('Index') in self.overlap1:
                                if dic.get('Error') != self.error:
                                    overlap2_ind = self.overlap2[self.overlap1.index(dic.get('Index'))]
                                    overlap2_err = [x for x in self.current_doc_errors.values() if x['Index'] == overlap2_ind][-1]
                                    if 'Right' in dic and 'Right' in overlap2_err:
                                        rn = self.find_overlap(dic['Right'],overlap2_err['Right'])
                                        wn = dic['Index'][1] - overlap2_err['Index'][0]
                                        indexes_comp = dic.get('Index')[1] - dic.get('Index')[0] - wn
                                        if rn == 0:
                                            new_file.write(str(dic.get('Right'))+'#'+str(indexes_comp)+'#'+str(dic.get('Wrong'))[:-wn])
                                        else:
                                            new_file.write(str(dic.get('Right')[:-rn])+'#'+str(indexes_comp)+'#'+str(dic.get('Wrong'))[:-wn])
                                        not_to_write_sym = len(str(dic.get('Wrong'))) - wn
                                        break

                            if dic.get('Index') in self.overlap2:
                                overlap1_ind = self.overlap1[self.overlap2.index(dic.get('Index'))]
                                overlap1_err = [x for x in self.current_doc_errors.values() if x['Index'] == overlap1_ind][-1]
                                if overlap1_err['Error'] == self.error:
                                    if dic.get('Error') != self.error:
                                        if 'Right' in dic and 'Right' in overlap1_err:
                                            rn = self.find_overlap(overlap1_err['Right'],dic['Right'])
                                            wn = overlap1_err['Index'][1] - dic['Index'][0]
                                            indexes_comp = dic.get('Index')[1] - dic.get('Index')[0] - wn
                                            new_file.write(dic.get('Wrong')[:wn] + dic.get('Right')[rn:] +'#'+str(indexes_comp)+ '#'+dic.get('Wrong')[wn:])
                                            not_to_write_sym = len(str(dic.get('Wrong')))
                                    break
                                    
                                    
                            if dic.get('Index') in self.error_intersects:
                                intersects.append((int(t_key[1:]),dic))
                                continue
    
                            if dic.get('Right'):
                                indexes_comp = dic.get('Index')[1] - dic.get('Index')[0]
                                if dic.get('Error') == self.error:
                                    new_file.write('**'+str(dic.get('Right'))+'**'+str(indexes_comp)+'**')
                                else:
                                    new_file.write(dic.get('Right') +
                                                   '#'+str(indexes_comp)+ '#')
                            else:
                                if dic.get('Delete'):
                                    indexes_comp = dic.get('Index')[1] - dic.get('Index')[0]
                                    if dic.get('Error') == self.error:
                                        new_file.write("**DELETE**"+str(indexes_comp)+"**")
                                    else:
                                        new_file.write("#DELETE#"+str(indexes_comp)+"#")
                                    
                    if intersects:
                        intersects = sorted(intersects,key=lambda x: x[0])
                        intersects = [x[1] for x in intersects]
                        needed_error_types = [x for x in intersects if x['Error'] == self.error]
                        if needed_error_types and 'Right' in needed_error_types[-1]:
                            saving = needed_error_types[-1]
                            intersects.remove(saving)
                            if intersects:
                                to_change = intersects[-1]
                                if 'Right' not in to_change or to_change['Right'] == saving['Right']:
                                    indexes_comp = saving['Index'][1] - saving['Index'][0]
                                    new_file.write('**'+str(saving['Right'])+'**'+str(indexes_comp)+'**')
                                else: 
                                    indexes_comp = len(to_change['Right'])
                                    not_to_write_sym = saving['Index'][1] - saving['Index'][0]
                                    new_file.write('**'+str(saving['Right'])+'**'+str(indexes_comp)+'**'+to_change['Right'])
                        else:
                            if 'Right' in intersects[-1]:
                                if len(intersects) > 1 and 'Right' in intersects[-2]:
                                    indexes_comp = len(intersects[-2]['Right'])
                                    not_to_write_sym = intersects[-1]['Index'][1] - intersects[-1]['Index'][0]
                                    new_file.write(intersects[-1]['Right'] + '#'+str(indexes_comp)+ '#' + intersects[-2]['Right'])
                                else:
                                    indexes_comp = intersects[-1]['Index'][1] - intersects[-1]['Index'][0]
                                    new_file.write(intersects[-1]['Right'] + '#'+str(indexes_comp)+ '#')
                    if not not_to_write_sym:
                        new_file.write(sym)
                    else:
                        not_to_write_sym -= 1
                        

    # ========= COPYPASTE FROM REALEC EXERCISES - END ============

    def text_generator(self):
        #self.collect_errors_info()
        for f in os.listdir(self.path_new):
            error_intervals = []
            new_text = ''
            with open(self.path_new + f,'r', encoding='utf-8', newline='') as one_doc:
                text = one_doc.read()
                if '#' in text:
                    text_array = text.split('#')
                    current_number = 0
                    for words in text_array:
                        words = words.replace('\n', ' ').replace('\ufeff', '')
                        if re.match('^[0-9]+$', words):
                            if words != '':
                                current_number = int(words)
                        elif words == 'DELETE':
                            current_number = 0
                        else:
                            new_text += words[current_number:]
                            current_number = 0
                    if '**' in new_text:
                        text_spl = new_text.split('**')
                        new_new_text = ''
                        for i in range(0,len(text_spl),3):
                            if len(text_spl[i:i+4]) > 1:
                                text_before, right_answer, index, text_after = text_spl[i],text_spl[i+1],text_spl[i+2],text_spl[i+3]
                                if not new_new_text:
                                    new_new_text = text_before
                                st_index = len(new_new_text)
                                new_new_text += text_after
                                end_index = st_index + int(index)
                                error_intervals.append((st_index,end_index,right_answer))               
                        yield new_new_text,error_intervals

if __name__ == '__main__':
    a = RealecExtractor('Articles',{'Spelling'},path_to_corpus='../REALEC/exam/exam2017')
    for t,e in a.text_generator():
        pass
        #print(t,e)
        #print('================================')
