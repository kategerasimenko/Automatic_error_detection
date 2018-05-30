# text processing
# create tables, get preds for each table, form ann file
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.data import load
from nltk import StanfordPOSTagger
from intervaltree import Interval, IntervalTree
from .spellchecker import spellchecker, dummy_spellchecker, very_dummy_spellchecker
from .idx_linked_list import IdxList
from gensim.models import KeyedVectors
from .articles import ArticleCorrector
from .prepositions import PrepositionCorrector
from .agreement import check_agreement
from string import punctuation
import re



class AnnotationCompiler:
    def __init__(self,filename):
        self.filename = filename
        self.tokenizer = TreebankWordTokenizer()
        self.sent_tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))
        self.st = StanfordPOSTagger('../stanfordPOStagger/english-bidirectional-distsim.tagger',
                                    '../stanfordPOStagger/stanford-postagger.jar',
                                    java_options='-mx2048m')
        #self.w2v_model = KeyedVectors.load_word2vec_format(
        #    "C:/Users/PC1/Desktop/python/деплом/deplom/constructions/GoogleNews-vectors-negative300.bin.gz",
        #    binary=True)
        self.w2v_model = None
        self.text = self.get_text()
        self.anns = []
        self.idx_list = IdxList()
        self.punct = punctuation + '‘’— \t\n'
        

    def get_text(self):
        with open(self.filename, encoding='utf-8-sig',newline='') as f:
            text = f.read()
        return text

    def correct_text(self,corrs,abs_idx=False):
        # given corrections (start idx,initial text, correction), correct the text
        slices = []
        last_idx = 0
        for item in corrs:
            if abs_idx:
                idx,initial,corr = item
                sent_idx = 0
            else:
                idx,sent_idx,initial,corr = item
            slices.append(self.text[last_idx:idx+sent_idx])
            slices.append(corr)
            last_idx = sent_idx + idx + len(initial)
        slices.append(self.text[last_idx:])
        return ''.join(slices)
        

    def ann_from_spelling(self,corrs):
        # create annotations and correct text from aspell output
        matches = [(m.group(0), m.start()) for m in re.finditer(r'[^\s\-]+', self.text)]
        matches = [x for x in matches if re.search('[0-9\\W]+',x[0]) is None or re.search('[0-9\\W]+',x[0]).group() != x[0]]
        tokens, idx = zip(*matches)
        final_corrs = []
        anns = []
        for i,corr in enumerate(corrs):
            if corr is not None:
                tag = 'Spelling'
                start_idx = idx[i]
                end_idx = start_idx + len(corr[0])
                self.idx_list.add(end_idx,len(corr[0])-len(corr[1]))
                anns.append(('%s %d %d\t%s' % (tag,start_idx,end_idx,corr[0]),
                             'AnnotatorNotes <ERROR>\t%s' % (corr[1])))
                final_corrs.append((start_idx,corr[0],corr[1]))
        self.text = self.correct_text(final_corrs,abs_idx=True)  # SHOULD BE SELF.TEXT WHEN IDXS ARE TACKLED
        return anns


    def ann_from_correction(self,corrs,tag):
        # start idx, sent start idx, initial np, predicted np
        anns = []
        for corr in corrs:
            start_idx = corr[0] + corr[1]
            end_idx = start_idx + len(corr[2])
            anns.append(('%s %d %d\t%s' % (tag,self.idx_list.find_old_idx(start_idx),
                                           self.idx_list.find_old_idx(end_idx),corr[2]),
                         'AnnotatorNotes <ERROR>\t%s' % (corr[3])))
        self.text = self.correct_text(corrs)
        return anns    


    def tokenize(self):
        sents = self.sent_tokenizer.tokenize(self.text)
        sent_spans = self.sent_tokenizer.span_tokenize(self.text)
        tokens = [self.tokenizer.tokenize(sent) for sent in sents]
        idxs = [align_tokens(['"' if x in ['``',"''"] else x for x in toks],sent)
                for sent,toks in zip(sents,tokens)]
        return sents,tokens,idxs,sent_spans
        

    def compile_annotation(self,path='.'):
        # collect all corrections
        sents,tokens,idxs,sent_spans = self.tokenize()
        with open(path+'/initial_sents.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(sents))
        spelling = very_dummy_spellchecker(path)
        print('Spelling')
        spell_anns = self.ann_from_spelling(spelling)
        self.anns.extend(spell_anns)
        #print([self.text])
        #print(self.idx_list)
        with open(path+'/corrected_spelling.txt','w',encoding='utf-8',newline='') as f:
            f.write(self.text)
        print('Tokenizing')
        sents,tokens,idxs,sent_spans = self.tokenize()
        #with open('init_sents_for_prepositions_test_parsed.txt','r',encoding='utf-8') as f:
        #    trees = parse_tree(f.read())
##        #agr_corrs = check_agreement(trees,sent_spans)
        print('Tagging')
        tsents = self.st.tag_sents(tokens)
        print('Prepositions')
        prep_corrector = PrepositionCorrector()
        prep_corrs,prep_idxs = prep_corrector.detect_errors(self.w2v_model,tokens,tsents,idxs,sents,sent_spans)
        prep_anns = self.ann_from_correction(prep_corrs,'Prepositions')
        for idx in prep_idxs:
            self.idx_list.add(idx[0],idx[1])
##        print(self.idx_list)
        with open(path+'/corrected_prepositions.txt','w',encoding='utf-8') as f:
            f.write(self.text)
        self.anns.extend(prep_anns)
        print('Articles')
        sents,tokens,idxs,sent_spans = self.tokenize()
        tsents = self.st.tag_sents(tokens)
        art_corrector = ArticleCorrector()
        art_corrs,art_idxs = art_corrector.detect_errors(self.w2v_model,tokens,tsents,idxs,sents,sent_spans)
        art_anns = self.ann_from_correction(art_corrs,'Articles')
        for idx in art_idxs:
            self.idx_list.add(idx[0],idx[1])
        with open(path+'/corrected_articles.txt','w',encoding='utf-8') as f:
            f.write(self.text)
        self.anns.extend(art_anns)
        print('Writing annotation')
        self.write_annotation()

    def write_annotation(self):
        with open(self.filename[:-4]+'.ann','w',encoding='utf-8') as f:
            for i,ann in enumerate(self.anns):
                f.write('T%d\t'%(i+1)+ann[0]+'\n'+\
                        '#%d\t'%(i+1)+ann[1].replace('<ERROR>','T%d'%(i+1))+'\n')
        
# exam 2014 EEm_33_2
# exam 2017 EGe_100_2
#a = AnnotationCompiler('test_text.txt')
#a.compile_annotation()
        
