import subprocess
from time import time

with open('../../kenlm/BNC_plain.txt','r',encoding='utf-8-sig') as f:
  sents = f.read().split('\n')
  
open('BNC_tagged.txt','w',encoding='utf-8').close()

lim = len(sents)
print(lim)

step = 10000
start = time()
for j in range(step,lim+1,step):
  print(j)
  with open('tmp.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(sents[j-step:j]))
  parsed = subprocess.run('java -cp "../../inspector/stanfordPOStagger/stanford-postagger.jar" edu.stanford.nlp.tagger.maxent.MaxentTagger -model ../../inspector/stanfordPOStagger/english-bidirectional-distsim.tagger -textFile tmp.txt -nthreads 4 -sentenceDelimiter newline -tokenize false',shell=True, stdout=subprocess.PIPE, universal_newlines=True).stdout
  with open('BNC_tagged.txt','a',encoding='utf-8') as f:
    f.write(parsed)
    #f.write('\n')
  print(time()-start)
