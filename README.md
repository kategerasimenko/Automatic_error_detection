# Automatic error detection - project repository

## Text plan
1. Introduction
    1. Background
    2. Purpose - two purposes, check whether learner data is useful for error detection systems and create the system
2. Methods
    1. Corpora<br>BNC, REALEC
    2. Models<br>classification, language model, rules?
    3. Data processing<br>Modules and tools that are used<br>kenlm, UDPipe, Stanford POS tagger, pretrained w2v model, sklearn
3. Spellchecker<br>Aspell + language model
4. Articles and prepositions
     1. Literature review<br>Corpora, features, models
     2. Features
     3. Models
          1. L1 models - classifiers, use of LM
          2. Metaclassifier
     4. Results<br>Result tables, compare with only L1 models<br>Quality of found spans and proposed corrections
     5. Error analysis<br>Analyze errors made by the system
5. Conclusion
     1. Summary
     2. Further development of the project
