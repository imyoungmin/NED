#  NED: Collective Named Entity Disambiguation via Personalized Page Rank and Context Embeddings
### CS 273 · Data and Knowledge Bases Term Project ###
University of California, Santa Barbara

By Luis Ángel (c) June, 2019.

## Introduction ##

We present a method to solve the problem of **Named Entity Disambiguation**.  Our approach is based on
the foundations provided in the following research papers:
- _Graph Ranking for Collective Named Entity Disambiguation_, by Alhelbawy, A., and Gaizauskas, R. 2014.
- _Linking Named Entities in Tweets with Knowledge Base via User Interest Modeling_, by Shen, W., Wang, J., 
Luo, P., and Wang, M. 2013.
- _A Simple but Though-to-Beat Baseline for Sentence Embeddings_, by Arora, S., Liang, Y., and Ma, T. 2017.

In particular, we combine the **sentence embeddings** approach to formalize a local context feature metric 
for named entity mentions with the **topical relatedness** expressed through _Wikilinks_ to collaboratively 
resolve the NED problem using **personalized PageRank** with **maximal discriminant selection**.

## Materials ##

The present work has been implemented on **Python 3.6** (_Anaconda_ environment) and **MongoDB** as underlying 
DB engine.  We make use of a processed version of **Wikipedia** (c) (v. 2014) and a set of 300-dimensional 
pre-trained English-word embeddings obtained from **fastText** (c).  The overall process may be divided into two
stages:
1. Processing Wikipedia and word embeddings to generate the DB collections.  The steps for this long task are
well documented in the corresponding README file under the WikiParser directory.  We refer the reader to that
location for further details before attempting anything in step 2.
2. Evaluation of our method on an input text file or on the **CoNLL 2003 Reuter's Newswire Dataset** for the NER 
task (https://www.clips.uantwerpen.be/conll2003/ner/).  This step is further described below.

## Disambiguating Entities ##

We offer two ways to evaluate the performance of our method: on a user-defined input text and on the CoNLL 2003
dataset.  These tasks are shown by example in the script `Main.py`:
- If you want to disambiguate entities in a textfile, get ready to provide your text with at least one entity
mention.  Each entity mention should be annotated or denoted by using `[[...]]`.  We refer the reader to the
test file `Datasets/madonna.txt` to check out the expected format of the input to our system.  Please be aware that we do not differentiate between the mentions `Madonna` and `madonna` as all of the surface forms are normalized to a lowercase equivalence.  Furthermore, if there is only one distinct surface form in your text,
our method defaults to evaluating just the prior probability of candidate mapping entities and their context
similarities with respect to the 50-token windows around each occurrence of the entity mention.  If more than 
one distinct entity mention appears in the text, we use the full version of our approach: initial score + 
propagation score + maximal discriminant selection.  Once you have your input text ready, execute the 
corresponding following lines in `Main.py`:
```
# Disambiguating regular text with labeled named entity mentions.
results = T.Task.disambiguateTextFile( "yourFile.txt" )
```
- If you want to verify the perfomance of our system on the CoNLL 2003 dataset, removed the comments from the
appropriate lines in `Main.py`:
```
# Disambiguating entities in a big dataset to measure accuracy.
T.Task.debug = False
T.Task.evaluateAccuracy( "Datasets/AIDA-YAGO2-dataset.tsv" )
```

For more details on the algorithms and methodologies employed in the current project, please check out the
`Documentation` folder.  Do not forget to keep the `mongod` service running in order to execute the code!

 