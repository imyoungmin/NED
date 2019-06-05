# Compiling Word Embeddings and Wikipedia

## Prerequisites

1. Download the pretrained word embeddings from **FastText**: 
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
2. Download the following English `Wikipedia` dump files from https://archive.org/download/enwiki-20141106:
    1. Multistream index: `enwiki-20141106-pages-articles-multistream-index.txt`,
    2. Multistream dump: `enwiki-20141106-pages-articles-multistream.xml.bz2`, and
    3. Regular dump: `enwiki-20141106-pages-articles.xml`
3. Download the **WikiExtractor** from https://github.com/attardi/wikiextractor
4. Use the following command to extract plain text (preserving links and lists) into `bz2` archives 
from the regular `Wikipedia` English dump file: 
```
/usr/bin/python WikiExtractor.py /<in-dir>/enwiki-20141106-pages-articles.xml.bz2 
    -o /<out-dir>/Extracted -c 
    --no-templates --bytes 4M --links --lists --processes 3
```
After this step, you'll have `*.bz2` archives in several folders (e.g. `AA`) in the `Extracted/`
directory.  Notice that **WikiExtractor** works only for python 2.*.

## Parsing

We do the compilation of our dataset in 2 steps:

### 1. Compiling Word Embiddings and SIF Collections

Provide the right paths to your files above, and remove comments from `ParseWikipedia.py`, so that
only the following code is executed (`sifParser = SIF.SIFParser()` is always needed):
```
sifParser.initDBCollections()
sifParser.buildWordEmbeddings( _WORD_EMBEDDINGS )
```

The previous lines should have created the `word_embeddings` collection in the `ned` database.

Next, we compute term frequencies for every word in `word_embeddings` and also tokenize the 
extracted `*.bz2` archives from the *Prerequisites*.  To do this, we **highly** advise you distribute
all of the `AA`, `AB`, etc., folders into three or more parts (e.g. different directories like `Part1/`).
Then, comment out  `sifParser.initDBCollections()` and `sifParser.buildWordEmbeddings( _WORD_EMBEDDINGS )`
and make sure that only
```
sifParser.buildSIFDocuments( _Extracted_XML )
``` 
gets executed for as many partitions you have for the `*.bz2` files.  This process may take 
several hours, but in the end `ned.entity_id` and `ned.sif_documents` will have the final data we
need for similarity measurements.

Finally, execute the statement
```
sifParser.saveTotalWordCount()
```
so that the total word frequencies gest recorded into `Datasets/wordcount.txt` --this may take a
minute to compute.

### 2. Compiling disambiguation collections

Now, we'll build the surface forms dictionary, `ned_dictionary`, and the linking collection, `ned_linking`.
For this task comment out anything related to SIF computation from previous step (we don't want to erase
any currently loaded data in the SIF collections!).  Then, execute the following commands:
```
nedParser = NP.NEDParser()
nedParser.initDBCollections()
nedParser.parseSFFromEntityNames()
nedParser.parseSFsAndLsFromWikilinks( _Extracted_XML )
```
where the last line may be executed as many times as directories you have splited the extracted
`*.bz2` archives.  Whenever you need to re-execute the latter, recall to comment out `nedParser.initDBCollections()`
and `nedParser.parseSFFromEntityNames()` --i.e., these two commands should be executed once!

Finally, after the previous compilation is complete, proceed to execute only the following line:
```
nedParser.parseSFFromRedirectPages( _Multistream_Index, _Multistream_Dump )
```
This concludes the dataset creation for our disambiguation task.