import importlib
from WikiParser import NEDParser as NP
from WikiParser import SIFParser as SIF
importlib.reload( NP )
importlib.reload( SIF )

# File locations.
_ROOT = "/Volumes/YoungMinEXT/2014/"													# The root directory of the Wikipedia files.
_Multistream_Index = _ROOT + "enwiki-20141106-pages-articles-multistream-index.txt"		# Use the multistream Wikipedia dump to save space.
_Multistream_Dump = _ROOT + "enwiki-20141106-pages-articles-multistream.xml.bz2"
_Extracted_XML = _ROOT + "Extracted/"													# Contains extracted XML dumped files.
_WORD_EMBEDDINGS = _ROOT + "wiki.en.vec"												# Word vectors.

if __name__ is "__main__":

	# Compute and fill collections for SIF and word embeddings.
	sifParser = SIF.SIFParser()
	# sifParser.initDBCollections()
	# sifParser.buildWordEmbeddings( _WORD_EMBEDDINGS )
	# sifParser.buildSIFDocuments( _Extracted_XML )
	# sifParser.saveTotalWordCount()
	# sifParser.saveSIFDocumentsRawEmbeddings()

	# Compute surface forms and fill collections for NED.
	# nedParser = NP.NEDParser()
	# nedParser.initDBCollections()
	# nedParser.parseSFFromEntityNames()
	# nedParser.parseSFsAndLsFromWikilinks( _Extracted_XML )
	# nedParser.parseSFFromRedirectPages( _Multistream_Index, _Multistream_Dump )
