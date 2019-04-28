import importlib
from WikiParser import NEDParser as NP
from WikiParser import TFIDFParser as TFIDF
importlib.reload( NP )
importlib.reload( TFIDF )

# File locations.
_ROOT = "/Volumes/YoungMinEXT/2014/"													# The root directory of the Wikipedia files.
_Multistream_Index = _ROOT + "enwiki-20141106-pages-articles-multistream-index.txt"		# Use the multistream Wikipedia dump to save space.
_Multistream_Dump = _ROOT + "enwiki-20141106-pages-articles-multistream.xml.bz2"
_Extracted_XML = "/Users/youngmin/Downloads/Extracted/"									# Contains extracted XML dumped files.

if __name__ is "__main__":

	# Compute and fill collections for TFIDF.
	# tfIdfParser = TFIDF.TFIDFParser()
	# tfIdfParser.initDBCollections()
	# tfIdfParser.buildTFIDFDictionary( _Extracted_XML )
	# tfIdfParser.computeIDFFromDocumentFrequencies()
	# tfIdfParser.computeAndNormalizeTermWeights()

	# Compute surface forms and fill collections for NED.
	nedParser = NP.NEDParser()
	nedParser.initDBCollections()
	nedParser.parseSFFromEntityNames()
	nedParser.parseSFFromWikilinks( _Extracted_XML )
