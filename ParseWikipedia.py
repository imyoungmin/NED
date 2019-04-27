import importlib
import WikiParser.Parser as Parser
importlib.reload( Parser )

# File locations.
_ROOT = "/Volumes/YoungMinEXT/2014/"													# The root directory of the Wikipedia files.
_Multistream_Index = _ROOT + "enwiki-20141106-pages-articles-multistream-index.txt"		# Use the multistream Wikipedia dump to save space.
_Multistream_Dump = _ROOT + "enwiki-20141106-pages-articles-multistream.xml.bz2"
_Extracted_XML = _ROOT + "Extracted/Part4/"												# Contains extracted XML dumped files.

if __name__ is "__main__":
#	Parser.buildTFIDFDictionary( _Extracted_XML, clearDBCollections=True )

	# Compute IDF and update term weights in all of the documents.
#	Parser.computeIDFFromDocumentFrequencies()
#	Parser.computeAndNormalizeTermWeights()

	print( "Done!" )
