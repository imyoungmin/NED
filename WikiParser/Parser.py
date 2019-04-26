import importlib
import bz2
import os
import re
import sys
import time
import math
import PorterStemmer as PS
from multiprocessing import Pool
from nltk.corpus import stopwords
import pymongo
from pymongo import MongoClient
from urllib.parse import unquote
import html
import Tokenizer
importlib.reload( Tokenizer )

# File locations.
_ROOT = "/Volumes/YoungMinEXT/"															# The root directory of the Wikipedia files.
_Multistream_Index = _ROOT + "enwiki-20141106-pages-articles-multistream-index.txt"		# Use the multistream Wikipedia dump to save space.
_Multistream_Dump = _ROOT + "enwiki-20141106-pages-articles-multistream.xml.bz2"
_Extracted_XML = "/Users/youngmin/Downloads/Extracted/"									# Contains extracted XML dumped files.

_UrlProtocols = [
    'bitcoin', 'ftp', 'ftps', 'geo', 'git', 'gopher', 'http', 'https', 'irc', 'ircs', 'magnet', 'mailto', 'mms', 'news',
    'nntp', 'redis', 'sftp', 'sip', 'sips', 'sms', 'ssh', 'svn', 'tel', 'telnet', 'urn', 'worldwind', 'xmpp'
]

# Regular expressions.
_FilenamePattern = re.compile( r"^wiki_\d+\.bz2$", re.I )										# Checking only the right files.
_DocStartPattern = re.compile( r"^<doc.+?id=\"(\d+)\".+?title=\"\s*(.+)\s*\".*?>$", re.I )		# Document head tag.
_DisambiguationPattern = re.compile( r"^(.+)\s+\(disambiguation\)$", re.I )						# Disambiguation title pattern.
_ListTitlePattern = re.compile( r"^lists?\s+of", re.I )											# List title pattern.
_ExternalLinkPattern = re.compile( r"<a.*?href=['\"]((" + r"|".join( _UrlProtocols ) + r")(?:%3a|:)|//).*?>\s*(.*?)\s*</a>", re.I )	# None Wikilinks.
_LinkPattern = re.compile( r"<a.*?href=\"\s*(.+?)\s*\".*?>\s*(.*?)\s*</a>", re.I )				# Links: internals and externals.
_UrlPattern = re.compile( r"(?:" + r"|".join( _UrlProtocols ) + r")(?:%3a|:)/.*?", re.I )		# URL pattern not inside a link.
_PunctuationOnlyPattern = re.compile( r"^\W+$" )

# Undesired tags to remove *before* tokenizing text.
_undesiredTags = ["<onlyinclude>", "</onlyinclude>", "<nowiki>", "</nowiki>", "<br>", "<br/>"]

# Stop words set: use nltk.download('stopwords').  Then add: "n't" and "'s".
_stopWords = set( stopwords.words( "english" ) )

# Porter stemmer.
_porterStemmer = PS.PorterStemmer()

# MongoDB connection and collection variables.
_mClient = MongoClient( "mongodb://localhost:27017/" )
_mNED = _mClient.ned											# Connection to DB 'ned'.
_mIdf_Dictionary = _mNED["idf_dictionary"]						# {_id:str, idf:float}.
_mEntity_ID = _mNED["entity_id"]								# {_id:int, e:str}.
_mTf_Documents = _mNED["tf_documents"]							# {_id:int, t:[t1, ..., tn], w:[w1, ..., wn]}.

# Total number of tokenized documents (used for IDF).
_nEntities = 0


def buildTFIDFDictionary():
	"""
	Build the TF-IDF dictionary by tokenizing non-disambiguation
	"""
	global _nEntities

	_initDBCollections()
	_nEntities = 0

	directories = os.listdir( _Extracted_XML )					# Get directories of the form AA, AB, AC, etc.
	for directory in directories:
		fullDir = _Extracted_XML + directory
		if os.path.isdir( fullDir ):
			print( "[*] Processing", directory )
			files = os.listdir( fullDir )						# Get all files in current parsing directory, e.g. AA/wiki_00.bz2.
			files.sort()
			for file in files:
				fullFile = fullDir + "/" + file
				if os.path.isfile( fullFile ) and _FilenamePattern.match( file ):

					# Read bz2 file and process it.
					startTime = time.time()
					with bz2.open( fullFile, "rt", encoding="utf-8" ) as bz2File:
						documents = _extractWikiPagesFromBZ2( bz2File.readlines() )

					# Use multithreading to tokenize each extracted document.
					pool = Pool()
					documents = pool.map( _tokenizeDoc, documents )		# Each document object in its own thread.
					pool.close()
					pool.join()											# Close pool and wait for work to finish.

					# Update DB collections.
					_nEntities += _updateTFIDFCollections( documents )

					endTime = time.time()
					print( "[**] Done with", file, ":", endTime - startTime )

	# Compute IDF and update term weights in all of the documents.
	_computeIDFFromDocumentFrequencies()
	_computeAndNormalizeTermWeights()

	print( "[!] Total number of entities:", _nEntities )


def _computeIDFFromDocumentFrequencies():
	"""
	Use the document frequencies stored in the idf_dictionary collection to calculate log(N/(df_t + 1)).
	"""
	global _nEntities
	startTime = time.time()
	LOG_N = math.log( _nEntities )

	print( "[!] Computing IDF from document frequencies..." )
	requests = []														# We'll use bulk writes to speed up process.
	BATCH_SIZE = 10000
	totalRequests = 0
	for t in _mIdf_Dictionary.find():
		requests.append( pymongo.UpdateOne( {"_id": t["_id"]}, {"$set": {"idf": LOG_N - math.log( t["idf"] + 1.0 )}} ) )
		totalRequests += 1
		if len( requests ) == BATCH_SIZE:								# Send lots of update requests.
			_mIdf_Dictionary.bulk_write( requests )
			print( "[*]", totalRequests, "processed" )
			requests = []

	if requests:
		_mIdf_Dictionary.bulk_write( requests )							# Process remaining requests.
		print( "[*]", totalRequests, "processed" )

	endTime = time.time()
	print( "[!] Done after", endTime - startTime, "secs." )


def _computeAndNormalizeTermWeights():
	"""
	Compute weights for each entity document's terms by using the formula: [0.5 + 0.5*f(t,d)/MaxFreq(d)]*[log(N/(df_t + 1))].
	Each factor is already stored in tf_documents and idf_dictionary; so we need to combine them and then normalize the weight.
	"""
	startTime = time.time()
	print( "[!] Computing and normalizing term frequency weights in entity documents... " )
	requests = []														# We'll use bulk writes to speed up process.
	BATCH_SIZE = 100
	totalRequests = 0
	for e in _mTf_Documents.find():										# For each entity document....
		idfDict = {}
		for termIdf in _mIdf_Dictionary.find( {"_id": {"$in": e["t"]}} ):
			idfDict[termIdf["_id"]] = termIdf["idf"]					# Get the IDF of its terms...
		weights = []
		sumW2 = 0.0
		for i in range( len( e["t"] ) ):								# Multiply each TF by IDF...
			w = e["w"][i] * idfDict[e["t"][i]]
			sumW2 += w * w
			weights.append( w )
		normFactor = math.sqrt( sumW2 )
		for i in range( len( e["t"] ) ):								# And normalize weight by document length.
			weights[i] = weights[i] / normFactor

		requests.append( pymongo.UpdateOne( {"_id": e["_id"]}, {"$set": {"w": weights}} ) )
		totalRequests += 1
		if len( requests ) == BATCH_SIZE:								# Send lots of update requests.
			_mTf_Documents.bulk_write( requests )
			print( "[*]", totalRequests, "processed" )
			requests = []

	if requests:														# Process remaining requests.
		_mTf_Documents.bulk_write( requests )
		print( "[*]", totalRequests, "processed" )

	endTime = time.time()
	print( "[!] Done after", endTime - startTime, "secs." )


def _updateTFIDFCollections( documents ):
	"""
	Write collections related to TF-IDF computations.
	:param documents: A list of dictionaries of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
	:return: Number of non-empty entity documents processed.
	"""
	# Insert Wikipedia titles and documents IDs in entity_id collection.
	# This serves the purpose of knowing which entities DO exist in the KB.
	bulkEntities = [{ "_id": doc["id"], "e": doc["title"] } for doc in documents if doc]		# Skip any empty document.
	_mEntity_ID.insert_many( bulkEntities )

	# Upsert terms and document frequencies in idf_dictionary collection.
	for doc in documents:
		if not doc: continue								# Skip empty documents.

		s = set( doc["tokens"] )  							# Set of terms to upsert.
		toUpdate = set()  									# Set of term documents IDs to update.
		for t in _mIdf_Dictionary.find( { "_id": { "$in": list( s ) } } ):
			toUpdate.add( t["_id"] )
			s.remove( t["_id"] )

		if s:  												# Insert new term documents if there are terms left in set.
			toInsert = [{ "_id": w, "idf": 1 } for w in s]  # List of new term documents to bulk-insert in collection.
			_mIdf_Dictionary.insert_many( toInsert )

		if toUpdate:
			_mIdf_Dictionary.update_many( { "_id": { "$in": list( toUpdate ) } }, {
				"$inc": { "idf": +1.0 } } )  				# Basically increase by one the term document frequency.

	# Bulk insertion of term frequencies for each parsed document.
	ds = []
	for doc in documents:
		if not doc: continue								# Skip empty documents.

		terms = []
		weights = []
		for t in doc["tokens"]:
			terms.append( t )								# One array with terms, and another with weights (for now, frequencies).
			weights.append( doc["tokens"][t] )
		ds.append( {"_id": doc["id"], "t": terms, "w": weights} )

	_mTf_Documents.insert_many( ds )
	return len( bulkEntities )


def _extractWikiPagesFromBZ2( lines ):
	"""
	Extract non-disambiguation articles from extracted Wikipedia bz2 archives.
	:param lines: A list of sentences.
	:return: List of document objects: {id:int, title:str, lines:[str]}.
	"""
	documents = []												# We collect all documents in a list.
	doc = {}													# A processed document is a dictionary: {id: x, title: y}

	extractingContents = False									# On/off when we enter the body of a document.
	isDisambiguation = False									# On/off when processing a disambiguation document.
	isList = False
	firstLine = False											# Skip first line since it's a repetition of the title.
	for line in lines:
		line = line.strip()
		if line == "": continue									# Skip empty lines.
		if not extractingContents:								# Wait for the sentinel: <doc ...>
			m = _DocStartPattern.match( line )
			if m:														# Start of document?
				title = html.unescape( m.group( 2 ) )					# Title without html entities.
				if _DisambiguationPattern.match( title ) is not None:	# Skipping disambiguation pages.
					isDisambiguation = True
				elif _ListTitlePattern.match( title ) is not None:		# Skipping list pages.
					isList = True

				doc = { "id": int(m.group(1)), 							# A new dictionary for this document.
					    "title": title,
						"lines": [] }
				extractingContents = True								# Turn on the flag: we started reading a document.
				firstLine = True										# Will be reading title repetion next in the first line.
			else:
				print( "Line:", line, "is not in any document!", file=sys.stderr )
		else:
			if line == "</doc>":
				if not isDisambiguation and not isList:
					documents += [doc]  								# Add extracted document to list for further processing in caller function.
				extractingContents = False
				isDisambiguation = False
				isList = False
			elif not isDisambiguation and not isList:					# Process text within <doc></doc> for non disambiguation pages.
				if firstLine:
					firstLine = False
				else:
#					_ExternalLinkPattern.sub( r"\3", line )				# Replace external links for their anchor texts.  (No interwiki links affected).
					doc["lines"].append( line.lower() )					# Add (lowercase) line of text for multithreading.

	return documents


def _tokenizeDoc( doc ):
	"""
	Tokenize a document object.
	:param doc: Document dictionary to process: {id:int, title:str, lines:[str]}.
	:return: A dictionary of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
	"""
	nDoc = { "id": doc["id"], "title": doc["title"], "tokens": { } }

	maxFreq = 0
	for line in doc["lines"]:
		for tag in _undesiredTags:  									# Remove undesired tags.
			line = line.replace( tag, "" )

		line = _LinkPattern.sub( r"\2", line )  						# Replace all links with their anchor text.

		tokens = Tokenizer.tokenize( line )  							# Tokenize a lower-cased version of article text.
		tokens = [w for w in tokens if not w in _stopWords ]  			# Remove stop words.

		for token in tokens:
			if len( token ) <= 128:												# Skip too long tokens.
				if _UrlPattern.search( token ) is None:							# Skip URLs.
					if _PunctuationOnlyPattern.match( token ) is None:			# Skip patterns like '...' and '#' and '--'
						t = _porterStemmer.stem( token, 0, len( token ) - 1 )	# Stem token.
						if nDoc["tokens"].get( t ) is None:
							nDoc["tokens"][t] = 1								# Create token in dictionary if it doesn't exist.
						else:
							nDoc["tokens"][t] += 1
						maxFreq = max( nDoc["tokens"][t], maxFreq )				# Keep track of maximum term frequency within document.

	# Normalize frequency with formula: TF(t,d) = 0.5 + 0.5*f(t,d)/MaxFreq(d).
	if maxFreq == 0:
		print( "[W] Empty document:", nDoc, file=sys.stderr )
		return {}

	for token in nDoc["tokens"]:
		nDoc["tokens"][token] = 0.5 + 0.5 * nDoc["tokens"][token] / maxFreq

#	print( "[***]", nDoc["id"], nDoc["title"], "... Done!" )
	return nDoc


def _initDBCollections():
	"""
	Reset the DB collections to start afresh.
	"""
	_mIdf_Dictionary.drop()
	_mTf_Documents.drop()
	_mEntity_ID.drop()

	# Create indices on (re)created collections.
	_mEntity_ID.create_index( [("e", pymongo.ASCENDING)], unique=True )

	print( "[!] Collections have been dropped")



# def go():
# 	"""
# 	Launch Wikipedia parsing process.
# 	:return:
# 	"""
# 	print( "[!] Started to parse Wikipedia files" )
# 	with open( _Multistream_Index, "r", encoding="utf-8" ) as indexFile:
# 		seekByte = -1
# 		for lineNumber, line in enumerate( indexFile ):			# Read index line by line.
# 			components = line.strip().split( ":" )				# [ByteStart, DocID, DocTitle]
# 			newSeekByte = int( components[0] )					# Find the next seek byte start that is different to current (defines a block).
#
# 			if seekByte == -1:									# First time reading seek byte from file.
# 				seekByte = newSeekByte
# 				continue
#
# 			if newSeekByte != seekByte:							# Changed seek-byte?
# 				count = newSeekByte - seekByte					# Number of bytes to read from bz2 stream.
# 				_processBZ2Block( seekByte, count )		# Read Wikipedia docs in this block.
# 				seekByte = newSeekByte
# 				break	# TODO: Remove to process all blocks.
#
# 		# TODO: Process the last seek byte count = -1.
#
# 	print( "[!] Finished parsing Wikipedia" )
#
#
# def _processBZ2Block( seekByte, count ):
# 	with open( _Multistream_Dump, "rb" ) as bz2File:
# 		bz2File.seek( seekByte )
# 		block = bz2File.read( count )
#
# 		dData = bz2.decompress( block )
# 		print( dData )