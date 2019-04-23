import importlib
import bz2
import os
import re
import sys
import time
from multiprocessing import Pool
from nltk.corpus import stopwords
import pymongo
from pymongo import MongoClient
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
_ExternalLinkPattern = re.compile( r"<a.*?href=['\"]((" + r"|".join( _UrlProtocols ) + r")%3A|//).*?>\s*(.+?)\s*</a>", re.I )	# None Wikilinks.
_LinkPattern = re.compile( r"<a.*?href=\"\s*(.+?)\s*\".*?>\s*(.+?)\s*</a>", re.I )				# Links: internals and externals.
_PunctuationOnlyPattern = re.compile( r"^\W+$" )

# Undesired tags to remove *before* tokenizing text.
_undesiredTags = ["<onlyinclude>", "</onlyinclude>", "<nowiki>", "</nowiki>"]

# Stop words set: use nltk.download('stopwords').  Then add: "n't" and "'s".
_stopWords = set( stopwords.words( "english" ) )

# MongoDB connection and collection variables.
_mClient = MongoClient( "mongodb://localhost:27017/" )
_mNED = _mClient.ned											# Connection to DB 'ned'.
_mIdf_Dictionary = _mNED["idf_dictionary"]						# {_id:str, idf:float}.
_mEntity_ID = _mNED["entity_id"]								# {_id:int, e:str}.

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
			for file in files:
				fullFile = fullDir + "/" + file
				if os.path.isfile( fullFile ) and _FilenamePattern.match( file ):

					# Read bz2 file and process it.
					startTime = time.time()
					with bz2.open( fullFile, "rt", encoding="utf-8" ) as bz2File:
						documents = _extractWikiPagesFromBZ2( bz2File.readlines() )

					_nEntities += len( documents )

					# Use multithreading to tokenize each extracted document.
					pool = Pool()
					documents = pool.map( _tokenizeDoc, documents )		# Each document object in its own thread.
					pool.close()
					pool.join()											# Close pool and wait for work to finish.

					# Update DB collections.
					_updateTFIDFCollections( documents )

					endTime = time.time()
					print( "[**] Done with", file, ":", endTime - startTime )

	print( "[!] Total number of entities:", _nEntities )


def _updateTFIDFCollections( documents ):
	"""
	Write collections related to TF-IDF computations.
	:param documents: A list of dictionaries of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
	"""
	# Insert Wikipedia titles and documents IDs in entity_id collection.
	bulkEntities = [{ "_id": doc["id"], "e": doc["title"] } for doc in documents]
	_mEntity_ID.insert_many( bulkEntities )

	# Upsert terms and document frequencies in idf_dictionary collection.
	for doc in documents:
		s = set( doc["tokens"] )  							# Set of terms to upsert.
		existentTerms = _mIdf_Dictionary.find( { "_id": { "$in": list( s ) } } )
		toUpdate = []  										# List of term documents IDs to update.
		for t in existentTerms:
			toUpdate.append( t["_id"] )
			s.remove( t["_id"] )

		if s:  												# Insert new term documents if there are terms left in set.
			toInsert = [{ "_id": t, "idf": 1 } for t in s]  # List of new term documents to bulk-insert in collection.
			_mIdf_Dictionary.insert_many( toInsert )

		if toUpdate:
			_mIdf_Dictionary.update_many( { "_id": { "$in": toUpdate } }, {
				"$inc": { "idf": +1.0 } } )  				# Basically increase by one the term document frequency.


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
	for line in lines:
		line = line.strip()
		if line == "": continue									# Skip empty lines.
		if not extractingContents:								# Wait for the sentinel: <doc ...>
			m = _DocStartPattern.match( line )
			if m:														# Start of document?
				md = _DisambiguationPattern.match( m.group( 2 ) )
				if md:
					isDisambiguation = True
					print( "[***] Skipping", m.group(2) )				# Skipping disambiguation pages.

				doc = { "id": int(m.group(1)), 							# A new dictionary for this document.
					    "title": m.group( 2 ),
						"lines": [] }
				extractingContents = True								# Turn on the flag: we started reading a document.
			else:
				print( "Line:", line, "is not in any document!", file=sys.stderr )
		else:
			if line == "</doc>":
				if not isDisambiguation:
					documents += [doc]  								# Add extracted document to list for further processing in caller function.
				extractingContents = False
				isDisambiguation = False
			elif not isDisambiguation:									# Process text within <doc></doc> for non disambiguation pages
#				_ExternalLinkPattern.sub( r"\3", line )					# Replace external links for their anchor texts.  (No interwiki links affected).
				doc["lines"].append( line.lower() )						# Add (lowercase) line of text for multithreading.

	return documents


def _tokenizeDoc( doc ):
	"""
	Tokenize a document object.
	:param doc: Document dictionary to process: {id:int, title:str, lines:[str]}.
	:return: A dictionary of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
	"""
	nDoc = { "id": doc["id"], "title": doc["title"], "tokens": { } }

	for line in doc["lines"]:
		for tag in _undesiredTags:  									# Remove undesired tags.
			line = line.replace( tag, "" )

		line = _LinkPattern.sub( r"\2", line )  # Replace all links with their anchor text.

		tokens = Tokenizer.tokenize( line )  							# Tokenize a lower-cased version of article text.
		tokens = [w for w in tokens if not w in _stopWords ]  			# Remove stop words.

		for token in tokens:
			if _PunctuationOnlyPattern.match( token ) is None:			# Skip patterns like ... #
				if nDoc["tokens"].get( token ) is None:
					nDoc["tokens"][token] = 1							# Create token in dictionary if it doesn't exist.
				else:
					nDoc["tokens"][token] += 1

	print( "[***]", doc["title"], "... Done!" )
	return nDoc


def _initDBCollections():
	"""
	Reset the DB collections to start afresh.
	"""
	_mIdf_Dictionary.drop()
	_mEntity_ID.drop()

	# Create indices on (re)created collections.
	_mEntity_ID.create_index( [("e", pymongo.ASCENDING)], unique=True )

	print( "[!] Collections have been dropped")



def go():
	"""
	Launch Wikipedia parsing process.
	:return:
	"""
	print( "[!] Started to parse Wikipedia files" )
	with open( _Multistream_Index, "r", encoding="utf-8" ) as indexFile:
		seekByte = -1
		for lineNumber, line in enumerate( indexFile ):			# Read index line by line.
			components = line.strip().split( ":" )				# [ByteStart, DocID, DocTitle]
			newSeekByte = int( components[0] )					# Find the next seek byte start that is different to current (defines a block).

			if seekByte == -1:									# First time reading seek byte from file.
				seekByte = newSeekByte
				continue

			if newSeekByte != seekByte:							# Changed seek-byte?
				count = newSeekByte - seekByte					# Number of bytes to read from bz2 stream.
				_processBZ2Block( seekByte, count )		# Read Wikipedia docs in this block.
				seekByte = newSeekByte
				break	# TODO: Remove to process all blocks.

		# TODO: Process the last seek byte count = -1.

	print( "[!] Finished parsing Wikipedia" )


def _processBZ2Block( seekByte, count ):
	with open( _Multistream_Dump, "rb" ) as bz2File:
		bz2File.seek( seekByte )
		block = bz2File.read( count )

		dData = bz2.decompress( block )
		print( dData )