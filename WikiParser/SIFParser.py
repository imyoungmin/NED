import importlib
import bz2
import os
import time
import sys
from typing import Dict
from multiprocessing import Pool
import pymongo
from . import Parser as P
importlib.reload( P )


class SIFParser( P.Parser ):
	"""
	Parsing Wikipedia extracted archives to construct the SIF and the entity-id collections.
	"""


	def __init__( self ):
		"""
		Constructor.
		"""
		P.Parser.__init__( self )

		# Defining connections to collections for TFIDF.
		self._mWord_Embeddings = self._mNED["word_embeddings"]		# {_id:str, e:[float1, float2,...], f:int}.
		self._mSif_Documents = self._mNED["sif_documents"]			# {_id:int, w:[word1, word2,...], f:[int1, int2,...]}.

		self._wordMap: Dict[str, bool] = {}							# Cache for words from word_embeddings collection.
		self._loadWordMap()


	def _loadWordMap( self ):
		"""
		Populate the cache word map.
		"""
		print( "[!] Loading unique terms from word_embeddings collection... " )
		startTime = time.time()
		i = 0
		for t in self._mWord_Embeddings.find( {}, projection={ "_id": True } ):
			self._wordMap[t["_id"]] = True
			i += 1
			if i % 100000 == 0:
				print( "  Loaded", i )
		endTime = time.time()
		print( "[!] Done loading", i, "tokens after", endTime - startTime, "secs" )


	def buildWordEmbeddings( self, filePath: str ):
		"""
		Collect the word embeddings from FastText pre-training file.
		:param filePath: Word embeddings file.
		"""
		print( "------- Building word embeddings collection -------" )

		with open( filePath, "r", encoding="utf-8", newline="\n", errors="ignore" ) as fIn:
			n, d = fIn.readline().split()
			print( "[*] Starting to load", n, "word embeddings of dimension", d )
			startTime = time.time()
			totalRequests = 0
			MAX_REQUESTS = 20000							# We'll do bulk insertions.
			requests = []

			for line in fIn:
				tokens = line.rstrip().split( " " )
				word = tokens[0]
				embedding = [float( e ) for e in tokens[1:]]
				if word != word.lower():
					print( "[W] Word", word, "is not lowercased!", file=sys.stderr )
				requests.append( { "_id": word, "e": embedding, "f": 0.0 } )
				totalRequests += 1
				if len( requests ) == MAX_REQUESTS:			# Batch ready?
					self._mWord_Embeddings.insert_many( requests )
					print( "[*]", totalRequests, "processed" )
					requests = []

			if requests:
				self._mWord_Embeddings.insert_many( requests )
				print( "[*]", totalRequests, "processed" )

			endTime = time.time()
			print( "[*] Finished loading", totalRequests, "word embeddings in", endTime - startTime, "seconds" )


	def buildSIFDocuments( self, extractedDir ):
		"""
		Build the sif_documents collection by tokenizing content-based Wikipedia pages.
		Skip processing disambiguation pages, lists, and Wikipedia templates, files, etc.
		Use this method for incremental tokenization of extracted Wikipedia BZ2 files.
		Afterwards, use computeIDFFromDocumentFrequencies() and computeAndNormalizeTermWeights() to finalize TFIDF structs.
		:param extractedDir: Directory where the extracted BZ2 files are located: must end in "/".
		"""
		print( "------- Building SIF collections -------" )
		nEntities = 0												# Total number of tokenized documents in this pass.

		startTotalTime = time.time()
		directories = os.listdir( extractedDir )					# Get directories of the form AA, AB, AC, etc.
		for directory in directories:
			fullDir = extractedDir + directory
			if os.path.isdir( fullDir ):
				print( "[*] Processing", directory )
				files = os.listdir( fullDir )						# Get all files in current parsing directory, e.g. AA/wiki_00.bz2.
				files.sort()
				for file in files:
					fullFile = fullDir + "/" + file
					if os.path.isfile( fullFile ) and P.Parser._FilenamePattern.match( file ):

						# Read bz2 file and process it.
						startTime = time.time()
						with bz2.open( fullFile, "rt", encoding="utf-8" ) as bz2File:
							documents = self._extractWikiPagesFromBZ2( bz2File.readlines() )

						# Use multithreading to tokenize each extracted document.
						pool = Pool()
						documents = pool.map( P.Parser.tokenizeDoc, documents )	# Each document object in its own thread.
						pool.close()
						pool.join()												# Close pool and wait for work to finish.

						# Update DB collections.
						nEntities += self._updateSIFCollectionsAndEntityIDs( documents )

						endTime = time.time()
						print( "[**] Done with", file, ":", endTime - startTime )

		endTotalTime = time.time()
		print( "[!] Total number of entities:", nEntities, "in", endTotalTime - startTotalTime, "secs" )


	def _updateSIFCollectionsAndEntityIDs( self, documents ):
		"""
		Write data to sif_documents, update word_embeddings, and insert entity information to entity_id collections.
		:param documents: A list of dictionaries of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
		:return: Number of non-empty entity documents processed.
		"""

		# We only update terms from a document iff these terms exist in the word_embeddings collection.
		checkedDocuments = []
		requests = []
		totalRequests = 0
		BATCH_SIZE = 50000										# We'll do bulk update to word_embeddings.
		for doc in documents:
			if not doc: continue								# Skip empty documents.

			words = []											# Put tokens that DO exist in word_embeddings in this list with
			freqs = []											# matching indexes in freqs.
			for w in doc["tokens"]:
				if self._wordMap.get( w ) is None: continue		# Skip words that are not in the vocabulary.

				f = doc["tokens"][w]
				words.append( w )
				freqs.append( f )

				# Try to update word_embeddings by increasing tokens frequencies.
				requests.append( pymongo.UpdateOne( { "_id": w }, { "$inc": { "f": f } } ) )
				totalRequests += 1
				if len( requests ) >= BATCH_SIZE:
					self._mWord_Embeddings.bulk_write( requests )
					print( "[*]", totalRequests, "processed" )
					requests = []

			if not words: continue								# Check the document still have valid tokens.

			# New valid document.
			checkedDocuments.append( { "_id": doc["id"], "e": doc["title"], "w": words, "f": freqs } )

		# Send remaining word_embeddings update requests.
		if requests:
			self._mWord_Embeddings.bulk_write( requests )
			print( "[*]", totalRequests, "processed" )

		# Bulk insertion of term frequencies for each checked entity document.
		sd = [ { "_id": doc["_id"], "w": doc["w"], "f": doc["f"] } for doc in checkedDocuments ]
		self._mSif_Documents.insert_many( sd )

		# Insert Wikipedia titles and documents IDs in entity_id collection.
		# This serves the purpose of knowing which entities DO exist in the KB.
		# TODO: Remove comments once changes are done.
		bulkEntities = [ { "_id": doc["_id"], "e": doc["e"], "e_l": doc["e"].lower() } for doc in checkedDocuments ]
		# self._mEntity_ID.insert_many( bulkEntities )
		return len( bulkEntities )


	def initDBCollections( self ):
		"""
		Reset the TFIDF DB collections to start afresh.
		"""
		# self._mWord_Embeddings.drop()		# TODO: Remove comments after refactoring.
		self._mSif_Documents.drop()
		# self._mEntity_ID.drop()

		# Create indices on (re)created collections.
		# self._mEntity_ID.create_index( [("e", pymongo.ASCENDING)], unique=True )
		# self._mEntity_ID.create_index( [("e_l", pymongo.ASCENDING)] )		# Can't be unique: example - ALGOL and Algol both exist as entity names.

		print( "[!] Collections have been dropped")


	def addMissingLowercaseEntityName( self ):
		"""
		Execute this function to add "e_l" to the entity_id collection ONLY in the case it's missing.
		"""
		print( "------- Adding missing column 'e_l' to entity_id collection -------" )

		startTime = time.time()

		# Create unique index on missing column.
		self._mEntity_ID.create_index( [("e_l", pymongo.ASCENDING)] )

		requests = []  									# We'll use bulk writes to speed up process.
		BATCH_SIZE = 10000
		totalRequests = 0
		for t in self._mEntity_ID.find():
			requests.append( pymongo.UpdateOne( { "_id": t["_id"] }, { "$set": { "e_l": t["e"].lower() } } ) )
			totalRequests += 1
			if len( requests ) == BATCH_SIZE:  			# Send lots of update requests.
				self._mEntity_ID.bulk_write( requests )
				print( "[*]", totalRequests, "processed" )
				requests = []
		if requests:
			self._mEntity_ID.bulk_write( requests )  	# Process remaining requests.
			print( "[*]", totalRequests, "processed" )
		endTime = time.time()
		print( "[!] Done after", endTime - startTime, "secs." )