import importlib
import bz2
import os
import time
import math
from multiprocessing import Pool
import pymongo
from . import Parser as P
import Tokenizer
importlib.reload( Tokenizer )
importlib.reload( P )


class TFIDFParser( P.Parser ):
	"""
	Parsing Wikipedia extracted archives to construct the TFIDF and the entity-id collections.
	"""


	def __init__( self ):
		"""
		Constructor.
		"""
		P.Parser.__init__( self )

		# Defining connections to collections for TFIDF.
		self._mIdf_Dictionary = self._mNED["idf_dictionary"]						# {_id:str, idf:float}.
		self._mTf_Documents = self._mNED["tf_documents"]							# {_id:int, t:[t1, ..., tn], w:[w1, ..., wn]}.


	def buildTFIDFDictionary( self, extractedDir ):
		"""
		Build the TF-IDF dictionary by tokenizing content-based Wikipedia pages.
		Skip processing disambiguation pages, lists, and Wikipedia templates, files, etc.
		Use this method for incremental tokenization of extracted Wikipedia BZ2 files.
		Afterwards, use computeIDFFromDocumentFrequencies() and computeAndNormalizeTermWeights() to finalize TFIDF structs.
		:param extractedDir: Directory where the extracted BZ2 files are located: must end in "/".
		"""
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
						nEntities += self._updateTFIDFCollections( documents )

						endTime = time.time()
						print( "[**] Done with", file, ":", endTime - startTime )

		endTotalTime = time.time()
		print( "[!] Total number of entities:", nEntities, "in", endTotalTime - startTotalTime, "secs" )


	def computeIDFFromDocumentFrequencies( self ):
		"""
		Use the document frequencies stored in the idf_dictionary collection to calculate log(N/(df_t + 1)).
		"""
		nEntities = self._mTf_Documents.count()					# Retrieve number of entities and terms in DB.
		nTerms = self._mIdf_Dictionary.count()
		startTime = time.time()
		LOG_N = math.log( nEntities )

		print( "[!] Detected", nEntities, "entities in ned.tf_documents collection" )
		print( "[!] Computing IDF from document frequencies for", nTerms, "in ned.idf_dictionary collection" )
		requests = []											# We'll use bulk writes to speed up process.
		BATCH_SIZE = 10000
		totalRequests = 0
		for t in self._mIdf_Dictionary.find():
			requests.append( pymongo.UpdateOne( {"_id": t["_id"]}, {"$set": {"idf": LOG_N - math.log( t["idf"] + 1.0 )}} ) )
			totalRequests += 1
			if len( requests ) == BATCH_SIZE:					# Send lots of update requests.
				self._mIdf_Dictionary.bulk_write( requests )
				print( "[*]", totalRequests, "processed" )
				requests = []

		if requests:
			self._mIdf_Dictionary.bulk_write( requests )		# Process remaining requests.
			print( "[*]", totalRequests, "processed" )

		endTime = time.time()
		print( "[!] Done after", endTime - startTime, "secs." )


	def computeAndNormalizeTermWeights( self ):
		"""
		Compute weights for each entity document's terms by using the formula: [0.5 + 0.5*f(t,d)/MaxFreq(d)]*[log(N/(df_t + 1))].
		Each factor is already stored in tf_documents and idf_dictionary; so we need to combine them and then normalize the weight.
		"""
		startTime = time.time()
		print( "[!] Computing and normalizing term frequency weights in entity documents... " )
		requests = []											# We'll use bulk writes to speed up process.
		BATCH_SIZE = 200
		totalRequests = 0
		for e in self._mTf_Documents.find():					# For each entity document....
			idfDict = {}
			for termIdf in self._mIdf_Dictionary.find( {"_id": {"$in": e["t"]}} ):
				idfDict[termIdf["_id"]] = termIdf["idf"]		# Get the IDF of its terms...
			weights = []
			sumW2 = 0.0
			for i in range( len( e["t"] ) ):					# Multiply each TF by IDF...
				w = e["w"][i] * idfDict[e["t"][i]]
				sumW2 += w * w
				weights.append( w )
			normFactor = math.sqrt( sumW2 )
			for i in range( len( e["t"] ) ):					# And normalize weight by document length.
				weights[i] = weights[i] / normFactor

			requests.append( pymongo.UpdateOne( {"_id": e["_id"]}, {"$set": {"w": weights}} ) )
			totalRequests += 1
			if len( requests ) == BATCH_SIZE:					# Send lots of update requests.
				self._mTf_Documents.bulk_write( requests )
				print( "[*]", totalRequests, "processed" )
				requests = []

		if requests:											# Process remaining requests.
			self._mTf_Documents.bulk_write( requests )
			print( "[*]", totalRequests, "processed" )

		endTime = time.time()
		print( "[!] Done after", endTime - startTime, "secs." )


	def _updateTFIDFCollections( self, documents ):
		"""
		Write collections related to TF-IDF computations.
		:param documents: A list of dictionaries of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
		:return: Number of non-empty entity documents processed.
		"""
		# Insert Wikipedia titles and documents IDs in entity_id collection.
		# This serves the purpose of knowing which entities DO exist in the KB.
		bulkEntities = [{ "_id": doc["id"], "e": doc["title"] } for doc in documents if doc]		# Skip any empty document.
		self._mEntity_ID.insert_many( bulkEntities )

		# Upsert terms and document frequencies in idf_dictionary collection.
		for doc in documents:
			if not doc: continue								# Skip empty documents.

			s = set( doc["tokens"] )  							# Set of terms to upsert.
			toUpdate = set()  									# Set of term documents IDs to update.
			for t in self._mIdf_Dictionary.find( { "_id": { "$in": list( s ) } } ):
				toUpdate.add( t["_id"] )
				s.remove( t["_id"] )

			if s:  												# Insert new term documents if there are terms left in set.
				toInsert = [{ "_id": w, "idf": 1 } for w in s]  # List of new term documents to bulk-insert in collection.
				self._mIdf_Dictionary.insert_many( toInsert )

			if toUpdate:
				self._mIdf_Dictionary.update_many( { "_id": { "$in": list( toUpdate ) } },
												   { "$inc": { "idf": +1.0 } } )  	# Basically increase by one the term document frequency.

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

		self._mTf_Documents.insert_many( ds )
		return len( bulkEntities )


	def initDBCollections( self ):
		"""
		Reset the TFIDF DB collections to start afresh.
		"""
		self._mIdf_Dictionary.drop()
		self._mTf_Documents.drop()
		self._mEntity_ID.drop()

		# Create indices on (re)created collections.
		self._mEntity_ID.create_index( [("e", pymongo.ASCENDING)], unique=True )

		print( "[!] Collections have been dropped")