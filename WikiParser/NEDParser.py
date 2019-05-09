import importlib
import bz2
import os
import time
import re
from multiprocessing import Pool
from pymongo import MongoClient
import pymongo
from urllib.parse import unquote
import html
import sys
from . import Parser as P
importlib.reload( P )


class NEDParser( P.Parser ):
	"""
	Parsing Wikipedia extracted and multistream archives to construct the surface forms dictionary and the inter-links table.
	"""


	# Static members.
	_TitlePattern = re.compile( r"<title>\s*(.+?)\s*</title>", re.I )
	_RedirectTitlePattern = re.compile( r"<redirect\s+title=\"\s*(.+?)\s*\"\s*/>", re.I )


	def __init__( self ):
		"""
		Constructor.
		"""
		P.Parser.__init__( self )

		# Defining connections to collections for entity disambiguation.
		self._mNed_Dictionary = self._mNED["ned_dictionary"]			# {_id:str, m:{"e_1":int, "e_2":int,..., "e_n":int}}.	-- m stands for 'mapping.'
		self._mNed_Linking = self._mNED["ned_linking"]					# {_id:int, f:{"e_1":true, "e_2":true,..., "e_3":true}}.   -- f stands for 'from.'


	def parseSFFromEntityNames( self ):
		"""
		Grab the entity names from entity_id collection and insert their surface forms in ned_dictionary.
		"""
		nEntities = self._mEntity_ID.count() 							# Retrieve number of entities in DB.
		startTime = time.time()

		print( "------- Creating surface forms from Wikipedia titles -------" )
		print( "[!] Detected", nEntities, "entities in entity_id collection" )
		requests = []  													# We'll use bulk writes to speed up process.
		BATCH_SIZE = 10000
		totalRequests = 0
		for t in self._mEntity_ID.find():								# Surface forms are in lowercase.
			requests.append( pymongo.UpdateOne( { "_id": t["e_l"] }, { "$inc": { "m." + str( t["_id"] ): +1 } }, upsert=True ) )
			totalRequests += 1
			if len( requests ) == BATCH_SIZE:  							# Send lots of update requests.
				self._mNed_Dictionary.bulk_write( requests )
				print( "[*]", totalRequests, "processed" )
				requests = []

		if requests:
			self._mNed_Dictionary.bulk_write( requests )  				# Process remaining requests.
			print( "[*]", totalRequests, "processed" )

		endTime = time.time()
		print( "[!] Done after", endTime - startTime, "secs." )


	def parseSFsAndLsFromWikilinks( self, extractedDir ):
		"""
		Grab surface forms and link relationships from wikilinks in valid entity pages.
		Skip processing disambiguation pages, lists, and Wikipedia templates, files, etc.
		Use this method for incremental analysis of extracted Wikipedia BZ2 files.
		:param extractedDir: Directory where the individual BZ2 files are located: must end in "/".
		"""
		print( "------- Creating surface forms and links from Wikilinks and Disambiguation pages -------" )

		startTotalTime = time.time()
		directories = os.listdir( extractedDir )  		# Get directories of the form AA, AB, AC, etc.
		chunks = []
		MAX_CHUNK_SIZE = 20
		for directory in directories:
			fullDir = extractedDir + directory
			if os.path.isdir( fullDir ):
				print( "[*] Processing", directory )
				files = os.listdir( fullDir )  			# Get all files in current parsing directory, e.g. AA/wiki_00.bz2.
				files.sort()
				for file in files:
					fullFile = fullDir + "/" + file
					if os.path.isfile( fullFile ) and P.Parser._FilenamePattern.match( file ):		# Read bz2 file and process it.
						with bz2.open( fullFile, "rt", encoding="utf-8" ) as bz2File:
							documents = self._extractWikiPagesFromBZ2( bz2File.readlines(), keepDisambiguation=True, lowerCase=False )

						# Add documents to a chunk list in preparation for multiprocessing.
						chunks.append( documents )
						if len( chunks ) == MAX_CHUNK_SIZE:
							self._extractAndProcessWikilinks( chunks )		# Extract Wikilinks and update collections in DB.
							chunks = []

		if chunks:
			self._extractAndProcessWikilinks( chunks )						# Process remaining chunks of documents.

		endTotalTime = time.time()
		print( "[!] Completed process in", endTotalTime - startTotalTime, "secs" )


	def _extractAndProcessWikilinks( self, chunks ):
		"""
		Extract wikilinks from regular and disambiguation pages, and write results to ned_dictionary and ned_linking collections.
		:param chunks: A list of lists of extracted wiki documents of the form {id:int, title:str, lines:[str]}.
		"""
		startTime = time.time()
		pool = Pool()
		rChunks = pool.map( NEDParser._extractWikilinks, chunks )  	# Each chunk of document objects in its own thread.
		pool.close()
		pool.join()  												# Close pool and wait for work to finish.

		# Split rChunks' lists of tuples into surface form dicts and linking dicts.
		sfDocuments = []
		linkDocuments = []
		for chunk in rChunks:
			for doc in chunk:
				if doc[0]: sfDocuments.append( doc[0] )				# Surface forms.
				if doc[1]["to"]: linkDocuments.append( doc[1] )		# Link information.

		# Update DB collections.
		self._updateSurfaceFormsDictionary( sfDocuments )
		self._updateLinkingCollection( linkDocuments )

		endTime = time.time()
		print( "[**] Processed", len( chunks ), "chunks in", endTime - startTime, "secs" )


	def parseSFFromRedirectPages( self, msIndexFilePath, msDumpFilePath ):
		"""
		Extract surface form from redirect pages.
		We don't modify the ned_linking collection in this function since redirect pages are only aliases for entity pages.
		:param msIndexFilePath: Multistream index file path (e.g. enwiki-20141106-pages-articles-multistream-index.txt).
		:param msDumpFilePath: Multistream dump file path (e.g. enwiki-20141106-pages-articles-multistream.xml.bz2).
		"""
		print( "------- Creating surface forms from redirect pages -------" )

		startTotalTime = time.time()
		startTime = time.time()
		blockRequests = []													# Accumulate blocks for multithreading processing.
		REQUESTS_BLOCK_COUNT = 1000
		requestId = 0
		with open( msDumpFilePath, "rb" ) as bz2File:
			with open( msIndexFilePath, "r", encoding="utf-8" ) as indexFile:
				seekByte = -1
				for lineNumber, line in enumerate( indexFile ):				# Read index line by line.
					components = line.strip().split( ":" )					# [ByteStart, DocID, DocTitle]
					newSeekByte = int( components[0] )						# Find the next seek byte start that is different to current (defines a block).

					if seekByte == -1:										# First time reading seek byte from file.
						seekByte = newSeekByte
						continue

					if newSeekByte != seekByte:									# Changed seek-byte?
						requestId += 1
						count = newSeekByte - seekByte							# Number of bytes to read from bz2 stream.

						bz2File.seek( seekByte )								# Read block of data.
						block = bz2File.read( count )
						dData = bz2.decompress( block ).decode( "utf-8" )
						blockRequests.append( (requestId, dData) )				# Append new block to requests.

						if len( blockRequests ) == REQUESTS_BLOCK_COUNT:		# Accumulate REQUESTS_BLOCK_COUNT requests for incremental parsing.
							self._parseMSBZ2BlockRequests( blockRequests )
							print( "[*]", len( blockRequests ), "request blocks starting at byte", seekByte, "parsed after", time.time() - startTime, "secs" )
							blockRequests = []
							startTime = time.time()

						seekByte = newSeekByte

				# Add the last seek byte with count = -1 to read all until EOF.
				requestId += 1
				bz2File.seek( seekByte )  										# Read block of data.
				block = bz2File.read( -1 )
				dData = bz2.decompress( block ).decode( "utf-8" )
				blockRequests.append( (requestId, dData) )  					# Append new block to requests.

				self._parseMSBZ2BlockRequests( blockRequests )					# And parse this last block.
				print( "[*]", len( blockRequests ), "request blocks starting at byte", seekByte, "parsed after", time.time() - startTime, "secs" )

		endTotalTime = time.time()
		print( "[!] Completed process for redirect surface forms in", endTotalTime - startTotalTime, "secs" )


	def _parseMSBZ2BlockRequests( self, requests ):
		"""
		Read surface forms from blocks of Wikipedia documents obtained from the BZ2 multi-stream dump file.
		:param requests: A list of tuples of the form (requestID, blockStringData)
		"""
		pool = Pool()
		sfDocuments = pool.map( NEDParser._processMSBZ2Block, requests )  		# Each block request tuple in its own thread.
		pool.close()
		pool.join()

		# Update ned_dictionary collection.
		self._updateSurfaceFormsDictionary( sfDocuments )


	@staticmethod
	def _processMSBZ2Block( block ):
		"""
		Read and process data from a block in multi-stream Wikipedia dump file.
		:param block: A tuple containing (requestId, string block of data).
		:return: A dictionary of the form {"sf1":{"m.EID1":int,..., "m.EIDn":int}, "sf2":{"m.EID1":int,..., "m.EIDn":int}, ...}
		"""
		startTime = time.time()

		mClient = MongoClient( "mongodb://localhost:27017/" )  # One single connection to rule all requests in this block.
		mNED = mClient.ned
		mEntity_ID = mNED["entity_id"]

		requestId = block[0]
		dData = block[1]

		# Obtain the title and redirect titles from read block.
		lines = dData.split( "\n" )
		skipLine = False				# To avoid unnecessary checks.
		title = None
		nDoc = {}						# Output dictionary.
		for line in lines:
			line = line.strip()
			if not line:				# Skip empty lines.
				continue

			if line == "</page>":							# End of document?
				skipLine = False
				title = None
			elif not skipLine:
				if not title:								# Find page title first.
					m = NEDParser._TitlePattern.search( line )
					if m:
						title = m.group( 1 ).lower()  					# Now reading a <page>.  Make title a lowercase surface form.
				else:
					m = NEDParser._RedirectTitlePattern.search( line )	# Find redirect title.
					if m:
						entity = m.group( 1 )

						# Check against DB that the referenced real-world entity exists in entity_id collection.
						# Skip links to another disambiguation page or an invalid entity page.
						if P.Parser._DisambiguationPattern.match( entity ) is None and P.Parser._SkipTitlePattern.match( entity ) is None:

							record = None  								# Sentinel for found entity in DB.

							# First check how many entities match the lowercase version given in link: we may have ALGOL and Algol...
							n = mEntity_ID.find( { "e_l": entity.lower() } ).count()
							if n == 1:  								# One match?  Then retrieve entity ID.
								record = mEntity_ID.find_one( { "e_l": entity.lower() }, projection={ "_id": True } )
							elif n > 1:  								# If more than one record, then Wikilink must match the true entity name: case sensitive.
								record = mEntity_ID.find_one( { "e": entity }, projection={ "_id": True } )

							if record:  								# Process only those entities existing in entity_id collection.
								eId = "m." + str( record["_id"] )

								# Creating entry in output dict.
								if nDoc.get( title ) is None:
									nDoc[title] = {}
								if nDoc[title].get( eId ) is None:
									nDoc[title][eId] = 0
								nDoc[title][eId] += 1  				# Entity is referred to by this surface form one more time (i.e. increase count).
							# else:
							# 	print( "[!] Entity", entity, "doesn't exist in the DB!", file=sys.stderr )
						# else:
						# 	print( "[W] Skipping entry", title, "pointing to invalid entity", entity, file=sys.stderr )

						skipLine = True								# Found what we wanted... skip the rest of the page.
					elif line.find( "<text", 0, 6 ) != -1:			# Reached the body of the page?
						skipLine = True								# No need to continue checking for redirect until page ends in </page>.

		mClient.close()
		endTime = time.time()
		print( "[**] Processed", len( nDoc ), "redirect entries in block", requestId, "in", endTime - startTime )
		return nDoc


	def initDBCollections( self ):
		"""
		Reset the DB collections to start afresh.
		"""
		self._mNed_Dictionary.drop()			# Note that we don't drop the entity_id collection here: use the TFIDFParser for that.
		self._mNed_Linking.drop()

		print( "[!] Collections for surface forms computations have been dropped" )


	@staticmethod
	def _extractWikilinks( docs ):
		"""
		Parse inter-Wikilinks from an entity page or disambiguation page to obtain surface forms (by convention these will be lowercased).
		Also, collect the IDs of entities that current document points to, as long as current doc is a non-disambiguatio page.
		:param docs: List or chunk of document dictionaries to process: {id:int, title:str, lines:[str]}.
		:return: A list of tuples with two dicts: one of the form {"sf1":{"m.EID1":int,..., "m.EIDn":int}, "sf2":{"m.EID1":int,..., "m.EIDn":int}, ...}, \
				 and another of the form {from:int, to: set{int, int, ..., int}}.
		"""
		mClient = MongoClient( "mongodb://localhost:27017/" )		# One single connection to rule all docs' DB requests.
		mNED = mClient.ned
		mEntity_ID = mNED["entity_id"]

		result = []
		for doc in docs:
			nDoc = {}		# This dict stores the surface forms and their corresponding entity mappings with a reference count.
			nSet = set()	# Stores the IDs of pages pointed to by this non-disambiguation document (e.g. nSet is empty for a disambiguation page).

			# Treat disambiguation pages differently than regular valid pages.
			surfaceForm = ""
			isDisambiguation = False
			m = P.Parser._DisambiguationPattern.match( doc["title"] )
			if m is not None:
				isDisambiguation = True									# We'll be processing a disambiguation page.
				surfaceForm = m.group( 1 ).strip().lower()				# This will become the common surface name for all wikilinks within current disambiguation page.

			for line in doc["lines"]:
				line = P.Parser._ExternalLinkPattern.sub( r"\3", line )	# Remove external links to avoid false positives.
				for matchTuple in P.Parser._LinkPattern.findall( line ):
					entity = html.unescape( unquote( matchTuple[0] ) ).strip()	# Clean entity name: e.g. "B%20%26amp%3B%20W" -> "B &amp; W" -> "B & W".

					if not isDisambiguation:
						surfaceForm = matchTuple[1].lower()				# For non disambiguation pages, anchor text is the surface form.

					if len( surfaceForm ) > 128:						# Skip too long of a surface form.
						continue

					# Skip links to another disambiguation page or an invalid entity page.
					if P.Parser._DisambiguationPattern.match( entity ) is None and P.Parser._SkipTitlePattern.match( entity ) is None:

						record = None  									# Sentinel for found entity in DB.

						# First check how many entities match the lowercase version given in link: we may have ALGOL and Algol...
						n = mEntity_ID.find( { "e_l": entity.lower() } ).count()
						if n == 1:										# One match?  Then retrieve entity ID.
							record = mEntity_ID.find_one( { "e_l": entity.lower() }, projection={ "_id": True } )
						elif n > 1:										# If more than one record, then Wikilink must match the true entity name: case sensitive.
							record = mEntity_ID.find_one( { "e": entity }, projection={ "_id": True } )

						if record:										# Process only those entities existing in entity_id collection.
							eId = "m." + str( record["_id"] )

							# Creating entry in output dict.
							if nDoc.get( surfaceForm ) is None:
								nDoc[surfaceForm] = {}
							if nDoc[surfaceForm].get( eId ) is None:
								nDoc[surfaceForm][eId] = 0
							nDoc[surfaceForm][eId] += 1					# Entity is referred to by this surface form one more time (i.e. increase count).

							if not isDisambiguation:
								nSet.add( record["_id"] )				# Keep track of page IDs pointed to by this non-disambiguation document.
						else:
							print( "[!] Entity", entity, "doesn't exist in the DB!", file=sys.stderr )
					else:
						print( "[W] Skipping entry", surfaceForm, "pointing to invalid entity", entity, file=sys.stderr )

			# print( "[***]", doc["id"], doc["title"], "... Done!" )
			result.append( ( nDoc, { "from": doc["id"], "to": nSet } ) )

		mClient.close()
		return result


	def _updateSurfaceFormsDictionary( self, sfDocuments ):
		"""
		Update the NED dictionary of surface forms.
		:param sfDocuments: List of (possibly empty) surface form docs of the form {"sf1":{"m.EID1":int,..., "m.EIDn":int}, "sf2":{"m.EID1":int,..., "m.EIDn":int}, ...}.
		"""
		print( "[*] Updating ned_dictionary collection... ", end="" )

		requests = []  										# We'll use bulk writes to speed up process.
		BATCH_SIZE = 10000
		totalRequests = 0
		for sfDoc in sfDocuments:
			if not sfDoc: continue							# Skip empty sf dictionaries.

			for sf in sfDoc:								# Iterate over surface forms in current dict.
				requests.append( pymongo.UpdateOne( { "_id": sf }, { "$inc": sfDoc[sf] }, upsert=True ) )
				totalRequests += 1
				if len( requests ) == BATCH_SIZE:  			# Send lots of update requests.
					self._mNed_Dictionary.bulk_write( requests )
					requests = []

		if requests:
			self._mNed_Dictionary.bulk_write( requests )  	# Process remaining requests.
		print( "Done with", totalRequests, "requests sent!" )


	def _updateLinkingCollection( self, linkDocuments ):
		"""
		Add more link references to the ned_linking collection.
		:param linkDocuments: A list of dicts of the form {from:int, to: set{int, int, ..., int}}.
		"""
		print( "[*] Updating ned_linking collection... ", end="" )

		# Conform input to the following format: {"eId1":{"f.eId2":True, "f.eId3":True}, ..., "eIdn":{"f.eId1":True,...}}
		toFrom = {}
		for doc in linkDocuments:
			fId = "f." + str( doc["from"] )				# --> from 12 to "f.12".
			for to in doc["to"]:
				if toFrom.get( to )	is None:
					toFrom[to] = {}
				toFrom[to][fId] = True

		# Now, upsert ned_linking collection with UpdateOne bulk writes.
		requests = []
		BATCH_SIZE = 10000
		totalRequests = 0
		for to in toFrom:
			requests.append( pymongo.UpdateOne( { "_id": int( to ) }, { "$set": toFrom[to] }, upsert=True ) )
			totalRequests += 1
			if len( requests ) == BATCH_SIZE:			# Send lots of update requests at once.
				self._mNed_Linking.bulk_write( requests )
				requests = []

		if requests:
			self._mNed_Linking.bulk_write( requests )	# Process remaining requests.

		print( "Done with", totalRequests, "requests sent!" )
