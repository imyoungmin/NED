import importlib
import bz2
import os
import sys
import time
from multiprocessing import Pool
import pymongo
from urllib.parse import unquote
import html
from . import Parser as P
import Tokenizer
importlib.reload( P )
importlib.reload( Tokenizer )


class NEDParser( P.Parser ):
	"""
	Parsing Wikipedia extracted and multistream archives to construct the surface forms dictionary and the inter-links table.
	"""


	def __init__( self ):
		"""
		Constructor.
		"""
		P.Parser.__init__( self )

		# Defining connections to collections for entity disambiguation.
		self._mNed_Dictionary = self._mNED["ned_dictionary"]			# {_id:str, m:{"e_1":int, "e_2":int,..., "e_n":int}}.
		self._mNed_Linking = self._mNED["ned_linking"]					# {_id:int, from:{"e_1":true, "e_2":true,..., "e_3":true}}.


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


	def parseSFFromWikilinks( self, extractedDir ):
		"""
		Grab surface forms from wikilinks in valid entity pages.
		Skip processing disambiguation pages, lists, and Wikipedia templates, files, etc.
		Use this method for incremental analysis of extracted Wikipedia BZ2 files.
		:param extractedDir: Directory where the individual BZ2 files are located: must end in "/".
		"""
		print( "------- Creating surface forms from Wikilinks and Disambiguation pages -------" )

		nDocs = 0
		startTotalTime = time.time()
		directories = os.listdir( extractedDir )  		# Get directories of the form AA, AB, AC, etc.
		for directory in directories:
			fullDir = extractedDir + directory
			if os.path.isdir( fullDir ):
				print( "[*] Processing", directory )
				files = os.listdir( fullDir )  			# Get all files in current parsing directory, e.g. AA/wiki_00.bz2.
				files.sort()
				for file in files:
					fullFile = fullDir + "/" + file
					if os.path.isfile( fullFile ) and P.Parser._FilenamePattern.match( file ):		# Read bz2 file and process it.
						startTime = time.time()
						with bz2.open( fullFile, "rt", encoding="utf-8" ) as bz2File:
							documents = self._extractWikiPagesFromBZ2( bz2File.readlines(), keepDisambiguation=True, lowerCase=False )

						# Process documents which contain regular entity pages and disambiguation pages.
						sfDocuments = []
						for doc in documents:
							sfDocuments.append( self._extractWikilinks( doc ) )

						# Update DB collections.
#						nDocs += self._updateSurfaceFormsDictionary( sfDocuments )

						endTime = time.time()
						print( "[**] Done with", file, ":", endTime - startTime )

		endTotalTime = time.time()
		print( "[!] Total number of documents:", nDocs, "in", endTotalTime - startTotalTime, "secs" )


	def parseSFFromRedirectPages( self, msIndexFilePath, msDumpFilePath ):
		"""

		:param msIndexFilePath:
		:param msDumpFilePath:
		:return:
		"""
		pass	# TODO: Complete body.


	def initDBCollections( self ):
		"""
		Reset the DB collections to start afresh.
		"""
		self._mNed_Dictionary.drop()			# Note that we don't drop the entity_id collection here: use the TFIDFParser for that.
		self._mNed_Linking.drop()

		print( "[!] Collections for surface forms computations have been dropped" )


	def _extractWikilinks( self, doc ):
		"""
		Parse inter-Wikilinks from an entity page or disambiguation page to obtain surface forms (by convention these will be lowercased).
		:param doc: Document dictionary to process: {id:int, title:str, lines:[str]}.
		:return: A dictionary of the form {"sf1":{"m.EID1":int,..., "m.EIDn":int}, "sf2":{"m.EID1":int,..., "m.EIDn":int}, ...}.
		"""
		nDoc = {}		# This dict stores the surface forms and their corresponding entity mappings with a reference count.


		# Treat disambiguation pages differently than regular valid pages.
		surfaceForm = ""
		isDisambiguation = False
		m = self._DisambiguationPattern.match( doc["title"] )
		if m is not None:
			isDisambiguation = True									# We'll be processing a disambiguation page.
			surfaceForm = m.group( 1 ).strip().lower()				# This will become the common surface name for all wikilinks within current disambiguation page.

		for line in doc["lines"]:
			line = self._ExternalLinkPattern.sub( r"\3", line )		# Remove external links to avoid false positives.
			for matchTuple in self._LinkPattern.findall( line ):
				entity = html.unescape( unquote( matchTuple[0] ) ).strip()	# Clean entity name: e.g. "B%20%26amp%3B%20W" -> "B &amp; W" -> "B & W".

				if not isDisambiguation:
					surfaceForm = matchTuple[1].lower()				# For non disambiguation pages, anchor text is the surface form.

				if len( surfaceForm ) > 128:						# Skip too long of a surface form.
					continue

				# Skip links to another disambiguation page or an invalid entity page.
				if self._DisambiguationPattern.match( entity ) is None and self._SkipTitlePattern.match( entity ) is None:

					record = None  									# Sentinel for found entity in DB.

					# First check how many entities match the lowercase version given in link: we may have ALGOL and Algol...
					n = self._mEntity_ID.find( { "e_l": entity.lower() } ).count()
					if n == 1:										# One match?  Then retrieve entity ID.
						record = self._mEntity_ID.find_one( { "e_l": entity.lower() }, projection={ "_id": True } )
					elif n > 1:										# If more than one record, then Wikilink must match the true entity name: case sensitive.
						record = self._mEntity_ID.find_one( { "e": entity }, projection={ "_id": True } )

					if record:													# Process only those entities existing in entity_id collection.
						eId = "m." + str( record["_id"] )

						# Creating entry in output dict.
						if nDoc.get( surfaceForm ) is None:
							nDoc[surfaceForm] = {}
						if nDoc[surfaceForm].get( eId ) is None:
							nDoc[surfaceForm][eId] = 0
						nDoc[surfaceForm][eId] += 1								# Entity is referred to by this surface form one more time (i.e. increase count).
					# else:
					# 	print( "[!] Entity", entity, "doesn't exist in the DB!", file=sys.stderr )

		print( "[***]", doc["id"], doc["title"], "... Done!" )
		return nDoc


	def _updateSurfaceFormsDictionary( self, sfDocuments ):
		"""
		Update the NED dictionary of surface forms.
		:param sfDocuments: List of (possibly empty) surface form docs of the form {"sf1":{"m.EID1":int,..., "m.EIDn":int}, "sf2":{"m.EID1":int,..., "m.EIDn":int}, ...}.
		:return: Number of non-empty surface form documents processed.
		"""
		requests = []  										# We'll use bulk writes to speed up process.
		BATCH_SIZE = 200
		totalRequests = 0
		for sfDoc in sfDocuments:
			if not sfDoc: continue							# Skip empty sf dictionaries.

			for sf in sfDoc:								# Iterate over surface forms in current dict.
				requests.append( pymongo.UpdateOne( { "_id": sf }, { "$inc": sfDoc[sf] }, upsert=True ) )
				totalRequests += 1
				if len( requests ) == BATCH_SIZE:  			# Send lots of update requests.
					self._mNed_Dictionary.bulk_write( requests )
					print( "[*]", totalRequests, "processed" )
					requests = []

		if requests:
			self._mNed_Dictionary.bulk_write( requests )  	# Process remaining requests.
			print( "[*]", totalRequests, "processed" )

		return totalRequests


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