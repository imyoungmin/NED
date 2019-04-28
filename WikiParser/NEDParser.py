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
from urllib.parse import unquote
import html
from . import Parser as P
import Tokenizer
importlib.reload( P )
importlib.reload( Tokenizer )

"""
Parsing Wikipedia extracted and multistream archives to construct the surface forms dictionary and the inter-links table.
"""
class NEDParser( P.Parser ):


	def __init__(self):
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
		for t in self._mEntity_ID.find():
			requests.append( pymongo.UpdateOne( { "_id": t["e"].lower() }, { "$inc": { "m." + str( t["_id"] ): +1 } }, upsert=True ) )
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


	def parseSFFromDisambiguationPages( self, extractedDir ):
		"""
		Grab the surface forms from disambiguation pages.
		:param: extractedDir: Directory where the individual BZ2 files are located: must end in "/".
		"""
		pass	# TODO: Complete body.


	def parseSFFromWikilinks( self, extractedDir ):
		"""
		Grab surface forms from wikilinks in valid entity pages.
		:param extractedDir: Directory where the individual BZ2 files are located: must end in "/".
		:return:
		"""
		pass	# TODO: Complete body.


	def parseSFFromRedirectPages( self, msIndexFilePath, musDumpFilePath ):
		"""

		:param msIndexFilePath:
		:param musDumpFilePath:
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