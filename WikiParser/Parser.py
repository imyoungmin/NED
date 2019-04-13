import bz2

class Parser:
	_ROOT = "/Volumes/YoungMinEXT/"															# The root directory of the Wikipedia files.
	_Multistream_Index = _ROOT + "enwiki-20141106-pages-articles-multistream-index.txt"		# Use the multistream Wikipedia dump to save space.
	_Multistream_Dump = _ROOT + "enwiki-20141106-pages-articles-multistream.xml.bz2"

	def go( self ):
		"""
		Launch Wikipedia parsing process.
		:return:
		"""
		print( "[!] Started to parse Wikipedia files" )
		with open( self._Multistream_Index, "r", encoding="utf-8" ) as indexFile:
			seekByte = -1
			for lineNumber, line in enumerate( indexFile ):			# Read index line by line.
				components = line.strip().split( ":" )				# [ByteStart, DocID, DocTitle]
				newSeekByte = int( components[0] )					# Find the next seek byte start that is different to current (defines a block).

				if seekByte == -1:									# First time reading seek byte from file.
					seekByte = newSeekByte
					continue

				if newSeekByte != seekByte:							# Changed seek-byte?
					count = newSeekByte - seekByte					# Number of bytes to read from bz2 stream.
					self._processBZ2Block( seekByte, count )		# Read Wikipedia docs in this block.
					seekByte = newSeekByte
					break	# TODO: Remove to process all blocks.

			# TODO: Process the last seek byte count = -1.

		print( "[!] Finished parsing Wikipedia" )


	def _processBZ2Block( self, seekByte, count ):
		with open( self._Multistream_Dump, "rb" ) as bz2File:
			bz2File.seek( seekByte )
			block = bz2File.read( count )

			dData = bz2.decompress( block )
			print( dData )