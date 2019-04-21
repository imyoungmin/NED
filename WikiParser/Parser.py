import bz2
import os
import re
import sys
import time
from nltk.corpus import stopwords
import Tokenizer

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

# Stop words set: use nltk.download('stopwords').  Then add: "n't".
_stopWords = set( stopwords.words( "english" ) )

def buildTFIDFDictionary():

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
						_tokenizeWikiPagesFromBZ2( bz2File.readlines() )
					endTime = time.time()

					print( "[**] Done with", file, ":", endTime - startTime )


def _tokenizeWikiPagesFromBZ2( lines ):
	"""
	Tokenize non-disambiguation articles from extracted Wikipedia bz2 archives.
	:param lines: A list of sentences.
	:return: List of document objects: {id:int, title:str, tokens:{token1:freq1, token2:freq2, ...}}.
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
				else:
					print( "[***]", m.group(2), "...", end="" )
				doc = { "id": int(m.group(1)), 							# A new dictionary for this document.
					    "title": m.group( 2 ),
						"tokens": {}}									# Dictionary of curated/unique tokens with frequencies.
				extractingContents = True								# Turn on the flag: we started reading a document.
			else:
				print( "Line:", line, "is not in any document!", file=sys.stderr )
		else:
			if line == "</doc>":
				if not isDisambiguation:
					documents += [doc]  # Add extracted document to list for further processing in caller function.
					print( "Done!" )
				extractingContents = False
				isDisambiguation = False
			elif not isDisambiguation:									# Process text within <doc></doc> for non disambiguation pages
				for tag in _undesiredTags:								# Remove undesired tags.
					line = line.replace( tag, "" )

#				_ExternalLinkPattern.sub( r"\3", line )					# Replace external links for their anchor texts.  (No interwiki links affected).
				line = _LinkPattern.sub( r"\2", line )					# Replace all links with their anchor text.
				_tokenizeSentence( line, doc["tokens"] )				# Update dictionary of tokens and frequencies.

	return documents


def _tokenizeSentence( sentence, dictionary ):
	"""
	Curate and count frequencies of unique tokens.
	:param sentence: Line of text to tokenize and process.
	:param dictionary: Output dictionary (function update contents of existing data).
	"""
	tokens = Tokenizer.tokenize( sentence.lower() )  		# Tokenize a lower-cased version of read line.
	tokens = [w for w in tokens if not w in _stopWords ]  	# Remove stop words.

	for token in tokens:
		if _PunctuationOnlyPattern.match( token ) is None:	# Skip patterns like ... #
			if dictionary.get( token ) is None:
				dictionary[token] = 0						# Create token in dictionary if it doesn't exist.
			dictionary[token] += 1


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