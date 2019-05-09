import importlib
import re
from pymongo import MongoClient
import html
from abc import ABCMeta, abstractmethod
from bs4 import BeautifulSoup
import warnings
import sys
from typing import List
import Tokenizer
importlib.reload( Tokenizer )


# Supress warnings from BeautifulSoup module.
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


class Parser( metaclass=ABCMeta ):
	"""
	Base class for parsing the Wikipedia dump files: either extracted by WikiExtractor or the multristream and corresponding
	index files.
	"""

	# Static data members.
	_UrlProtocols = [										# List of URL protocols to detect them in regular expressions for links.
		'bitcoin', 'ftp', 'ftps', 'geo', 'git', 'gopher', 'http', 'https', 'irc', 'ircs', 'magnet', 'mailto', 'mms',
		'news', 'nntp', 'redis', 'sftp', 'sip', 'sips', 'sms', 'ssh', 'svn', 'tel', 'telnet', 'urn', 'worldwind', 'xmpp'
	]

	# Regular expressions.
	_UrlPattern = re.compile( r"(?:" + r"|".join( _UrlProtocols ) + r")(?:%3a|:)/.*?", re.I )  				# URL pattern not inside a link.
	_FilenamePattern = re.compile( r"^wiki_\d+\.bz2$", re.I )  												# Checking only the right files.
	_DocStartPattern = re.compile( r"^<doc.+?id=\"(\d+)\".+?title=\"\s*(.+)\s*\".*?>$", re.I )  			# Document head tag.
	_DisambiguationPattern = re.compile( r"^(.+)\s+\(disambiguation\)$", re.I )  							# Disambiguation title pattern.
	_SkipTitlePattern = re.compile( r"^(?:lists?\s+of|wikipedia:|template:|category:|file:)", re.I )  		# Title pattern to skip.
	_ExternalLinkPattern = re.compile(
		r"<a.*?href=['\"]((" + r"|".join( _UrlProtocols ) + r")(?:%3a|:)|//).*?>\s*(.*?)\s*</a>", re.I )  	# None Wikilinks.
	_LinkPattern = re.compile( r"<a.*?href=\"\s*(.+?)\s*\".*?>\s*(.*?)\s*</a>", re.I )  					# Links: internals and externals.
	_PunctuationOnlyPattern = re.compile( r"^\W+$" )														# Punctuation symbols only.


	def __init__( self ):
		# MongoDB connections.
		self._mClient = MongoClient( "mongodb://localhost:27017/" )
		self._mNED = self._mClient.ned
		self._mEntity_ID = self._mNED["entity_id"]  		# Common collection connector to hold entity titles, their lower-case version, and IDs: {_id:int, e:str, e_l:str}.


	def _extractWikiPagesFromBZ2( self, lines, keepDisambiguation=False, lowerCase=True ):
		"""
		Extract valid articles from extracted Wikipedia bz2 archives.
		:param lines: A list of sentences.
		:param keepDisambiguation: Do not discard disambiguation Wiki pages.
		:param lowerCase: Transform lines in output documents to lowercase.
		:return: List of document objects: {id:int, title:str, lines:[str]}.
		"""
		documents = []												# We collect all documents in a list.
		doc = {}													# A processed document is a dictionary: {id: x, title: y}

		extractingContents = False									# On/off when we enter the body of a document.
		isToSkip = False
		firstLine = False											# Skip first line since it's a repetition of the title.
		for line in lines:
			line = line.strip()
			if line == "": continue									# Skip empty lines.
			if not extractingContents:								# Wait for the sentinel: <doc ...>
				m = self._DocStartPattern.match( line )
				if m:														# Start of document?
					title = html.unescape( m.group( 2 ) )					# Title without html entities.

					# Skipping undesired pages.
					if ( not keepDisambiguation and self._DisambiguationPattern.match( title ) is not None ) \
							or	self._SkipTitlePattern.match( title ) is not None:
						isToSkip = True

					doc = { "id": int(m.group(1)), 							# A new dictionary for this document.
							"title": title,
							"lines": [] }
					extractingContents = True								# Turn on the flag: we started reading a document.
					firstLine = True										# Will be reading title repetion next in the first line.
				else:
					print( "Line:", line, "is not in any document!", file=sys.stderr )
			else:
				if line == "</doc>":
					if not isToSkip:
						documents += [doc]  								# Add extracted document to list for further processing in caller function.
					extractingContents = False
					isToSkip = False
				elif not isToSkip:											# Process text within <doc></doc> for non-skipped pages.
					if firstLine:
						firstLine = False
					else:
						doc["lines"].append( line.lower() if lowerCase else line )		# Add (lowercased?) line of text to output document.

		return documents


	@abstractmethod
	def initDBCollections( self ):
		"""
		Reset DB collections when necessary/requested.
		"""
		pass


	@staticmethod
	def tokenizeText( text: str ) -> List[str]:
		"""
		Tokenize text. This method provide uniformity in tokenization in both an external caller and the Wikipedia parser.
		:param text: String to tokenize.
		:return: List of tokens.
		"""
		tokens = Tokenizer.tokenize( text )				# Preferably, tokenize a lower-cased version of text.
		return Parser.curateTokens( tokens )			# Curate tokens.


	@staticmethod
	def curateTokens( tkns: List[str] ) -> List[str]:
		"""
		Verify tokens comply with some constraints.
		We are leaving stop words, punctuation-pattern words, and we are not stemming tokens.
		:param tkns: List of tokens.
		:return: Curated list of tokens.
		"""
		result: List[str] = []
		for token in tkns:
			if len( token ) <= 128:  												# Skip too long tokens.
				if Parser._UrlPattern.search( token ) is None:  					# Skip URLs.
					result.append( token )

		return result


	@staticmethod
	def tokenizeDoc( doc ):
		"""
		Tokenize a document object.
		:param doc: Document dictionary to process: {id:int, title:str, lines:[str]}.
		:return: A dictionary of the form {id:int, title:str, tokens:{token1:freq1, ..., tokenn:freqn}}.
		"""
		nDoc = { "id": doc["id"], "title": doc["title"], "tokens": { } }

		maxFreq = 0
		for line in doc["lines"]:
			soup = BeautifulSoup( line, "html.parser" )  				# Get lines with just text: no <tags/>.
			line = soup.getText().strip()
			if not line: continue

			tokens = Parser.tokenizeText( line )  						# Tokenize a lower-cased version of article text.

			for t in tokens:
				if nDoc["tokens"].get( t ) is None:
					nDoc["tokens"][t] = 1  								# Create token in dictionary if it doesn't exist.
				else:
					nDoc["tokens"][t] += 1
				maxFreq = max( nDoc["tokens"][t], maxFreq )  			# Keep track of maximum term frequency within document.

		if maxFreq == 0:
			print( "[W] Empty document:", nDoc, file=sys.stderr )
			return { }

		# print( "[***]", nDoc["id"], nDoc["title"], "... Done!" )
		return nDoc