import importlib
import re
import PorterStemmer as PS
from nltk.corpus import stopwords
from pymongo import MongoClient
from abc import ABCMeta, abstractmethod
import Tokenizer
importlib.reload( Tokenizer )

"""
Base class for parsing the Wikipedia dump files: either extracted by WikiExtractor or the multristream and corresponding
index files.
"""
class Parser(metaclass=ABCMeta):


	def __init__( self ):
		# List of URL protocols to detect them in regular expressions for links.
		self._UrlProtocols = [
			'bitcoin', 'ftp', 'ftps', 'geo', 'git', 'gopher', 'http', 'https', 'irc', 'ircs', 'magnet', 'mailto', 'mms',
			'news', 'nntp', 'redis', 'sftp', 'sip', 'sips', 'sms', 'ssh', 'svn', 'tel', 'telnet', 'urn', 'worldwind', 'xmpp'
		]

		# Regular expressions.
		self._FilenamePattern = re.compile( r"^wiki_\d+\.bz2$", re.I )  												# Checking only the right files.
		self._DocStartPattern = re.compile( r"^<doc.+?id=\"(\d+)\".+?title=\"\s*(.+)\s*\".*?>$", re.I )  				# Document head tag.
		self._DisambiguationPattern = re.compile( r"^(.+)\s+\(disambiguation\)$", re.I )  								# Disambiguation title pattern.
		self._SkipTitlePattern = re.compile( r"^(?:lists?\s+of|wikipedia:|template:|category:|file:)", re.I )  			# Title pattern to skip.
		self._ExternalLinkPattern = re.compile(
			r"<a.*?href=['\"]((" + r"|".join( self._UrlProtocols ) + r")(?:%3a|:)|//).*?>\s*(.*?)\s*</a>", re.I )  		# None Wikilinks.
		self._LinkPattern = re.compile( r"<a.*?href=\"\s*(.+?)\s*\".*?>\s*(.*?)\s*</a>", re.I ) 			 			# Links: internals and externals.
		self._UrlPattern = re.compile( r"(?:" + r"|".join( self._UrlProtocols ) + r")(?:%3a|:)/.*?", re.I )  			# URL pattern not inside a link.
		self._PunctuationOnlyPattern = re.compile( r"^\W+$" )

		# Stop words set: use nltk.download('stopwords').  Then add: "n't" and "'s".
		self._stopWords = set( stopwords.words( "english" ) )

		# Porter stemmer.
		self._porterStemmer = PS.PorterStemmer()

		# MongoDB connections.
		self._mClient = MongoClient( "mongodb://localhost:27017/" )
		self._mNED = self._mClient.ned
		self._mEntity_ID = self._mNED["entity_id"]  		# Common collection connector to hold entity titles and IDs: {_id:int, e:str}.


	@abstractmethod
	def initDBCollections( self ):
		"""
		Reset DB collections when necessary/requested.
		"""
		pass