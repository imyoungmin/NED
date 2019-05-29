from typing import Dict, Tuple, List
import sys
import re
import time
from multiprocessing import Pool
from . import NED as N
from WikiParser import Parser as P
import importlib

importlib.reload( N )
importlib.reload( P )


class DatasetDocument:
	"""
	Auxiliary class to process dataset documents in parallel.
	"""
	def __init__( self, docTitle: str, tokens: List[str], surfaceForms: Dict[str, List[Tuple[int, int]]], expectedMappings: Dict[str, int], startTime: float ):
		"""
		Constructor.
		:param docTitle: Document title.
		:param tokens: List of ordered tokens.
		:param surfaceForms: Surface forms detected with tuples for their ocurrences in the token list.
		:param expectedMappings: For each surface form the expected mapping entity ID.
		:param startTime: When we started processing this document.
		"""
		self.docTitle = docTitle
		self.tokens = tokens
		self.surfaceForms = surfaceForms
		self.expectedMappings = expectedMappings
		self.startTime = startTime


class Task:
	"""
	Execute a particular disambiguation task.
	"""


	_DocTitlePattern = re.compile( r"^-DOCSTART-\s+\((\w+)\s+.+?\)$", re.I )	# Dataset document title pattern.
	debug = True																# Debug NED mode (i.e. print messages).


	@staticmethod
	def evaluateAccuracy( datasetPath: str ):
		"""
		Evaluate accuracy of current model against the CoNLL 2003 dataset.
		:param datasetPath: The annotated dataset input file path.
		"""
		total = 0  					# Total number of "intended" surface forms.
		totalCorrect = 0  			# Total number of correctly mapped surface forms.
		totalNIL = 0  				# Total number of surface forms with no candidate mappings in the KB.
		totalDocs = 0

		totalStartTime = startTime = time.time()

		print( "**************************** Evaluating Accuracy ****************************" )

		with open( datasetPath, "r", encoding="utf-8" ) as file:
			documentHasContents = False
			tokens: List[str] = []								# Tokens of individual documents in the dataset.
			surfaceForms: Dict[str, List[Tuple[int, int]]] = {}	# Surface forms indices in tokens to be sent to NED object.
			expectedMappings: Dict[str, int] = {}				# Expected entity IDs for each (different) surface form.
			docTitle = ""
			sfTokensStart = -1									# Keep track of token delimiter indices for a surface form: [sfTokenStart, sfTokenEnd).
			tokensIndex = -1
			processingSF = ""									# Currently processing tokens for this surface form.

			MAX_CHUNK_COUNT = 100								# Accumulate documents to process in parallel in this list.
			chunks: List[DatasetDocument] = []

			for line in file.readlines():
				line = line.strip()
				if not line: continue

				if line.find( "-DOCSTART-" ) == 0:				# New document?
					if documentHasContents:						# First process previously loaded document if any.
						if sfTokensStart > 0:					# Check whether we were reading tokens of a "last" named entity.
							surfaceForms[processingSF].append( (sfTokensStart, tokensIndex) )

						# Add dataset document to chunks and check if we start processing the latter.
						chunks.append( DatasetDocument( docTitle, tokens, surfaceForms, expectedMappings, startTime ) )
						if len( chunks ) == MAX_CHUNK_COUNT:
							totals = Task._processDatasetChunks( chunks )
							total += totals[0]  				# Accumulate results.
							totalCorrect += totals[1]
							totalNIL += totals[2]
							totalDocs += totals[3]
							chunks = []

					elif docTitle:								# Skip error if this is the first document.
						print( "[!]", docTitle, "has no contents!", sys.stderr )

					documentHasContents = False
					tokens = []
					surfaceForms = {}
					expectedMappings = {}				# Reset structs to "make space" for new dataset document just coming up.
					docTitle = Task._DocTitlePattern.match( line ).group( 1 )
					tokensIndex = 0						# Ready to read first token.
					sfTokensStart = -1
					startTime = time.time()
					print( ">> Now processing document", docTitle, "..." )
				else:
					# Process doc's line content: one token per line, but possibly line is split in several columns.
					parts = line.split( "\t" )
					if len( parts ) >= 4:				# token \t (B|I) \t surfaceForm \t (WikiName|--NME--).
						if parts[3] != "--NME--":		# Skip unidentified entities.
							if parts[1] == "B":			# Beginning of entity mention?
								if sfTokensStart > 0:	# But we were reading another surface form?
									surfaceForms[processingSF].append( (sfTokensStart, tokensIndex) )	# Finish adding it before getting the new sf.

								sf = parts[2].lower()
								expectedMappings[sf] = int( parts[5] )		# Expected entity mapping ID.
								if surfaceForms.get( sf ) is None:
									surfaceForms[sf] = []					# Prepare to add new surface form list of tuple occurences.
								sfTokensStart = tokensIndex
								processingSF = sf
						elif sfTokensStart > 0:								# Finish adding surface form if any.
							surfaceForms[processingSF].append( (sfTokensStart, tokensIndex) )
							sfTokensStart = -1

					elif sfTokensStart > 0:				# Were we reading tokens for an entity mention?
						surfaceForms[processingSF].append( ( sfTokensStart, tokensIndex ) )
						sfTokensStart = -1				# Change state to now reading regular tokens.

					documentHasContents = True
					tokens.append( parts[0].lower().strip() )
					tokensIndex += 1					# Basically keeps track of all valid tokens in doc.

			if documentHasContents:  					# Process last loaded document if any.
				if sfTokensStart > 0:  					# Check whether we were reading tokens of a "last" named entity.
					surfaceForms[processingSF].append( (sfTokensStart, tokensIndex) )

				chunks.append( DatasetDocument( docTitle, tokens, surfaceForms, expectedMappings, startTime ) )
				totals = Task._processDatasetChunks( chunks )
				total += totals[0]  					# Accumulate results.
				totalCorrect += totals[1]
				totalNIL += totals[2]
				totalDocs += totals[3]

		# Present statistics.
		print( "\n------------------------------- Statistics ------------------------------" )
		print( "+ Documents:", totalDocs )
		print( "+ Surface forms:", total )
		print( "        Correct:", totalCorrect )
		print( "            NIL:", totalNIL )
		print( "+ Processing time: ", time.time() - totalStartTime, "secs." )


	@staticmethod
	def _processDatasetChunks( chunks: List[DatasetDocument] ) -> Tuple[int, int, int, int]:
		"""
		Process groups of dataset documents in parallel.
		:param chunks: Group of dataset documents.
		:return: Tuple containing (Total no. of surface forms, Total no. of correct mappings, Total no. of NIL, Total no. of effective docs).
		"""
		pool = Pool()
		stats = pool.map( Task._processDatasetDocument, chunks )  	# Each document object in its own thread.
		pool.close()
		pool.join()  												# Close pool and wait for work to finish.

		total = 0				# Surface forms intented.
		totalCorrect = 0		# Surface forms correctly matched.
		totalNIL = 0			# Surface forms with no candidates.
		totalDocs = 0			# Number of effective, nonempty dataset documents.
		for totals in stats:	# Accumulate results.
			if totals[0] - totals[2] > 0:		# There can be the case that a doc has X surface forms, but none is in the KB.
				total += totals[0]				# So we only count effective docs.
				totalCorrect += totals[1]
				totalNIL += totals[2]
				totalDocs += 1

		return total, totalCorrect, totalNIL, totalDocs


	@staticmethod
	def _processDatasetDocument( datasetDocument: DatasetDocument ) -> (int, int, int):
		"""
		Process and evaluate accuracy of an individual dataset document.
		:param datasetDocument: Document to process.
		:return: A tuple (Total no. of surface forms, Total no. of correct mappings, Total no. of NIL).
		"""
		# Extract inputs.
		tokens = datasetDocument.tokens
		surfaceForms = datasetDocument.surfaceForms
		expectedMappings = datasetDocument.expectedMappings

		ned = N.NED( Task.debug )  					# A NED object opens its own connection to the Mongo "ned" DB.
		results = ned.go( tokens, surfaceForms )	# Evaluation.

		# Comparison to expected entities.
		total = 0			# Total number of "intended" surface forms.
		totalCorrect = 0	# Total number of correctly mapped surface forms.
		totalNIL = 0		# Total number of surface forms with no candidate mappings in the KB.
		for sf, eId in expectedMappings.items():
			total += 1
			if results.get( sf ) is not None:
				if results[sf][0] == eId:
					totalCorrect += 1
			else:
				totalNIL += 1

		# Print with tabs for postprocessing stats: only docs for which there ARE entities to disambiguate.
		if total - totalNIL > 0:
			print( "<< Done with doc", datasetDocument.docTitle, "after", time.time() - datasetDocument.startTime, "secs.",
				   "Stats:\t", total, "\t", totalCorrect, "\t", totalNIL, "\t", totalCorrect / (total - totalNIL) )

		return total, totalCorrect, totalNIL


	@staticmethod
	def disambiguateTextFile( filePath: str ) -> Dict[str, Tuple[int, str]]:
		"""
		Disambiguate a 'unique document' text file.
		:param filePath: Path to file to disambiguate
		:return: Dictionary with surface forms and mapping entities
		"""
		print( "**************************** Disambiguating Named Entities in Text ****************************" )

		ned = N.NED( Task.debug )  								# NED object opens its own connection to the Mongo "ned" DB.

		with open( filePath, "r", encoding="utf-8" ) as file:
			text = file.read().lower()							# Important! Text has to be lowercased.

		### Extract named entities and tokenize text ###

		tokens: List[str] = []
		surfaceForms: Dict[str, List[Tuple[int, int]]] = {}		# Saves surface form and where in the tokens list it appears: [start, end).
		i = 0													# Start from beginning of text.
		s = text.find( "[[", i )
		while s != -1:
			e = text.find( "]]", s )
			if e == -1:											# Missing closing ]]?
				print( "[x] Missing ']]' to enclose a named entity.  Check the input text!", file=sys.stderr )
				return {}

			sf = text[s+2:e]									# The surface form.
			if i < s:											# Tokenize text before named entity.
				tokens += P.Parser.tokenizeText( text[i:s] )
			sfTokens = P.Parser.tokenizeText( sf )				# Surface form tokens.
			sfTokensStart = len( tokens )
			tokens += sfTokens
			sfTokensEnd = len( tokens )

			# Add named entity.
			if surfaceForms.get( sf ) is None:
				surfaceForms[sf] = []
			surfaceForms[sf].append( ( sfTokensStart, sfTokensEnd ) )

			i = e + 2
			s = text.find( "[[", i )

		# Tokenize rest of text.
		if i < len( text ):
			tokens += P.Parser.tokenizeText( text[i:] )

		results = ned.go( tokens, surfaceForms )

		print( "\n----------------------------- Final results -----------------------------" )
		for sf, rt in results.items():
			print( "*", sf, ": (", rt[0], ") ", rt[1] )
		print( "-------------------------------------------------------------------------" )

		ned.saveData()

		return results