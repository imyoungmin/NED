import pymongo
from typing import Set, Dict, Tuple, List
import sys
import numpy as np
from WikiParser import Parser as P
import importlib

importlib.reload( P )


class Entity:
	"""
	Implementation of an entity.  We want to have at most 1 object per entity across surface forms' candidates.
	"""
	def __init__( self, eId: int, name: str, pointedToBy: Set[int], initialEmbedding: np.ndarray ):
		"""
		Constructor.
		:param eId: Entity ID.
		:param name: Entity name as it appears in its Wikipedia article.
		:param pointedToBy: Set of Wikipedia articles (i.e. entity IDs) pointing to this entity.
		:param initialEmbedding: Initial (unnormalized) document embedding vector calculated from entry in sif_documents.
		"""
		self.id = eId
		self.name = name
		self.pointedToBy = pointedToBy
		self.v = np.array( initialEmbedding )


class Candidate:
	"""
	Candidate mapping implementation.
	"""
	def __init__( self, eId: int, count: int ):
		"""
		Constructor.
		:param eId: Entity ID.
		:param count: Number of entities being referred to by (outter/caller) surface form.
		"""
		self.id = eId
		self.count = count
		self.priorProbability = 0.0		# To be updated when collecting candidates for a surface form.
		self.contextSimilarity = 0.0	# TODO: To be updated with the homonym function.
		self.topicalCoherence = 0.0		# To be updated during the iterative substitution algorithm.


class SurfaceForm:
	"""
	Named entity object.
	"""
	def __init__( self, candidates: Dict[int, Candidate], initialEmbedding: np.ndarray ):
		"""
		Constructor.
		:param candidates: Map of candidate mapping entities.
		:param initialEmbedding: Initial (unnormalized) document embedding vector calculated from context words.
		"""
		self.v = np.array( initialEmbedding )
		self.candidates = candidates
		self.mappingEntityId = 0						# Final entity mapping.


class Word:
	"""
	Struct for words.
	"""
	def __init__( self, embedding: List[float], p: float ):
		"""
		Constructor
		:param embedding: Word embedding as 300-D vector.
		:param p: Word probability across all of corpus vocabulary = word frequency divided by total word freq count.
		"""
		self.v = np.array( embedding )
		self.p = p


class NED:
	"""
	Implementation of Graph-based named entity disambiguation class.
	"""

	def __init__( self ):
		"""
		Constructor.
		"""
		# MongoDB connections.
		self._mClient: pymongo.mongo_client = pymongo.MongoClient( "mongodb://localhost:27017/" )
		self._mNED = self._mClient["ned"]
		self._mEntity_ID: pymongo.collection = self._mNED["entity_id"]  			# {_id:int, e:str, e_l:str}.

		# Connections to SIF collections.
		self._mWord_Embeddings: pymongo.collection = self._mNED["word_embeddings"] 	# {_id:str, e:List[float], f:int}
		self._mSif_Documents: pymongo.collection = self._mNED["sif_documents"]		# {_id:int, w:List[str], f:List[int]}

		# Defining connections to collections for entity disambiguation.
		self._mNed_Dictionary: pymongo.collection = self._mNED["ned_dictionary"]  	# {_id:str, m:{"e_1":int, "e_2":int,..., "e_n":int}}. -- m stands for "mapping".
		self._mNed_Linking: pymongo.collection = self._mNED["ned_linking"]  		# {_id:int, f:{"e_1":true, "e_2":true,..., "e_3":true}}. -- f stands for "from".

		# Retrieve total number of entities recorded in DB.
		self._WP = self._mEntity_ID.count()
		self._LOG_WP = np.log( self._WP )											# Log used in topic relatedness metric.
		print( "NED initialized with", self._WP, "entities" )

		# Read total word frequencies and initialize map of word objects.
		with open( "Datasets/wordcount.txt", "r", encoding="utf-8" ) as fIn:
			self._TOTAL_WORD_FREQ_COUNT = float( fIn.read() )
		print( "NED initialized with", self._TOTAL_WORD_FREQ_COUNT, "total word frequency count" )
		self._wordMap: Dict[str, Word] = {}

		self._a = 1e-3																# Parameter 'a' for SIF.

		# Initialize map of entities (as a cache).
		self._entityMap: Dict[int, Entity] = {}

		# Named entities map {"namedEntity1": NamedEntity1, "namedEntity2": NamedEntity2, ...}.
		self._surfaceForms: Dict[str, SurfaceForm] = {}
		self._WINDOW_SIZE = 10

		# Initial score constants.
		self._alpha = 0.25
		self._beta = 0.25
		self._gamma = 0.5


	def go( self, filePath: str ) -> Dict[str, Tuple[int, str]]:
		"""
		Disambiguate indentified entities given in input text file.
		:param filePath: File containing text with named entities enclosed in [[.]].
		:return: Dictionary with surface forms and mapping entities
		"""
		with open( filePath, "r", encoding="utf-8" ) as file:
			text = file.read().lower()

		### Extract named entities and tokenize text ###

		tokens: List[str] = []
		surfaceForms: Dict[str, List[Tuple[int, int]]] = {}		# Saves surface form and where in the tokens list it appears: [start, end).
		i = 0								# Start from beginning of text.
		s = text.find( "[[", i )
		while s != -1:
			e = text.find( "]]", s )
			if e == -1:						# Missing closing ]]?
				print( "[x] Missing ']]' to enclose a named entity.  Check the input text!", file=sys.stderr )
				sys.exit( 1 )

			sf = text[s+2:e]				# The surface form.
			if i < s:						# Tokenize text before named entity.
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

		# For each surface form compute an initial, unnormalized 'document' embedding.
		# Also collect the candidate mapping entities.
		# If a surface form doesn't have any candidates, skip it.
		for sf in surfaceForms:
			words: Set[str] = set()
			for occurrence in surfaceForms[sf]:
				start = max( 0, occurrence[0] - self._WINDOW_SIZE )				# Collect tokens around occurrence.
				end = min( occurrence[1] + self._WINDOW_SIZE, len( tokens ) )
				for i in range( start, end ):
					words.add( tokens[i] )

			# Context has at least the surface form's tokens themselves; now, get the unnormalized initial 'document' embedding.
			# It'll also check for surface form tokens to exist in word_embeddings collection.
			v = self._getRawDocumentEmbedding( words )
			if np.count_nonzero( v ):
				candidates = self._getCandidatesForNamedEntity( sf )
				if candidates:
					self._surfaceForms[sf] = SurfaceForm( candidates, v )
				else:
					print( "[W] Surface form [", sf, "] doesn't have any candidate mapping entity!  Will be skipped...", sys.stderr )
			else:
				print( "[W] Surface form [", sf, "] doesn't have a valid document embedding!  Will be skipped...", sys.stderr )


		return {}


	def _getCandidatesForNamedEntity( self, m_i: str ) -> Dict[int, Candidate]:
		"""
		Retrieve candidate mapping entities for given named entity.
		Calculate the prior probability at the same time.
		:param m_i: Entity mention (a.k.a surface form) in lowercase.
		:return: A dict {e_1_id:Candidate_1, e_2_id:Candidate_2,...}.
		"""
		result = {}
		record1 = self._mNed_Dictionary.find_one( { "_id": m_i }, projection={ "m": True } )
		if record1:
			total = 0											# Accumulate reference count for this surface form by the candidate mappings.
			for r_j in record1["m"]:
				r = int( r_j )

				# Check the cache for entity.
				if self._entityMap.get( r ) is None:
					record2 = self._mEntity_ID.find_one( { "_id": r }, projection={ "e": True } )		# Consult DB to retrieve information for new entity into cache.
					record3 = self._mSif_Documents.find_one( { "_id": r }, projection={ "w": True } )	# Extract words in entity document.
					vd = self._getRawDocumentEmbedding( set( record3["w"] ) )							# Get an initial unnormalized document embedding.
					self._entityMap[r] = Entity( r, record2["e"], self._getPagesLikingTo( r ), vd )		# And retrieve other pages linking to new entity.

				result[r] = Candidate( r, record1["m"][r_j] )	# Candidate has a reference ID to the entity object.
				total += record1["m"][r_j]

			# Update prior probability.
			for r in result:
				result[r].priorProbability = result[r].count / total

		print( "[*] Collected", len( result ), "candidate entities for [", m_i, "]" )
		return result			# Empty if no candidates were found for given entity mention.


	def _getRawDocumentEmbedding( self, d: Set[str] ) -> np.ndarray:
		"""
		Compute an unnormalized document embedding from input words.
		Retrieve and cache word embeddings at the same time.
		:param d: Bag of words (e.g. document) -- must be lowercased!
		:return: \sum_{w \in d}( \frac{a}{a + p(w)}v_w )
		"""
		vd = np.zeros( 300 )
		for w in d:
			if self._wordMap.get( w ) is None:			# Not in cache?
				r = self._mWord_Embeddings.find_one( { "_id": w } )
				if r is None:
					continue								# Skip words not in the vocabulary.

				vw = r["e"]									# Word embedding and probability.
				p = r["f"] / self._TOTAL_WORD_FREQ_COUNT
				self._wordMap[w] = Word( vw, p )			# Cache word object.

			vd += self._a / ( self._a + self._wordMap[w].p ) * self._wordMap[w].v

		return vd			# Still need to normalize by total number of documents and subtracting proj onto first singular vector.


	def _getPagesLikingTo( self, e: int ) -> Set[int]:
		"""
		Collect IDs of pages linking to entity e.
		:param e: Target entity.
		:return: Set of entity IDs.
		"""
		record = self._mNed_Linking.find_one( { "_id": e }, projection={ "f": True } )
		U = set()
		for u in record["f"]:
			U.add( int( u ) )
		return U


	def topicalRelatedness( self, u1: int, u2: int ) -> float:
		"""
		Calculate the Wikipedia topical relatedness between two entities.
		:param u1: First entity ID.
		:param u2: Second entity ID.
		:return: 1 - \frac{log(max(|U_1|, |U_2|)) - log(|U_1 \intersect U_2|)}{log|WP| - log(min(|U_1|, |U_2|))}
		"""
		lU1 = len( self._entityMap[u1].pointedToBy )
		lU2 = len( self._entityMap[u2].pointedToBy )
		lIntersection = len( self._entityMap[u1].pointedToBy.intersection( self._entityMap[u2].pointedToBy ) )
		if lIntersection > 0:
			return 1.0 - ( np.log( max( lU1, lU2 ) ) - np.log( lIntersection ) ) / ( self._LOG_WP - np.log( min( lU1, lU2 ) ) )
		else:
			return 0.0


	def contextSimilarity( self, sf: str, eId: int ) -> float:
		similarity = 0.0

		return similarity


	def topicalCoherence( self, surfaceForm: str, M: Dict[str, int] ) -> float:
		"""
		Compute topical coherence of mapping entity for given surface form with respect to mapping entities of the rest.
		:param surfaceForm: Central surface form whose (candidate) mapping entity we task as reference.
		:param M: Mappings for all surface forms (including the central one).
		:return: Coh(r_{i,j}) = \frac{1}{|M| - 1} \sum_{c=1, c\nej}^{|M|}TR(r_{i,j}, e_{i,c}).
		"""
		totalTR = 0
		for sf in M:
			if sf != surfaceForm:
				totalTR += self.topicalRelatedness( M[surfaceForm], M[sf] )
		return  totalTR / ( len( M ) - 1 )


	def totalInitialScore( self, M: Dict[str, int] ) -> float:
		"""
		Calculate the total initial score for a given set of mapping entities.
		This function is used in the iterative substitution algorithm to evaluate different candidates performance.
		:param M: Mapping entities to each surface form.
		:return: \sum_i^{|M|} ( \alpha * Pp(e_i) + \beta * Sim(e_i) + \gamma * Coh(e_i) )
		"""
		totalScore = 0
		for sf in M:
			e = self._surfaceForms[sf].candidates[M[sf]]
			totalScore += self._alpha * e.priorProbability \
						  + self._beta * e.contextSimilarity \
						  + self._gamma * self.topicalCoherence( sf, M )
		return totalScore


	def iterativeSubstitutionAlgorithm( self ):
		"""
		Solve for the best entity mappings for all named entities in order to have an initial score.
		"""
		# Pick the candidate with maximum prior probability as first approximation to mapping entity.
		for ne in self._surfaceForms:
			mostPopularProb = 0
			for cm in self._surfaceForms[ne].candidates:
				if self._surfaceForms[ne].candidates[cm].priorProbability > mostPopularProb:
					self._surfaceForms[ne].mappingEntityId = cm

		iteration = 1
		M: Dict[str: int] = { ne: self._surfaceForms[ne].mappingEntityId for ne in self._surfaceForms }	# Surface forms and respective mapping entities.
		bestTIS = self.totalInitialScore( M )

		print( "[ISA][", iteration,"] Initial score starts at", bestTIS )

		while True:
			# Find the candidate substition that increases the most the total initial score among all candidates and all surface forms.
			foundBetterScore = False
			bestNE = 0		# Best surface form and candidate mapping.
			bestCM = 0
			for ne in self._surfaceForms:
				M = { ne: self._surfaceForms[ne].mappingEntityId for ne in self._surfaceForms }  # Try-out mapping entities.
				for cm in self._surfaceForms[ne].candidates:
					if cm != self._surfaceForms[ne].mappingEntityId:		# Skip currently selected best candidate mapping.
						M[ne] = cm											# Check this candidate substitution.

						tis = self.totalInitialScore( M )
						if tis > bestTIS:									# Is it improving?
							foundBetterScore = True
							bestNE = ne
							bestCM = cm
							bestTIS = tis

			if foundBetterScore > 0:										# Did score ever increase?
				self._surfaceForms[bestNE].mappingEntityId = bestCM			# New best mapping.
				print( "[ISA][", iteration ,"] Score increased to", bestTIS )
				iteration += 1
			else:
				break						# Stop when we detect a negative performance.

		print( "[ISA] Finalized greedy optimization with an initial score of", bestTIS )