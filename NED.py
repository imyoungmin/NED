import pymongo
from typing import Set, Dict, Tuple, List
import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
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
		:param initialEmbedding: Initial (without common component removed) document embedding vector calculated from entry in sif_documents.
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
		self.contextSimilarity = 0.0	# To be updated with the homonym function.
		self.topicalCoherence = 0.0		# Coh(r_{i,j}) = \frac{1}{|M|-1} \sum_{c=1, c \neq i}^{|M|}TR(r_{i,j}, e_c).
		self.initialScore = 0.0			# p_{i,j} = \alpha * Pp(r_{i,j}) + \beta * Sim(r_{i,j}) + \gamma * Coh(r_{i,j}).
		self.finalScore = 0.0			# To be updated in the propagation algorithm.


class SurfaceForm:
	"""
	Named entity object.
	"""
	def __init__( self, candidates: Dict[int, Candidate], initialEmbedding: np.ndarray ):
		"""
		Constructor.
		:param candidates: Map of candidate mapping entities.
		:param initialEmbedding: Initial (without common component removed) document embedding vector calculated from context words.
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

		self._a = 0.001																# Parameter 'a' for SIF.

		# Initialize map of entities (as a cache).
		self._entityMap: Dict[int, Entity] = {}

		# Named entities map {"namedEntity1": NamedEntity1, "namedEntity2": NamedEntity2, ...}.
		self._surfaceForms: Dict[str, SurfaceForm] = {}
		self._WINDOW_SIZE = 20

		# Initial score constants.
		self._alpha = 0.1
		self._beta = 0.6
		self._gamma = 0.3

		# Propagation algorithm constant.
		self._lambda = 0.3

		# Map of array index to (surface form, candidate mapping entity ID).
		self._indexToSFC: List[Tuple[str, int]] = []


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

		# For each surface form compute an initial 'document' embedding (without common component removed).
		# Also collect the candidate mapping entities.
		# If a surface form doesn't have any candidates, skip it.
		print( "[*] Now collecting candidate mapping entities for surface forms in input text:" )
		for sf in surfaceForms:
			words: Set[str] = set()
			for occurrence in surfaceForms[sf]:
				start = max( 0, occurrence[0] - self._WINDOW_SIZE )							# Collect tokens around occurrence.
				end = min( occurrence[1] + self._WINDOW_SIZE, len( tokens ) )
				for i in range( start, end ):
					words.add( tokens[i] )

			# Context has at least the surface form's tokens themselves; now, get the initial 'document' embedding.
			# It'll also check for surface form tokens to exist in word_embeddings collection.
			v = self._getRawDocumentEmbedding( list( words ), [1 for _ in range( len( words ) )] )	# Send tokens with frequency 1.
			if np.count_nonzero( v ):
				candidates = self._getCandidatesForNamedEntity( sf )
				if candidates:
					self._surfaceForms[sf] = SurfaceForm( candidates, v )
				else:
					print( "[W] Surface form [", sf, "] doesn't have any candidate mapping entity!  Will be skipped...", sys.stderr )
			else:
				print( "[W] Surface form [", sf, "] doesn't have a valid document embedding!  Will be skipped...", sys.stderr )
		print( "... Done!" )

		### Getting ready for disambiguation ###

		result: Dict[str, Tuple[int, str]] = {}		# Result mapping for each named entity/surface form.

		if self._surfaceForms:
			self._removeCommonDiscourseEmbedding()	# Remove projection onto first singular vector (i.e. common discourse vector).
			self._computeContextSimilarity() 		# Compute context similary of surface forms' BOW with respect to candidates mapping entities.

			# Topical coherence depends on having more than 1 sf.
			if len( self._surfaceForms ) > 1:
				# Get an initial score for candidate mapping entities using iterative substitution.
				# Then, apply the page rank algorithm to calculate final candidate mapping entity scores.
				self._iterativeSubstitutionAlgorithm()
				self._propagationAlgorithm()
			else:
				print( "[*] There's only one surface form.  Topical coherence will be ignored!" )
				self._chooseBestCandidate_NoTopicalCoherence()
				print( "... Done!" )

			# Place results in return object.
			for sf, sfObj in self._surfaceForms.items():
				result[sf] = ( sfObj.mappingEntityId, self._entityMap[sfObj.mappingEntityId].name )
		else:
			print( "[X] Nothing to compute!  No valid surface forms collected!", sys.stderr )

		return result


	def _assignCandidatesInitialScore( self ) -> np.ndarray:
		"""
		Assign to each candidate mapping entity an initial score based on the results from the iter. subs. algorithm.
		Normalize this weight and assign to each pair (sf,cm) a unique int-index to locate its score in an np.array.
		:return: An np.array holding the normalized initial score for all graph nodes.
		"""
		totalScore = 0.0									# Used for normalization of 'node' scores.
		npk = []											# List of scores.
		for sf, sfObj in self._surfaceForms.items():
			M = { ne: self._surfaceForms[ne].mappingEntityId for ne in self._surfaceForms }  # Reset to currently best mappings.
			for cm, cmObj in sfObj.candidates.items():
				# First compute the topical coherence for each candidate because after the iterative substitution algorithm we
				# got the best candidate mapping entities for every entity mention.
				M[sf] = cm  								# Check this candidate.
				cmObj.topicalCoherence = self._topicalCoherence( sf, M )

				# Then, assign the initial score (unnormalized).
				cmObj.initialScore = self._alpha * cmObj.priorProbability \
									 + self._beta * cmObj.contextSimilarity \
									 + self._gamma * cmObj.topicalCoherence
				totalScore += cmObj.initialScore

				# Also, assign a unique index for accessing the result np.array of scores.
				npk.append( cmObj.initialScore )
				self._indexToSFC.append( ( sf, cm ) )

		# Normalize initial scores among all surface forms' candidate mapping entities so that their sum equals 1.
		return np.array( npk ) / totalScore


	def _buildMatrixB( self ) -> sparse.spmatrix:
		"""
		Build the |V|x|V| propagation strength matrix.
		:return: B matrix.
		"""
		nV = len( self._indexToSFC )
		B = sparse.lil_matrix( ( nV, nV ) )			# B is a sparse matrix for efficient multiplication.

		# We first take advantage of the symmetry of the B matrix.  That is, b_{i,j} is the propagation strength from
		# node j to node i, where none of i or j belong to the same named entity mention.  In other words, we get a
		# sparse matrix with a diagonal of 0-blocks, and we just calculate the lower diagonal part (and mirror to upper).
		# After that, we normalize each column so that its sum totals 1.
		for j in range( nV ):												# From node j.
			if self._indexToSFC[j][0] == self._indexToSFC[-1][0]:			# Skipping last surface form.
				break
			for i in range( j + 1, nV ):									# To node i.
				if self._indexToSFC[j][0] == self._indexToSFC[i][0]:		# Skip candidates of the same surface form.
					continue

				tr = self._topicalRelatedness( self._indexToSFC[j][1], self._indexToSFC[i][1] )
				if tr > 0:
					B[i,j] = tr
					B[j,i] = tr

		# Now do the column normalization.
		B = B.tocsc()				# Using compressed sparse column matrix format for fast arithmetic operations.
		c = B.sum( axis=0 )
		for j in range( nV ):
			if c[0,j] > 0:			# Avoid division by zero.
				B[:,j] /= c[0,j]

		return B


	def _propagationAlgorithm( self ):
		"""
		Apply PageRank collective inference algorithm to compute final candidate mapping entities' scores.
		Modify in place the final score attribute of each candidate and the final mapping entity for each surface form.
		"""
		print( "[*] Executing propagation score algorithm..." )

		p = self._assignCandidatesInitialScore()		# Normalized (candidate) nodes initial score.
		B = self._buildMatrixB()						# Propagation strength matrix.
		s = np.array( p )								# Final score to be refined iteratively.

		diff = 1.0
		THRESHOLD = 0.001
		iteration = 0
		while diff > THRESHOLD:
			ns = self._lambda * p + ( 1.0 - self._lambda ) * B.dot( s )
			diff = np.linalg.norm( ns - s )
			s = ns
			iteration += 1
			print( "[PA] Iteration", iteration, ", Difference:", diff )

		# Assign final scores to candidate mapping entities.
		for i, finalScore in enumerate( s ):
			sf, cm = self._indexToSFC[i]
			self._surfaceForms[sf].candidates[cm].finalScore = finalScore

		print( "... Done!" )


	def _chooseBestCandidate_NoTopicalCoherence( self ):
		"""
		Choose the best candidate mapping entity for a surface form by only considering the weighted sum of prior
		probability and context similarity.
		"""
		for sf, sfObj in self._surfaceForms.items():
			bestScore = 0
			for cm, cmObj in sfObj.candidates.items():
				score = self._alpha * cmObj.priorProbability + self._beta * cmObj.contextSimilarity
				if score > bestScore:
					bestScore = score
					sfObj.mappingEntityId = cm		# Upon exiting, surface form object holds ID of best mapping entity.


	def _removeCommonDiscourseEmbedding( self ):
		"""
		Project each initial document embedding onto first right singular vector of a matrix formed with all of the
		document vectors.
		"""
		print( "[*] Now removing common discourse embedding from surface forms and candidate mapping entities document vectors" )
		eL = [e.v for eId, e in self._entityMap.items()]  		# List of embeddings for entities.
		sfL = [s.v for sf, s in self._surfaceForms.items()]  	# List of surface forms' context embeddings.
		X = np.array( eL + sfL )  								# Matrix whose rows are the embeddings of all docs.
		svd = TruncatedSVD( n_components=1, random_state=0, n_iter=7 )
		svd.fit( X )
		v1 = svd.components_[0, :]  							# First component in V (not in U): 300 dimensions.

		for eId in self._entityMap:  							# Remove common discourse in entities and surface forms embeddings.
			self._entityMap[eId].v -= v1 * v1.dot( self._entityMap[eId].v )
		for sf in self._surfaceForms:
			self._surfaceForms[sf].v -= v1 * v1.dot( self._surfaceForms[sf].v )

		print( "... Done!" )


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
					U = self._getPagesLikingTo( r )				# If no one points to entity r, don't add it to candidate list.
					if not U: continue

					record2 = self._mEntity_ID.find_one( { "_id": r }, projection={ "e": True } )		# Consult DB to retrieve information for new entity into cache.
					record3 = self._mSif_Documents.find_one( { "_id": r } )								# Extract words and frequencies in entity document.
					vd = self._getRawDocumentEmbedding( record3["w"], record3["f"] )					# Get an initial document embedding (without common component removed).
					self._entityMap[r] = Entity( r, record2["e"], U, vd )

				result[r] = Candidate( r, record1["m"][r_j] )	# Candidate has a reference ID to the entity object.
				total += record1["m"][r_j]

			# Update prior probability.
			for r in result:
				result[r].priorProbability = result[r].count / total

		print( "[*] Collected", len( result ), "candidate entities for [", m_i, "]" )
		return result			# Empty if no candidates were found for given entity mention.


	def _getRawDocumentEmbedding( self, words: List[str], freqs: List[int] ) -> np.ndarray:
		"""
		Compute an initial document embedding from input words and respective frequencies.
		Retrieve and cache word embeddings at the same time.
		:param words: List of document words -- must be lowercased!
		:param freqs: List of frequencies corresponding to document words.
		:return: \frac{1}{|d|} \sum_{w \in d}( \frac{a}{a + p(w)}v_w )
		"""
		vd = np.zeros( 300 )

		totalFreq = 0.0									# Count freqs of effective words in document for normalization.
		for i, w in enumerate( words ):
			f = freqs[i]
			if self._wordMap.get( w ) is None:			# Not in cache?
				r = self._mWord_Embeddings.find_one( { "_id": w } )
				if r is None:
					continue								# Skip words not in the vocabulary.

				vw = r["e"]									# Word embedding and probability.
				p = r["f"] / self._TOTAL_WORD_FREQ_COUNT
				self._wordMap[w] = Word( vw, p )			# Cache word object.

			vd += f * self._a / ( self._a + self._wordMap[w].p ) * self._wordMap[w].v
			totalFreq += f

		return vd / totalFreq		# Still need to subtract proj onto first singular vector (i.e. common discourse vector).


	def _getPagesLikingTo( self, e: int ) -> Set[int]:
		"""
		Collect IDs of pages linking to entity e.
		:param e: Target entity.
		:return: Set of entity IDs.
		"""
		record = self._mNed_Linking.find_one( { "_id": e }, projection={ "f": True } )
		U = set()
		if record:		# We may have no pages pointing to this one.
			for u in record["f"]:
				U.add( int( u ) )
		return U


	def _topicalRelatedness( self, u1: int, u2: int ) -> float:
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


	def _computeContextSimilarity( self ):
		"""
		Compute the consine similarity between the document embedding of surface form and all of its candidate mapping entities.
		"""
		print( "[*] Computing context similarity between named entity mentions and candidate mapping entities" )
		for sf, sfObj in self._surfaceForms.items():
			for c, cObj in sfObj.candidates.items():
				u = self._surfaceForms[sf].v
				v = self._entityMap[c].v
				cs = u.dot( v ) / (np.linalg.norm( u ) * np.linalg.norm( v ))

				# Store context similarity for later iter. subs. alg.
				# Normalize to a value between 0 and 1 since with word2vec we can get negative cos sim.
				cObj.contextSimilarity = ( 1.0 + cs ) / 2.0

		print( "... Done!" )


	def _topicalCoherence( self, surfaceForm: str, M: Dict[str, int] ) -> float:
		"""
		Compute topical coherence of candidate mapping entity for given surface form with respect to mapping entities of the rest.
		:param surfaceForm: Central surface form whose (candidate) mapping entity we task as reference.
		:param M: Mappings for all surface forms (including the central one).
		:return: Coh(r_{i,j}) = \frac{1}{|M| - 1} \sum_{c=1, c \ne i}^{|M|}TR(r_{i,j}, e_c).
		"""
		totalTR = 0
		for sf in M:
			if sf != surfaceForm:
				totalTR += self._topicalRelatedness( M[surfaceForm], M[sf] )
		return  totalTR / ( len( M ) - 1 )


	def _totalInitialScore( self, M: Dict[str, int] ) -> float:
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
						  + self._gamma * self._topicalCoherence( sf, M )
		return totalScore


	def _iterativeSubstitutionAlgorithm( self ):
		"""
		Solve for the best entity mappings for all named entities in order to have an initial score.
		"""
		print( "[*] Executing iterative substitution algorithm to compute candidate mapping entities' initial scores" )

		# Pick the candidate with maximum prior probability as first approximation to mapping entity.
		for sf, sfObj in self._surfaceForms.items():
			mostPopularProb = 0
			for cm, cmObj in sfObj.candidates.items():
				if cmObj.priorProbability > mostPopularProb:
					sfObj.mappingEntityId = cm
					mostPopularProb = cmObj.priorProbability

		iteration = 1
		M: Dict[str: int] = { ne: self._surfaceForms[ne].mappingEntityId for ne in self._surfaceForms }	# Surface forms and respective mapping entities.
		bestTIS = self._totalInitialScore( M )

		print( "[ISA][", iteration,"] Initial score starts at", bestTIS )

		while True:
			# Find the candidate substition that increases the most the total initial score among all candidates and all surface forms.
			foundBetterScore = False
			bestNE = 0		# Best surface form and candidate mapping.
			bestCM = 0
			for ne in self._surfaceForms:
				M = { sf: self._surfaceForms[sf].mappingEntityId for sf in self._surfaceForms }  # Try-out mapping entities.
				for cm in self._surfaceForms[ne].candidates:
					if cm != self._surfaceForms[ne].mappingEntityId:		# Skip currently selected best candidate mapping.
						M[ne] = cm											# Check this candidate substitution.

						tis = self._totalInitialScore( M )
						if tis > bestTIS:									# Is it improving?
							foundBetterScore = True
							bestNE = ne
							bestCM = cm
							bestTIS = tis

			if foundBetterScore:											# Did score ever increase?
				self._surfaceForms[bestNE].mappingEntityId = bestCM			# New best mapping.
				print( "[ISA][", iteration ,"] Score increased to", bestTIS )
				iteration += 1
			else:
				break						# Stop when we detect a negative performance.

		print( "[ISA] Finalized greedy optimization with an initial score of", bestTIS )
		print( "... Done!" )