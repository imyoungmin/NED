import pymongo
from typing import Set, Dict, Tuple, List
import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Value
from scipy import sparse
from WikiParser import SIFParser as S
import importlib

importlib.reload( S )


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
		self.isaScore = 0.0				# p_{i,j} = \alpha * Pp(r_{i,j}) + \beta * Sim(r_{i,j}) + \gamma * Coh(r_{i,j}).
		self.paScore = 0.0				# To be updated in the propagation algorithm.


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


	# These static and shared variables prevent querying the DB and some files for all (multiprocessing) instances of these class.
	_WP = Value( "i", -1 )			# A negative value indicates we need to load their values.
	_LOG_WP = Value( "d", -1.0 )
	_TOTAL_WORD_FREQ_COUNT = Value( "d", -1.0 )

	def __init__( self, debug=True ):
		"""
		Constructor.
		:param debug: True for printing debug messages, false otherwise.
		"""
		self._debug = debug

		# MongoDB connections.
		self._mClient: pymongo.mongo_client = pymongo.MongoClient( "mongodb://localhost:27017/" )
		self._mNED = self._mClient["ned"]
		self._mEntity_ID: pymongo.collection.Collection = self._mNED["entity_id"]  				# {_id:int, e:str, e_l:str}.

		# Connections to SIF collections.
		self._mWord_Embeddings: pymongo.collection.Collection = self._mNED["word_embeddings"] 	# {_id:str, e:List[float], f:int}
		self._mSif_Documents: pymongo.collection.Collection = self._mNED["sif_documents"]		# {_id:int, w:List[str], f:List[int], e:List[float]}

		# Defining connections to collections for entity disambiguation.
		self._mNed_Dictionary: pymongo.collection.Collection = self._mNED["ned_dictionary"]  	# {_id:str, m:{"e_1":int, "e_2":int,..., "e_n":int}}. -- m stands for "mapping".
		self._mNed_Linking: pymongo.collection.Collection = self._mNED["ned_linking"]  			# {_id:int, f:{"e_1":true, "e_2":true,..., "e_3":true}}. -- f stands for "from".

		# Retrieve shared static constants if they haven't been loaded.
		with NED._WP.get_lock():
			if NED._WP.value < 0 or NED._LOG_WP.value < 0 or NED._TOTAL_WORD_FREQ_COUNT.value < 0:
				# Retrieve total number of entities recorded in DB.
				NED._WP.value = self._mEntity_ID.count()
				NED._LOG_WP.value = np.log( NED._WP.value )							# Log used in topic relatedness metric.
				print( "NED initialized with", NED._WP.value, "entities" )

				# Read total word frequencies.
				with open( "Datasets/wordcount.txt", "r", encoding="utf-8" ) as fIn:
					NED._TOTAL_WORD_FREQ_COUNT.value = float( fIn.read() )
				print( "NED initialized with", NED._TOTAL_WORD_FREQ_COUNT.value, "total word frequency count" )

		# Initialize map of word objects.
		self._wordMap: Dict[str, Word] = {}

		self._a = S.SIFParser.A_SIF_PARAMETER										# Parameter 'a' for SIF.

		# Initialize map of entities (as a cache).
		self._entityMap: Dict[int, Entity] = {}

		# Named entities map {"namedEntity1": NamedEntity1, "namedEntity2": NamedEntity2, ...}.
		self._surfaceForms: Dict[str, SurfaceForm] = {}
		self._WINDOW_SIZE = 50

		# Initial score constants.
		self._alpha = 0.4
		self._beta = 0.6
		self._gamma = 0.0

		# Propagation algorithm damping constant.
		self._lambda = 0.4

		# Map of array index to (surface form, candidate mapping entity ID).
		self._indexToSFC: List[Tuple[str, int]] = []


	def go( self, tokens: List[str], surfaceForms: Dict[str, List[Tuple[int, int]]] ) -> Dict[str, Tuple[int, str]]:
		"""
		Disambiguate indentified entities given in tokenized input list for the input map of identified surface forms.
		:param tokens: List of tokenized text (including surface form tokens).
		:param surfaceForms: Dictionary of surface forms with a tuple indicating [start, end) in the token list.
		:return: Dictionary with surface forms and mapping entities
		"""

		# For each surface form compute an initial 'document' embedding (without common component removed).
		# Also collect the candidate mapping entities.
		# If a surface form doesn't have any candidates, skip it.
		if self._debug: print( "[*] Now collecting candidate mapping entities for surface forms in input text:" )
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
					if self._debug: print( "[W] Surface form [", sf, "] doesn't have any candidate mapping entity!  Will be ignored...", sys.stderr )
			else:
				if self._debug: print( "[W] Surface form [", sf, "] doesn't have a valid document embedding!  Will be ignored...", sys.stderr )
		if self._debug: print( "... Done!" )

		### Getting ready for disambiguation ###

		result: Dict[str, Tuple[int, str]] = {}		# Result mapping for each named entity/surface form.

		if self._surfaceForms:
			self._removeCommonDiscourseEmbedding()	# Remove projection onto first singular vector (i.e. common discourse vector).
			self._computeContextSimilarity() 		# Compute context similary of surface forms' BOW with respect to candidates mapping entities.

			# Get an initial score for candidate mapping entities.
			self._chooseBestCandidate_NoTopicalCoherence()
			if len( self._surfaceForms ) > 1:

				if self._debug:
					print( "----------------------------- Initial results -----------------------------" )
					for sf, sfObj in self._surfaceForms.items():
						print( "*", sf, ": (", sfObj.mappingEntityId, ") ", self._entityMap[sfObj.mappingEntityId].name )
					print( "---------------------------------------------------------------------------" )

				# Then, apply the page rank algorithm to calculate final candidate mapping entity scores.
				self._propagationAlgorithm()
				self._reReranking()
			else:
				if self._debug: print( "[!] There's only one surface form.  No need for propagation algorithm!" )

			# Place results in return object.
			for sf, sfObj in self._surfaceForms.items():
				result[sf] = ( sfObj.mappingEntityId, self._entityMap[sfObj.mappingEntityId].name )
		else:
			if self._debug: print( "[x] Nothing to compute!  No valid surface forms collected!", sys.stderr )

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
			for cm, cmObj in sfObj.candidates.items():
				# Assign the initial score (unnormalized).
				cmObj.isaScore = self._alpha * cmObj.priorProbability + self._beta * cmObj.contextSimilarity
				totalScore += cmObj.isaScore

				# Also, assign a unique index for accessing the result np.array of scores.
				npk.append( cmObj.isaScore )
				self._indexToSFC.append( ( sf, cm ) )

		# Normalize initial scores among all surface forms' candidate mapping entities so that their sum equals 1.
		for _, sfObj in self._surfaceForms.items():
			for _, cmObj in sfObj.candidates.items():
				cmObj.isaScore /= totalScore

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
		if self._debug: print( "[*] Executing propagation score algorithm..." )

		p = self._assignCandidatesInitialScore()		# Normalized (candidate) nodes initial score.
		B = self._buildMatrixB()						# Propagation strength matrix.
		s = np.array( p )								# Final score to be refined iteratively.

		diff = 1.0
		THRESHOLD = 0.001
		iteration = 0
		while diff > THRESHOLD:
			ns = self._lambda * p + ( 1.0 - self._lambda ) * B.dot( s )
			ns /= np.sum( ns )
			diff = np.linalg.norm( ns - s )
			s = ns
			iteration += 1
			if self._debug: print( "[PA] Iteration", iteration, ", Difference:", diff )

		# Assign final scores to candidate mapping entities.
		for i, paScore in enumerate( s ):
			sf, cm = self._indexToSFC[i]
			self._surfaceForms[sf].candidates[cm].paScore = paScore

		# Select the best candidate mapping entity for each surface form.
		for sf, sfObj in self._surfaceForms.items():
			bestScore = 0
			for cm, cmObj in sfObj.candidates.items():
				if cmObj.paScore > bestScore:
					bestScore = cmObj.paScore
					sfObj.mappingEntityId = cm

		if self._debug: print( "... Done!" )


	def _reReranking( self ):
		"""
		Compute R_s and R_m by summing and multiplying the initial and propagation scores of each candidate.
		Then, select the highest ranked candidate such that its distance to the second place is the largest possible.
		"""
		if self._debug: print("[*] Re-ranking candidate mapping entities...")

		for sf, sfObj in self._surfaceForms.items():
			if len( sfObj.candidates ) == 1: continue				# Skip syrface forms with just one candidate mapping entity.

			# Collect candidate initial and PA scores.
			candidateScores: Dict[int, List[float]] = {}			# Each candidate has a pair with [isaScore, paScore].
			totalIsaScore = 0
			totalPaScore = 0
			for cm, cmObj in sfObj.candidates.items():
				candidateScores[cm] = [ cmObj.isaScore, cmObj.paScore ]
				totalIsaScore += cmObj.isaScore
				totalPaScore += cmObj.paScore

			# Normalize initial and PA scores for sf's candidate mappings so that their sum is 1.
			for cm in candidateScores:
				candidateScores[cm][0] /= totalIsaScore
				candidateScores[cm][1] /= totalPaScore

			# Compute the R_s and R_m lists of scores.
			rankedCandidates: Dict[int, List[float]] = {}			# Each candidate has a pair with [R_s(r_j), R_m(r_j)].
			totalRsScore = 0										# For normalization of sums and products below per
			totalRmScore = 0										# surface form.
			for cm, cmObj in candidateScores.items():				# R_s = isaScore + paScore; R_m = isaScore * paScore.
				rankedCandidates[cm] = [ cmObj[0] + cmObj[1], cmObj[0] * cmObj[1] ]
				totalRsScore += rankedCandidates[cm][0]
				totalRmScore += rankedCandidates[cm][1]

			# Normalize Rs and Rm for surface forms' candidate mapping entities so that they sum up to one.
			for cm in rankedCandidates:
				rankedCandidates[cm][0] /= totalRsScore
				rankedCandidates[cm][1] /= totalRmScore

			# Find the first and second place candidates from each of the Rs and Rm ranks by sorting.
			Rs = sorted( rankedCandidates, key=lambda x: rankedCandidates[x][0], reverse=True )
			Rm = sorted( rankedCandidates, key=lambda x: rankedCandidates[x][1], reverse=True )

			# Rerank based on distance between first and second place.  Choose the most discriminative criterion.
			RsDiff = rankedCandidates[Rs[0]][0] - rankedCandidates[Rs[1]][0]
			RmDiff = rankedCandidates[Rm[0]][1] - rankedCandidates[Rm[1]][1]
			if RsDiff > RmDiff:
				sfObj.mappingEntityId = Rs[0]
			else:
				sfObj.mappingEntityId = Rm[0]

		if self._debug: print( "... Done!" )



	def _chooseBestCandidate_NoTopicalCoherence( self ):
		"""
		Choose the best candidate mapping entity for surface forms by only considering the weighted sum of prior
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
		if self._debug: print( "[*] Now removing common discourse embedding from surface forms and candidate mapping entities document vectors" )
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

		if self._debug: print( "... Done!" )


	def _getCandidatesForNamedEntity( self, m_i: str ) -> Dict[int, Candidate]:
		"""
		Retrieve candidate mapping entities for given named entity.
		Calculate the prior probability at the same time.
		:param m_i: Entity mention (a.k.a surface form) in lowercase.
		:return: A dict {e_1_id:Candidate_1, e_2_id:Candidate_2,...}.
		"""
		result = {}
		record1 = self._mNed_Dictionary.find_one( { "_id": m_i }, projection={ "m": True } )
		oneCounters = 0											# Count number of documents with 'count' 1 if skipOneCounters is true.
		if record1:
			total = 0											# Accumulate reference count for this surface form by the candidate mappings.

			skipOneCounters = len( record1["m"] ) > 50			# If there are a lot of candidates skip those with a count of 1.

			for r_j in record1["m"]:
				r = int( r_j )

				if skipOneCounters and record1["m"][r_j] == 1:
					oneCounters += 1
					continue

				# Check the cache for entity.
				if self._entityMap.get( r ) is None:
					U = self._getPagesLikingTo( r )				# If no one points to entity r, don't add it to candidate list.
					if not U: continue

					record2 = self._mEntity_ID.find_one( { "_id": r }, projection={ "e": True } )		# Consult DB to retrieve information for new entity into cache.
					record3 = self._mSif_Documents.find_one( { "_id": r } )								# Extract words and frequencies in entity document.

					# Get the initial document embedding from the DB. If it's not there, compute it and store it for fast, later computations.
					if record3.get( "e" ) is not None:
						vd = np.array( record3["e"] )
					else:
						vd = self._getRawDocumentEmbedding( record3["w"], record3["f"] )					# Compute an initial document embedding (without common
						self._mSif_Documents.update_one( { "_id": r }, { "$set": { "e": vd.tolist() } } )	# component removed) and save it.
						if self._debug: print( "    + Saved document embedding for entity", r, "[", record2["e"], "]" )

					if not np.count_nonzero( vd ): continue		# Skip an entity with no doc embedding.

					self._entityMap[r] = Entity( r, record2["e"], U, vd )

				result[r] = Candidate( r, record1["m"][r_j] )	# Candidate has a reference ID to the entity object.
				total += record1["m"][r_j]

			# Update prior probability.
			for r in result:
				result[r].priorProbability = result[r].count / total

		if self._debug: print( "    Collected", len( result ), "candidate entities for [", m_i, "]. Skipped", oneCounters, "one-counters." )
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

		totalFreq = 0.0										# Count freqs of effective words in document for normalization.
		for i, w in enumerate( words ):
			f = freqs[i]
			if self._wordMap.get( w ) is None:				# Not in cache?
				r = self._mWord_Embeddings.find_one( { "_id": w } )
				if r is None:
					continue								# Skip words not in the vocabulary.

				vw = r["e"]									# Word embedding and probability.
				p = r["f"] / self._TOTAL_WORD_FREQ_COUNT.value
				self._wordMap[w] = Word( vw, p )			# Cache word object.

			vd += f * self._a / ( self._a + self._wordMap[w].p ) * self._wordMap[w].v
			totalFreq += f

		if totalFreq > 0:
			return vd / totalFreq		# Still need to subtract proj onto first singular vector (i.e. common discourse vector).
		else:
			return  vd


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
			return 1.0 - ( np.log( max( lU1, lU2 ) ) - np.log( lIntersection ) ) / ( NED._LOG_WP.value - np.log( min( lU1, lU2 ) ) )
		else:
			return 0.0


	def _computeContextSimilarity( self ):
		"""
		Compute the consine similarity between the document embedding of surface form and all of its candidate mapping entities.
		"""
		if self._debug: print( "[*] Computing context similarity between named entity mentions and candidate mapping entities" )
		for sf, sfObj in self._surfaceForms.items():
			for c, cObj in sfObj.candidates.items():
				u = self._surfaceForms[sf].v
				v = self._entityMap[c].v
				cs = u.dot( v ) / (np.linalg.norm( u ) * np.linalg.norm( v ))

				# Store context similarity for later iter. subs. alg.
				# Normalize to a value between 0 and 1 since with word2vec we can get negative cos sim.
				cObj.contextSimilarity = ( 1.0 + cs ) / 2.0

		if self._debug: print( "... Done!" )


	def reset( self ):
		"""
		Release map references for surface forms and candidate mapping entities so that this NED object can be reused.
		"""
		for sf in list( self._surfaceForms ):							# Forced to create a copy of the dict keys so
			candidateList = list( self._surfaceForms[sf].candidates )	# we can safely delete references.
			for cm in candidateList:
				del self._surfaceForms[sf].candidates[cm]
			self._surfaceForms[sf].candidates.clear()
			del self._surfaceForms[sf]

		self._surfaceForms.clear()


	def __del__( self ):
		"""
		Destructor.
		"""
		self.reset()

		# Release words and entites.
		for w in list( self._wordMap ):
			del self._wordMap[w]
		self._wordMap.clear()

		for e in list( self._entityMap ):
			del self._entityMap[e]
		self._entityMap.clear()

		# Close connection to DB.
		self._mClient.close()
		if self._debug: print( "[-] NED instance deleted.  Connection to DB 'ned' has been closed" )
