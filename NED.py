from pymongo import MongoClient
from typing import Set, Dict
import numpy as np


class Entity:
	"""
	Implementation of an entity.  We want to have at most 1 object per entity across surface forms' candidates.
	"""

	def __init__( self, eId: int, name: str, pointedToBy: Set[int] ):
		"""
		Constructor.
		:param eId: Entity ID.
		:param name: Entity name as it appears in its Wikipedia article.
		:param pointedToBy: Set of Wikipedia articles (i.e. entity IDs) pointing to this entity.
		"""
		self.id = eId
		self.name = name
		self.pointedToBy = pointedToBy


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


class NED:
	"""
	Implementation of Graph-based named entity disambiguation class.
	"""

	def __init__( self ):
		"""
		Constructor.
		"""
		# MongoDB connections.
		self._mClient = MongoClient( "mongodb://localhost:27017/" )
		self._mNED = self._mClient.ned
		self._mEntity_ID = self._mNED["entity_id"]  			# {_id:int, e:str, e_l:str}.

		# Connections to TFIDF collections.
		self._mIdf_Dictionary = self._mNED["idf_dictionary"]  	# {_id:str, idf:float}.
		self._mTf_Documents = self._mNED["tf_documents"]  		# {_id:int, t:[t1, ..., tn], w:[w1, ..., wn]}. -- t stands for "term", w for "weight".

		# Defining connections to collections for entity disambiguation.
		self._mNed_Dictionary = self._mNED["ned_dictionary"]  	# {_id:str, m:{"e_1":int, "e_2":int,..., "e_n":int}}. -- m stands for "mapping".
		self._mNed_Linking = self._mNED["ned_linking"]  		# {_id:int, f:{"e_1":true, "e_2":true,..., "e_3":true}}. -- f stands for "from".

		# Retrieve total number of entities recorded in DB.
		self._WP = self._mEntity_ID.count()
		self._LOG_WP = np.log( self._WP )						# Log used in topic relatedness metric.
		print( "NED initialized with", self._WP, "entities" )

		# Initialize map of entities (as a cache).
		self._entityMap = {}


	def getCandidatesForEntityMention( self, m_i: str ) -> Dict[int, Candidate]:
		"""
		Retrieve candidate mapping entities for given entity mention.
		Calculate the prior probability at the same time.
		:param m_i: Entity mention (a.k.a surface form).
		:return: A dict {e_1_id:(str, int), "e_2_id":(str, int),...}, where the tuples contain the mapping name and count.
		"""
		result = {}
		record1 = self._mNed_Dictionary.find_one( { "_id": m_i.lower() }, projection={ "m": True } )
		if record1:
			total = 0											# Accumulate reference count for this surface form by the candidate mappings.
			for r_j in record1["m"]:
				r = int( r_j )

				# Check the cache for candidate entity.
				if self._entityMap.get( r ) is None:
					record2 = self._mEntity_ID.find_one( { "_id": r }, projection={ "e": True } )		# Consult DB to retrieve information for new entity in cache.
					self._entityMap[r] = Entity( r, record2["e"], self._getPagesLikingTo( r ) )			# And retrieve other pages linking to new entity.

				result[r] = Candidate( r, record1["m"][r_j] )	# Candidate has a reference ID to the entity object.
				total += record1["m"][r_j]

			# Update prior probability.
			for r in result:
				result[r].priorProbability = result[r].count / total

		print( "[*] Collected", len( result ), "candidate entities for [", m_i, "]" )
		return result			# Empty if no candidates were found for given entity mention.


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


	def topicalRelatedness( self, u1: Entity, u2: Entity ) -> float:
		"""
		Calculate the Wikipedia topical relatedness between two entities.
		:param u1: First entity.
		:param u2: Second entity.
		:return: 1 - \frac{log(max(|U_1|, |U_2|)) - log(|U_1 \intersect U_2|)}{log|WP| - log(min(|U_1|, |U_2|))}
		"""
		lU1 = len( u1.pointedToBy )
		lU2 = len( u2.pointedToBy )
		lIntersection = len( u1.pointedToBy.intersection( u2.pointedToBy ) )
		if lIntersection > 0:
			return 1.0 - ( np.log( max( lU1, lU2 ) ) - np.log( lIntersection ) ) / ( self._LOG_WP - np.log( min( lU1, lU2 ) ) )
		else:
			return 0.0
