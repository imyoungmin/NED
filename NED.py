from pymongo import MongoClient
from typing import Set, Dict


class Candidate:
	"""
	Candidate mapping implementation.
	"""

	def __init__( self, eId: int, name: str, count: int, pointedToBy: Set[int] ):
		"""
		Constructor.
		:param eId: Entity ID.
		:param name: Entity name.
		:param count: Number of entities being referred to by (outter/caller) surface form.
		:param pointedToBy: Set of entity IDs that have a link to this candidate.
		"""
		self.id = eId
		self.name = name
		self.count = count
		self.pointedToBy = pointedToBy


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
		print( "NED initialized with", self._WP, "entities" )


	def getCandidatesForEntityMention( self, m_i: str ) -> Dict[int, Candidate]:
		"""
		Retrieve candidate mapping entities for given entity mention.
		:param m_i: Entity mention (a.k.a surface form).
		:return: A dict {e_1_id:(str, int), "e_2_id":(str, int),...}, where the tuples contain the mapping name and count.
		"""
		result = {}
		record1 = self._mNed_Dictionary.find_one( { "_id": m_i.lower() }, projection={ "m": True } )
		if record1:
			for r_j in record1["m"]:
				r = int( r_j )
				record2 = self._mEntity_ID.find_one( { "_id": r }, projection={ "e": True } )
				result[r] = Candidate( r, record2["e"], record1["m"][r_j], self._getPagesLikingTo( r ) )	# Build resulting candidate object.

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


	def topicalRelatedness( self, u1: int, u2: int ):
		"""
		Calculate the Wikipedia topical relatedness between two entities.
		:param u1: Entity ID 1.
		:param u2: Entity ID 2
		:return: 1 - \frac{log(max(|U_1|, |U_2|)) - log(|U_1 \intersect U_2|)}{log|WP| - log(min(|U_1|, |U_2|))}
		"""
		pass
