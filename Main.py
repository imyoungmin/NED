import importlib
import NED as N
importlib.reload( N )

if __name__ is "__main__":
	ned = N.NED()
	candidatesNBA = ned.getCandidatesForNamedEntity( "NBA" )
	candidatesAllen = ned.getCandidatesForNamedEntity( "Tony Allen" )

	print( "Done!" )


