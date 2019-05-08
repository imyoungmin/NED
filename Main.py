import importlib
import NED as N
importlib.reload( N )

if __name__ is "__main__":
	ned = N.NED()
	ned.go( "Datasets/comet.txt" )


