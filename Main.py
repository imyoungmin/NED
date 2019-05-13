import importlib
import NED as N
importlib.reload( N )

if __name__ is "__main__":
	ned = N.NED()
	results = ned.go( "Datasets/sports.txt" )

	# Print results.
	print( "------ Results ------")
	for sf, rt in results.items():
		print( sf, rt )