import importlib
from Disambiguation import Task as T

importlib.reload( T )

if __name__ is "__main__":
	task = T.Task()
	results = task.disambiguateTextFile( "Datasets/madonna.txt" )

	# Print results.
	print( "------ Results ------")
	for sf, rt in results.items():
		print( sf, rt )