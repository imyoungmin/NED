import importlib
from Disambiguation import Task as T

importlib.reload( T )


if __name__ is "__main__":
	# Disambiguating regular text with labeled named entity mentions.
	results = T.Task.disambiguateTextFile( "Datasets/madonna.txt" )

	# Disambiguating entities in a big dataset to measure accuracy.
	# T.Task.debug = False
	# T.Task.evaluateAccuracy( "Datasets/AIDA-YAGO2-dataset.tsv" )