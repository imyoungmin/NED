import importlib
from Disambiguation import Task as T

importlib.reload( T )

# Disambiguating regular text with labeled named entity mentions.
results = T.Task.disambiguateTextFile( "Datasets/madonna.txt" )

# Disambiguating entities in a big dataset to measure accuracy.
# T.Task.debug = False			# Avoid dumping all process info to the console.
# T.Task.evaluateAccuracy( "Datasets/AIDA-YAGO2-dataset.tsv" )