import importlib
import Tokenizer
importlib.reload( Tokenizer )

with open( "Datasets/tokenizerTest.txt" ) as file:
	text = file.read()
	Tokenizer.tokenize( text )