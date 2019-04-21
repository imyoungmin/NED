import importlib
import Tokenizer
import WikiParser.Parser as Parser
importlib.reload( Tokenizer )
importlib.reload( Parser )

# with open( "Datasets/tokenizerTest.txt" ) as file:
# 	text = file.read()
# 	print( Tokenizer.tokenize( text ) )

if __name__ is "__main__":
	Parser.buildTFIDFDictionary()
