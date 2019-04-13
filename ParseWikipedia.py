import importlib
from WikiParser import Parser

importlib.reload( Parser )

wikiParser = Parser.Parser()
wikiParser.go()