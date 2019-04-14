"""
Tokenization function as a translation from
https://www.clips.uantwerpen.be/conll2003/ner/ and https://www.clips.uantwerpen.be/conll2003/ner/bin/
"""
import re

_ABBREV = { "apr": True, "aug": True, "av": True, "bldg": True, "dec": True, "dr": True, "calif": True, "corp": True,
			"feb": True, "fla": True, "inc": True, "jan": True, "jul": True, "jun": True, "lt": True, "ltd": True,
			"mar": True, "mr": True, "mrs": True, "ms": True, "nov": True, "oct": True, "rev": True, "sep": True,
			"st": True, "vol": True, "vols": True, "vs": True }

_SPORTS = {
	"SOCCER": True, "CRICKET": True, "BASEBALL": True, "TENNIS": True, "GOLF": True, "CYCLING": True, "ATHLETICS": True,
	"BASKETBALL": True, "RALLYING": True, "SQUASH": True, "BOXING": True, "MOTORCYCLING": True, "BADMINTON": True,
	"MOTOCROSS": True, "SWIMMING": True, "ROWING": True, "HOCKEY": True, "REUTERS": True, "RTRS UNION": True,
	"RACING": True, "BASKETBALLSOCCER LEAGUE": True, "Soccer": True, "Tennis": True, "Cricket": True
}


def splice( l, offset, count, el ):
	"""
	Insert a list of elements in a given position of a list after removing a given number of previous elements.
	:param l: Source list.
	:param offset: Where to start inserting the new elements.
	:param count: How many elements will be disregarded from the source list.
	:param el: List of elements to insert after removing count elements starting at offset.
	:return:
	"""
	return l[:offset] + el + l[(offset + count):]


def abbrev( word ):
	"""
	Check if a word is an abbreviature.
	:param word: Word to check.
	:return: True if abbreviature, false otherwise.
	"""
	word = word.lower()
	if re.match( r"\.", word ) and re.match( r"\d", word ) is None:
		return True
	if re.match( r"^[a-z]$", word ):
		return True
	return bool( _ABBREV.get( word ) )


def tokenize( text ):
	"""
	Tokenize text according to CoNLL 2003 standards.
	:param text: Text to tokenize.
	:return: List of tokens.
	"""
	sentences = text.split( "\n" )
	tokens = []
	for sentence in sentences:
		s = sentence.strip()
		if s == "":
			continue

		words = re.split( r"\s+", s )
		i = 0
		while i < len( words ):				# Now split words from non-empty sentences.
			m = re.match( r"^([\"'()\[\]$:;,/%])(.+)$", words[i] )			# Remove punctuation from start of word.
			if m and re.match( r"^'[dsm]$", words[i], re.I ) is None and re.match( r"^'re$", words[i], re.I ) is None and re.match( r"^'ve$", words[i], re.I ) is None and re.match( r"^'ll$", words[i], re.I ) is None:
				words = splice( words, i, 1, [m.group(1), m.group(2)] )
				i += 1
			else:
				m = re.match( r"^(.+)([?!.])([\"'])$", words[i] )			# Remove sentence breaking punctuation with quote from end of word.
				if m:
					words = splice( words, i, 1, [m.group(1), m.group(2) + m.group(3), "\n"] )
				else:
					m = re.match( r"^(.+)([:;,\"')(\[\]%])$", words[i] )	# Remove non-sentence-breaking punctuation from end of word.
					if m:
						words = splice( words, i, 1, [m.group(1), m.group(2)] )
					else:
						m = re.match( r"^(.+)([?!])$", words[i] )			# Remove sentence-breaking punctuation (not period) from end of word.
						if m is None:
							m = re.match( r"^(.+[^.])(\.\.+)$", words[i] )
						if m:
							words = splice( words, i, 1, [m.group(1), m.group(2), "\n"] )
						else:
							m = re.match( r"^([a-z]+\$)(.+)$", words[i], re.I )		# Separate currency symbol from value.
							if m:
								words = splice( words, i, 1, [m.group(1), m.group(2)] )
								i += 1
							else:
								m = re.match( r"^(.*)-\$(.*)$", words[i], re.I )	# Separate currency symbol from other symbols.
								if m:
									words = splice( words, i, 1, [m.group(1), "-", "\$", m.group(2)] )
									i += 1
								else:
									m = re.match( r"^(.+)('re|'ve|'ll|n't|'[dsm])$", words[i], re.I )			# Split words like we're did't etcetera.
									if m:
										words = splice( words, i, 1, [m.group(1), m.group(2)] )
									else:
										m = re.match( r"^(.*[a-z].*)([\",()])(.*[a-z].*)$", words[i], re.I )	# Split words with punctuation in the middle.
										if m:
											words = splice( words, i, 1, [m.group(1), m.group(2), m.group(3)] )
										else:
											m = re.match( r"^(.*[^.])(\.\.+)([^.].*)$", words[i] )				# Separate words linked with sequence (>=2) of periods.
											if m:
												words = splice( words, i, 1, [m.group(1) + m.group(2), m.group(3)] )
											else:
												m = re.match( r"^(-+)([^\-].*)$", words[i] )					# Remove initial hyphens from word.
												if m:
													words = splice( words, i, 1, [m.group(1), m.group(2)] )
												else:
													m = re.match( r"^([A-Za-z]+)-(.*)$", words[i] )				# Separate sport types and first words in article titles.
													if m and _SPORTS.get( m.group(1) ):
														words = splice( words, i, 1, [m.group(1), "-", m.group(2)] )
													else:
														m = re.match( r"^([0-9/]+)-([A-Z][a-z].*)$", words[i] )				# Separate number and word linked with hyphen.
														if m:
															words = splice( words, i, 1, [m.group(1), "-", m.group(2)] )
														else:
															m = re.match( r"^([0-9/]+)\.([A-Z][a-z].*)$", words[i] )		# Separate number and word linked with period.
															if m:
																words = splice( words, i, 1, [m.group(1) + ".", m.group(2)] )
															else:
																m = re.match( r"^(.*)\.-([A-Z][a-z].*)$", words[i] ) 		# Separate number and word linked with period.
																if m:
																	words = splice( words, i, 1, [m.group(1) + ".", "-", m.group(2)] )
																else:
																	m = re.match( r"^([A-Z]\.)([A-Z][a-z].*)$", words[i] )	# Separate initial from name.
																	if m:
																		words = splice( words, i, 1, [m.group(1), m.group(2)] )
																	else:
																		m = re.match( r"^(.*[0-9])(\.)$", words[i] )		# Introduce sentence break after number followed by period.
																		if i != 0 and m:
																			words = splice( words, i, 1, [ m.group(1), m.group(2), "\n"] )
																		else:
																			m = re.match( r"^(.+)/(.+)$", words[i] )		# Split words containing a slash if they are not a URI.
																			if re.match( r"^(ht|f)tps*", words[i], re.I ) is None and re.match( r"[^0-9/\-]", words[i] ) and m:
																				words = splice( words, i, 1, [m.group(1), "/", m.group(2)] )
																			else:
																				m = re.match( r"^(.+)(\.)$", words[i] )		# Put sentence break after period if it is not an abbreviation.
																				if m and re.match( r"^\.+$", words[i] ) is None and re.match( r"^[0-9]+\.", words[i] ) is None:
																					word = m.group(1)
																					if i != len( words ) - 1 and abbrev( word ):
																						i += 1
																					else:
																						words = splice( words, i, 1, [m.group(1), m.group(2), "\n"] )
																				else:
																					i += 1
		if words[len( words ) - 1] != "\n":
			words = words + ["\n"]				# Add new line at the end of the token list.
		line = " ".join( words )
		line = re.sub( r" ([?!.]) \n ([\"']) ", r" \1 \2 \n ", line )
		line = re.sub( r" *\n *", r"\n", line )

		# Print every word on a separate line.
		line = re.sub( r" +", r" ", line )
#		line = re.sub( r"\n", r"\n\n", line )
#		line = re.sub( r" ", r"\n", line )
		tokens += line.split()

	return tokens