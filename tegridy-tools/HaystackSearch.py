"""
Needle in a haystack search

Original source code is located here:
https://github.com/agapow/py-gsp/blob/master/gsp/motifsearch.py
"""

"""
A modifiable GSP algorithm.
"""

__version__ = '0.1'


### IMPORTS

### CONSTANTS & DEFINES

PP_INDENT = 3


### CODE ###

class GspSearch (object):
	"""
	A generic GSP algorithm, alllowing the individual parts to be overridden.
	
	This is setup so the object can be created once, but searched multiple times
	at different thresholds. In this generic form, we assume that the transactions
	are simply strings. 
	"""
	
	def __init__ (self, raw_transactions):
		"""
		C'tor, simply shaping the raw transactions into a useful form.
		"""
		self.process_transactions (raw_transactions)
		
	def process_transactions (self, raw_transactions):
		"""
		Create the alphabet & (normalized) transactions.
		"""
		self.transactions = []
		alpha = {}
		for r in raw_transactions:
			for c in r:
				alpha[c] = True
			self.transactions.append (r)
		self.alpha = alpha.keys()
			
	def generate_init_candidates (self):
		"""
		Make the initial set of candidate.
		
		Usually this would just be the alphabet.
		"""
		return list (self.alpha)
		
	def generate_new_candidates (self, freq_pat):
		"""
		Given existing patterns, generate a set of new patterns, one longer.
		"""
		old_cnt = len (freq_pat)
		old_len = len (freq_pat[0])
		print ("Generating new candidates from %s %s-mers ..." % (old_cnt, old_len))
		
		new_candidates = []
		for c in freq_pat:
			for d in freq_pat:
				merged_candidate = self.merge_candidates (c, d)
				if merged_candidate and (merged_candidate not in new_candidates):
					new_candidates.append (merged_candidate)
		
		## Postconditions & return:
		return new_candidates
		
	def merge_candidates (self, a, b):
		if a[1:] == b[:-1]:
			return a + b[-1:]
		else:
			return None
	
	def filter_candidates (self, trans_min):
		"""
		Return a list of the candidates that occur in at least the given number of transactions.
		"""
		filtered_candidates = []
		for c in self.candidates:
			curr_cand_hits = self.single_candidate_freq (c)
			if trans_min <= curr_cand_hits:
				filtered_candidates.append ((c, curr_cand_hits))
		return filtered_candidates
		
	def single_candidate_freq (self, c):
		"""
		Return true if a candidate is found in the transactions.
		"""
		hits = 0
		for t in self.transactions:
			if self.search_transaction (t, c):
				hits += 1
		return hits
		
	def search_transaction (self, t, c):
		"""
		Does this candidate appear in this transaction?
		"""
		return (t.find (c) != -1)
		
	def search (self, threshold):
		## Preparation:
		assert (0.0 < threshold) and (threshold <= 1.0)
		trans_cnt = len (self.transactions)
		trans_min = trans_cnt * threshold
		
		print ("The number of transactions is: %s" % trans_cnt)
		print ("The minimal support is: %s" % threshold)
		print ("The minimal transaction support is: %s" % trans_min)
			
		## Main:
		# generate initial candidates & do initial filter
		self.candidates = list (self.generate_init_candidates())
		print ("There are %s initial candidates." % len (self.candidates))
		freq_patterns = []
		new_freq_patterns = self.filter_candidates (trans_min)
		print ("The initial candidates have been filtered down to %s." % len (new_freq_patterns))
	
		while True:
			# is there anything left?
			if new_freq_patterns:
				freq_patterns = new_freq_patterns
			else:
				return freq_patterns
			
			# if any left, generate new candidates & filter
			self.candidates = self.generate_new_candidates ([x[0] for x in freq_patterns])
			print ("There are %s new candidates." % len (self.candidates))
			new_freq_patterns = self.filter_candidates (trans_min)
			print ("The candidates have been filtered down to %s." % len (new_freq_patterns))
			
### END ###

__version__ = '0.1'

### CONSTANTS & DEFINES

NULL_SYMBOL = 'X'

### CODE ###

def HaystackSearch(needle, haystack):
	"""
	Return the index of the needle in the haystack
	
	Parameters:
		needle: any iterable
		haystack: any other iterable
		
	Returns:
		the index of the start of needle or -1 if it is not found.
	
	Looking for a sub-list of a list is actually a tricky thing. This
	approach uses the Boyer-Moore-Horspool algorithm. Needle and haystack
	should be any iterable, as long as their elements are hashable.
	Example:
	
		>>> find ([1, 2], [1, 1, 2])
		1
		>>> find ((1, 2, 3), range (10))
		1
		>>> find ('gh', 'abcdefghi')
		6
		>>> find ([2, 3], [7, 8, 9])
		-1
	"""
	h = len (haystack)
	n = len (needle)
	skip = {needle[i]: n - i - 1 for i in range(n - 1)}
	i = n - 1
	while i < h:
		for j in range(n):
			if haystack[i - j] != needle[-j - 1]:
				i += skip.get(haystack[i], n)
				break
		else:
			return i - n + 1
	return -1