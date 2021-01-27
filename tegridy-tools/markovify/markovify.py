# -*- coding: utf-8 -*-
"""markovify.py"""

################################################################################
#
# Markovify stand-alone Python module (version 1.0)
#
# Based upon markovify package by @jsvine of GitHub:
# https://github.com/jsvine/markovify
#
# Used src code as of Dec 10, 2020 (c8eb4cf)
#
# Project Los Angeles
# Tegridy Code 2021
#
################################################################################
#
# Critical requirements/dependency:
#
# pip install unidecode
#
################################################################################

import random
import operator
import bisect
import json
import copy
import re

from unidecode import unidecode

################################################################################

# chain.py module

# Python3 compatibility
try: # pragma: no cover
    basestring
except NameError: # pragma: no cover
    basestring = str

BEGIN = "___BEGIN__"
END = "___END__"

def accumulate(iterable, func=operator.add):
    """
    Cumulative calculations. (Summation, by default.)
    Via: https://docs.python.org/3/library/itertools.html#itertools.accumulate
    """
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def compile_next(next_dict):
    words = list(next_dict.keys())
    cff = list(accumulate(next_dict.values()))
    return [words, cff]

class Chain(object):
    """
    A Markov chain representing processes that have both beginnings and ends.
    For example: Sentences.
    """
    def __init__(self, corpus, state_size, model=None):
        """
        `corpus`: A list of lists, where each outer list is a "run"
        of the process (e.g., a single sentence), and each inner list
        contains the steps (e.g., words) in the run. If you want to simulate
        an infinite process, you can come very close by passing just one, very
        long run.
        `state_size`: An integer indicating the number of items the model
        uses to represent its state. For text generation, 2 or 3 are typical.
        """
        self.state_size = state_size
        self.model = model or self.build(corpus, self.state_size)
        self.compiled = (len(self.model) > 0) and (type(self.model[tuple([BEGIN]*state_size)]) == list)
        if not self.compiled:
            self.precompute_begin_state()

    def compile(self, inplace = False):
        if self.compiled:
            if inplace: return self
            return Chain(None, self.state_size, model = copy.deepcopy(self.model))
        mdict = { state: compile_next(next_dict) for (state, next_dict) in self.model.items() }
        if not inplace: return Chain(None, self.state_size, model = mdict)
        self.model = mdict
        self.compiled = True
        return self

    def build(self, corpus, state_size):
        """
        Build a Python representation of the Markov model. Returns a dict
        of dicts where the keys of the outer dict represent all possible states,
        and point to the inner dicts. The inner dicts represent all possibilities
        for the "next" item in the chain, along with the count of times it
        appears.
        """

        # Using a DefaultDict here would be a lot more convenient, however the memory
        # usage is far higher.
        model = {}

        for run in corpus:
            items = ([ BEGIN ] * state_size) + run + [ END ]
            for i in range(len(run) + 1):
                state = tuple(items[i:i+state_size])
                follow = items[i+state_size]
                if state not in model:
                    model[state] = {}

                if follow not in model[state]:
                    model[state][follow] = 0

                model[state][follow] += 1
        return model

    def precompute_begin_state(self):
        """
        Caches the summation calculation and available choices for BEGIN * state_size.
        Significantly speeds up chain generation on large corpora. Thanks, @schollz!
        """
        begin_state = tuple([ BEGIN ] * self.state_size)
        choices, cumdist = compile_next(self.model[begin_state])
        self.begin_cumdist = cumdist
        self.begin_choices = choices

    def move(self, state):
        """
        Given a state, choose the next item at random.
        """
        if self.compiled:
            choices, cumdist = self.model[state]
        elif state == tuple([ BEGIN ] * self.state_size):
            choices = self.begin_choices
            cumdist = self.begin_cumdist
        else:
            choices, weights = zip(*self.model[state].items())
            cumdist = list(accumulate(weights))
        r = random.random() * cumdist[-1]
        selection = choices[bisect.bisect(cumdist, r)]
        return selection

    def gen(self, init_state=None):
        """
        Starting either with a naive BEGIN state, or the provided `init_state`
        (as a tuple), return a generator that will yield successive items
        until the chain reaches the END state.
        """
        state = init_state or (BEGIN,) * self.state_size
        while True:
            next_word = self.move(state)
            if next_word == END: break
            yield next_word
            state = tuple(state[1:]) + (next_word,)

    def walk(self, init_state=None):
        """
        Return a list representing a single run of the Markov model, either
        starting with a naive BEGIN state, or the provided `init_state`
        (as a tuple).
        """
        return list(self.gen(init_state))

    def to_json(self):
        """
        Dump the model as a JSON object, for loading later.
        """
        return json.dumps(list(self.model.items()))

    @classmethod
    def from_json(cls, json_thing):
        """
        Given a JSON object or JSON string that was created by `self.to_json`,
        return the corresponding markovify.Chain.
        """

        if isinstance(json_thing, basestring):
            obj = json.loads(json_thing)
        else:
            obj = json_thing

        if isinstance(obj, list):
            rehydrated = dict((tuple(item[0]), item[1]) for item in obj)
        elif isinstance(obj, dict):
            rehydrated = obj
        else:
            raise ValueError("Object should be dict or list")

        state_size = len(list(rehydrated.keys())[0])

        inst = cls(None, state_size, rehydrated)
        return inst

################################################################################

# splitters.py module

uppercase_letter_pat = re.compile(r"^[A-Z]$", re.UNICODE)
initialism_pat = re.compile(r"^[A-Za-z0-9]{1,2}(\.[A-Za-z0-9]{1,2})+\.$", re.UNICODE)

# States w/ with thanks to https://github.com/unitedstates/python-us
# Titles w/ thanks to https://github.com/nytimes/emphasis and @donohoe
abbr_capped = "|".join([
    "ala|ariz|ark|calif|colo|conn|del|fla|ga|ill|ind|kan|ky|la|md|mass|mich|minn|miss|mo|mont|neb|nev|okla|ore|pa|tenn|vt|va|wash|wis|wyo", # States
    "u.s",
    "mr|ms|mrs|msr|dr|gov|pres|sen|sens|rep|reps|prof|gen|messrs|col|sr|jf|sgt|mgr|fr|rev|jr|snr|atty|supt", # Titles
    "ave|blvd|st|rd|hwy", # Streets
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec", # Months
]).split("|")

abbr_lowercase = "etc|v|vs|viz|al|pct".split("|")

def is_abbreviation(dotted_word):
    clipped = dotted_word[:-1]
    if re.match(uppercase_letter_pat, clipped[0]):
        if len(clipped) == 1: # Initial
            return True
        elif clipped.lower() in abbr_capped:
            return True
        else:
            return False
    else:
        if clipped in abbr_lowercase: return True
        else: return False

def is_sentence_ender(word):
    if re.match(initialism_pat, word) is not None: return False
    if word[-1] in [ "?", "!" ]:
        return True
    if len(re.sub(r"[^A-Z]", "", word)) > 1:
        return True
    if word[-1] == "." and (not is_abbreviation(word)):
        return True
    return False

def split_into_sentences(text):
    potential_end_pat = re.compile(r"".join([
        r"([\w\.'’&\]\)]+[\.\?!])", # A word that ends with punctuation
        r"([‘’“”'\"\)\]]*)", # Followed by optional quote/parens/etc
        r"(\s+(?![a-z\-–—]))", # Followed by whitespace + non-(lowercase or dash)
        ]), re.U)
    dot_iter = re.finditer(potential_end_pat, text)
    end_indices = [ (x.start() + len(x.group(1)) + len(x.group(2)))
        for x in dot_iter
        if is_sentence_ender(x.group(1)) ]
    spans = zip([None] + end_indices, end_indices + [None])
    sentences = [ text[start:end].strip() for start, end in spans ]
    return sentences

################################################################################

# text.py module

DEFAULT_MAX_OVERLAP_RATIO = 0.7
DEFAULT_MAX_OVERLAP_TOTAL = 15
DEFAULT_TRIES = 10

class ParamError(Exception):
    pass

class Text(object):

    reject_pat = re.compile(r"(^')|('$)|\s'|'\s|[\"(\(\)\[\])]")

    def __init__(self, input_text, state_size=2, chain=None, parsed_sentences=None, retain_original=True, well_formed=True, reject_reg=''):
        """
        input_text: A string.
        state_size: An integer, indicating the number of words in the model's state.
        chain: A trained markovify.Chain instance for this text, if pre-processed.
        parsed_sentences: A list of lists, where each outer list is a "run"
              of the process (e.g. a single sentence), and each inner list
              contains the steps (e.g. words) in the run. If you want to simulate
              an infinite process, you can come very close by passing just one, very
              long run.
        retain_original: Indicates whether to keep the original corpus.
        well_formed: Indicates whether sentences should be well-formed, preventing
              unmatched quotes, parenthesis by default, or a custom regular expression
              can be provided.
        reject_reg: If well_formed is True, this can be provided to override the
              standard rejection pattern.
        """

        self.well_formed = well_formed
        if well_formed and reject_reg != '':
            self.reject_pat = re.compile(reject_reg)

        can_make_sentences = parsed_sentences is not None or input_text is not None
        self.retain_original = retain_original and can_make_sentences
        self.state_size = state_size

        if self.retain_original:
            self.parsed_sentences = parsed_sentences or list(self.generate_corpus(input_text))

            # Rejoined text lets us assess the novelty of generated sentences
            self.rejoined_text = self.sentence_join(map(self.word_join, self.parsed_sentences))
            self.chain = chain or Chain(self.parsed_sentences, state_size)
        else:
            if not chain:
                parsed = parsed_sentences or self.generate_corpus(input_text)
            self.chain = chain or Chain(parsed, state_size)

    def compile(self, inplace = False):
        if inplace:
            self.chain.compile(inplace = True)
            return self
        cchain = self.chain.compile(inplace = False)
        psent = None
        if hasattr(self, 'parsed_sentences'):
            psent = self.parsed_sentences
        return Text(None, \
                    state_size = self.state_size, \
                    chain = cchain, \
                    parsed_sentences = psent, \
                    retain_original = self.retain_original, \
                    well_formed = self.well_formed, \
                    reject_reg = self.reject_pat)

    def to_dict(self):
        """
        Returns the underlying data as a Python dict.
        """
        return {
            "state_size": self.state_size,
            "chain": self.chain.to_json(),
            "parsed_sentences": self.parsed_sentences if self.retain_original else None
        }

    def to_json(self):
        """
        Returns the underlying data as a JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, obj, **kwargs):
        return cls(
            None,
            state_size=obj["state_size"],
            chain=Chain.from_json(obj["chain"]),
            parsed_sentences=obj.get("parsed_sentences")
        )

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))

    def sentence_split(self, text):
        """
        Splits full-text string into a list of sentences.
        """
        return split_into_sentences(text)

    def sentence_join(self, sentences):
        """
        Re-joins a list of sentences into the full text.
        """
        return " ".join(sentences)

    word_split_pattern = re.compile(r"\s+")
    def word_split(self, sentence):
        """
        Splits a sentence into a list of words.
        """
        return re.split(self.word_split_pattern, sentence)

    def word_join(self, words):
        """
        Re-joins a list of words into a sentence.
        """
        return " ".join(words)

    def test_sentence_input(self, sentence):
        """
        A basic sentence filter. The default rejects sentences that contain
        the type of punctuation that would look strange on its own
        in a randomly-generated sentence.
        """
        if len(sentence.strip()) == 0: return False
        # Decode unicode, mainly to normalize fancy quotation marks
        if sentence.__class__.__name__ == "str": # pragma: no cover
            decoded = sentence
        else: # pragma: no cover
            decoded = unidecode(sentence)
        # Sentence shouldn't contain problematic characters
        if self.well_formed and self.reject_pat.search(decoded): return False
        return True

    def generate_corpus(self, text):
        """
        Given a text string, returns a list of lists; that is, a list of
        "sentences," each of which is a list of words. Before splitting into
        words, the sentences are filtered through `self.test_sentence_input`
        """
        if isinstance(text, str):
            sentences = self.sentence_split(text)
        else:
            sentences = []
            for line in text:
                sentences += self.sentence_split(line)
        passing = filter(self.test_sentence_input, sentences)
        runs = map(self.word_split, passing)
        return runs

    def test_sentence_output(self, words, max_overlap_ratio, max_overlap_total):
        """
        Given a generated list of words, accept or reject it. This one rejects
        sentences that too closely match the original text, namely those that
        contain any identical sequence of words of X length, where X is the
        smaller number of (a) `max_overlap_ratio` (default: 0.7) of the total
        number of words, and (b) `max_overlap_total` (default: 15).
        """
        # Reject large chunks of similarity
        overlap_ratio = int(round(max_overlap_ratio * len(words)))
        overlap_max = min(max_overlap_total, overlap_ratio)
        overlap_over = overlap_max + 1
        gram_count = max((len(words) - overlap_max), 1)
        grams = [ words[i:i+overlap_over] for i in range(gram_count) ]
        for g in grams:
            gram_joined = self.word_join(g)
            if gram_joined in self.rejoined_text:
                return False
        return True

    def make_sentence(self, init_state=None, **kwargs):
        """
        Attempts `tries` (default: 10) times to generate a valid sentence,
        based on the model and `test_sentence_output`. Passes `max_overlap_ratio`
        and `max_overlap_total` to `test_sentence_output`.
        If successful, returns the sentence as a string. If not, returns None.
        If `init_state` (a tuple of `self.chain.state_size` words) is not specified,
        this method chooses a sentence-start at random, in accordance with
        the model.
        If `test_output` is set as False then the `test_sentence_output` check
        will be skipped.
        If `max_words` or `min_words` are specified, the word count for the sentence will be
        evaluated against the provided limit(s).
        """
        tries = kwargs.get('tries', DEFAULT_TRIES)
        mor = kwargs.get('max_overlap_ratio', DEFAULT_MAX_OVERLAP_RATIO)
        mot = kwargs.get('max_overlap_total', DEFAULT_MAX_OVERLAP_TOTAL)
        test_output = kwargs.get('test_output', True)
        max_words = kwargs.get('max_words', None)
        min_words = kwargs.get('min_words', None)

        if init_state != None:
            prefix = list(init_state)
            for word in prefix:
                if word == BEGIN:
                    prefix = prefix[1:]
                else:
                    break
        else:
            prefix = []

        for _ in range(tries):
            words = prefix + self.chain.walk(init_state)
            if (max_words != None and len(words) > max_words) or (min_words != None and len(words) < min_words):
                continue # pragma: no cover # see https://github.com/nedbat/coveragepy/issues/198
            if test_output and hasattr(self, "rejoined_text"):
                if self.test_sentence_output(words, mor, mot):
                    return self.word_join(words)
            else:
                return self.word_join(words)
        return None

    def make_short_sentence(self, max_chars, min_chars=0, **kwargs):
        """
        Tries making a sentence of no more than `max_chars` characters and optionally
        no less than `min_chars` characters, passing **kwargs to `self.make_sentence`.
        """
        tries = kwargs.get('tries', DEFAULT_TRIES)

        for _ in range(tries):
            sentence = self.make_sentence(**kwargs)
            if sentence and len(sentence) <= max_chars and len(sentence) >= min_chars:
                return sentence

    def make_sentence_with_start(self, beginning, strict=True, **kwargs):
        """
        Tries making a sentence that begins with `beginning` string,
        which should be a string of one to `self.state` words known
        to exist in the corpus.
        
        If strict == True, then markovify will draw its initial inspiration
        only from sentences that start with the specified word/phrase.
        If strict == False, then markovify will draw its initial inspiration
        from any sentence containing the specified word/phrase.
        **kwargs are passed to `self.make_sentence`
        """
        split = tuple(self.word_split(beginning))
        word_count = len(split)

        if word_count == self.state_size:
            init_states = [ split ]

        elif word_count > 0 and word_count < self.state_size:
            if strict:
                init_states = [ (BEGIN,) * (self.state_size - word_count) + split ]

            else:
                init_states = [ key for key in self.chain.model.keys()
                    # check for starting with begin as well ordered lists
                    if tuple(filter(lambda x: x != BEGIN, key))[:word_count] == split ]

                random.shuffle(init_states)
        else:
            err_msg = "`make_sentence_with_start` for this model requires a string containing 1 to {0} words. Yours has {1}: {2}".format(self.state_size, word_count, str(split))
            raise ParamError(err_msg)

        for init_state in init_states:
            output = self.make_sentence(init_state, **kwargs)
            if output is not None:
                return output
        err_msg = "`make_sentence_with_start` can't find sentence beginning with {0}".format(beginning)
        raise ParamError(err_msg)

    @classmethod
    def from_chain(cls, chain_json, corpus=None, parsed_sentences=None):
        """
        Init a Text class based on an existing chain JSON string or object
        If corpus is None, overlap checking won't work.
        """
        chain = Chain.from_json(chain_json)
        return cls(corpus or None, parsed_sentences=parsed_sentences, state_size=chain.state_size, chain=chain)


class NewlineText(Text):
    """
    A (usable) example of subclassing markovify.Text. This one lets you markovify
    text where the sentences are separated by newlines instead of ". "
    """
    def sentence_split(self, text):
        return re.split(r"\s*\n\s*", text)

################################################################################

# utils.py module

def get_model_dict(thing):
    if isinstance(thing, Chain):
        if thing.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.model
    if isinstance(thing, Text):
        if thing.chain.compiled:
            raise ValueError("Not implemented for compiled markovify.Chain")
        return thing.chain.model
    if isinstance(thing, list):
        return dict(thing)
    if isinstance(thing, dict):
        return thing

    raise ValueError("`models` should be instances of list, dict, markovify.Chain, or markovify.Text")

def combine(models, weights=None):
    if weights == None:
        weights = [ 1 for _ in range(len(models)) ]

    if len(models) != len(weights):
        raise ValueError("`models` and `weights` lengths must be equal.")

    model_dicts = list(map(get_model_dict, models))
    state_sizes = [ len(list(md.keys())[0])
        for md in model_dicts ]

    if len(set(state_sizes)) != 1:
        raise ValueError("All `models` must have the same state size.")

    if len(set(map(type, models))) != 1:
        raise ValueError("All `models` must be of the same type.")

    c = {}

    for m, w in zip(model_dicts, weights):
        for state, options in m.items():
            current = c.get(state, {})
            for subseq_k, subseq_v in options.items():
                subseq_prev = current.get(subseq_k, 0)
                current[subseq_k] = subseq_prev + (subseq_v * w)
            c[state] = current

    ret_inst = models[0]

    if isinstance(ret_inst, Chain):
        return Chain.from_json(c)
    if isinstance(ret_inst, Text):
        if any(m.retain_original for m in models):
            combined_sentences = []
            for m in models:
                if m.retain_original:
                    combined_sentences += m.parsed_sentences
            return ret_inst.from_chain(c, parsed_sentences=combined_sentences)
        else:
            return ret_inst.from_chain(c)
    if isinstance(ret_inst, list):
        return list(c.items())
    if isinstance(ret_inst, dict):
        return c