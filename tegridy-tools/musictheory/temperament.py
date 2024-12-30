#!/usr/bin/python 
#-*- coding: UTF-8 -*-
# temperament.py: Representing musical temperaments (and storing sequences
# associated with them).
#
# Copyright (c) 2008-2020 Peter Murphy <peterkmurphy@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The names of its contributors may not be used to endorse or promote 
#       products derived from this software without specific prior written 
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE CONTRIBUTORS ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Too many ideas, so let's work on the fundamentals.
# A musical temperament consists of the following things - keys, and their 
# positions within it. For example, we will start with the 12 chromatic
# scale. We only need to specify the seven natural keys, and their position
# within it. Doing anymore is overkill. For example, this code does not concern
# itself with the frequencies of notes in Hz.

import re;
import unittest;
from musutility import rotate, multislice, repseq;

# Unicode string constants for music notation.

M_SHARP = "\u266f";
M_NATURAL = "\u266e";
M_FLAT = "\u266d";

# Since this may not display on browsers, there are non-Unicode equivalents.

MNU_SHARP = "#";
MNU_FLAT = "b";
MNU_NATURAL = "n";

# This constant is used for parsing note names into keys.

RE_NOTEPARSE = "(?P<basenotename>[A-Z](#|b|n|" + M_FLAT + "|" + M_NATURAL \
    + "|"+ M_SHARP + ")*)";

# Constants used for the nseq_type arguments in noteseqs. See scales.py for
# more information.

NSEQ_SCALE = 0; # Used for specifying scales.
NSEQ_CHORD = 1; # Used for specifying chords.

# Several competing approaches were timed for character replacement in Python
# strings.  The quickest result came from using the replace function. For more
# information, see: 
# http://stackoverflow.com/questions/2484156/ \
#    is-str-replace-replace-ad-nauseam-a-standard-idiom-in-python

def un_unicode_accdtls(str_note):
    """ Turns all Unicode musical characters in str_note to their ASCII 
        equivalent, and returns the result.
    
        Note: the function is intended for viewing musical scales through
        deficient browsers like IE 6 and 7. Characters may appear as blocks
        in these environments.
    """
    return str_note.replace(M_FLAT, MNU_FLAT).replace(M_SHARP, 
        MNU_SHARP).replace(M_NATURAL, MNU_NATURAL);

# The following class seq_dict is used as a multiple key map to noteseq 
# instances, which are defined in scales.poy. However, seq_dict instances are
# stored with temperament objects, which is why they are stored here.

class seq_dict():
    """ The seq_dict class acts as a dictionary for noteseq instances. It
        allows users and programmers to look up noteseqs by name, by
        abbreviation, and by the sequence of integers representing the
        positions in the sequence. Each seq_dict should be assigned to one
        dictionary.
    """
    
    class nseqtype_ins_dict():
        """ The nseqtype_ins_dict class represents a subdictionary of noteseq
            instances sharing the same nseq_type. It is useful to split up
            noteseqs by nseq_type, lest property lookup return the "wrong"
            object. For example, a major 13 chord has the same pattern of 
            integral positions as the major scale; i.e., [0, 2, 4, 5, 7, 9, 
            11]. To look up the correct object of the two by pattern, the
            nseq_type must be specified.
        """
        def __init__(self, nseq_type):
            """ The constructed for nseqtype_ins_dict. This sets up seperate
                dictionaries for names, abbreviations and integral position
                dictionaries.
            """
            self.nseq_type = nseq_type;
            self.name_dict = {};
            self.abbrv_dict = {};
            self.seqpos_dict = {};
            
    def __init__(self, nseq_types, nseq_temp):
        """ The constructor for seq_dict. Arguments:
            nseq_types: a list of possible nseq_type values for lookups.
            nseq_temp: the associated temperament object.
        """
        self.nseq_temp = nseq_temp;
        self.nseqtype_maps = {};
        for i in nseq_types:
            self.nseqtype_maps[i] = seq_dict.nseqtype_ins_dict(i);

    def add_elem(self, elem, nseq_type, name_s, abbrv_s, seqpos):
        """ This adds an element to seq_dict. The arguments:
            elem: the element to add to seq_dict instance.
            nseq_type: the type of the elemenet (such as scale or chord).
            name_s: a string, or a sequence of strings. This provides names
                as keys that map onto elem.
            abbrv_s: a string, or a sequence of strings. This provides 
                abbreviations as keys that map onto elem.
            seqpos: a sequence. A tuple form will be used as a key that
                maps onto elem.
                
            If nseq_type is not associated with any of the sub-dictionaries in
            seq_dict, then this function exits.
        """
        if nseq_type not in self.nseqtype_maps:
            return False;
        sub_dictionary = self.nseqtype_maps[nseq_type];
        
# PKM - suspicious code. Replace:        
        
#        sub_dictionary.seqpos_dict[tuple(seqpos)] = elem;

# By:
        ourmod = self.nseq_temp.no_keys;
        ourtuple = tuple(sorted([i % ourmod for i in seqpos]));
        sub_dictionary.seqpos_dict[ourtuple] = elem;


        if isinstance(name_s, str):
            sub_dictionary.name_dict[name_s] = elem;
        else:
            for s in name_s:
                sub_dictionary.name_dict[s] = elem;
        if isinstance(abbrv_s, str):
            sub_dictionary.abbrv_dict[abbrv_s] = elem;
        else:
            for s in abbrv_s:
                sub_dictionary.abbrv_dict[s] = elem;
        return True;

# The next few functions, while relatively trivial, at least save some typing
# time in practice.

    def check_nseqby_subdict(self, nseq_type):
        """ Checks if there is a subdictionary associated with nseq_type. """
        return nseq_type in self.nseqtype_maps;

    def check_nseqby_name(self, nseq_type, name):
        """ Checks if there is a noteseq with a given name. """
        return name in self.nseqtype_maps[nseq_type].name_dict;
        
    def check_nseqby_abbrv(self, nseq_type, abbrv):
        """ Checks if there is a noteseq with a given abbreviation. """
        return abbrv in self.nseqtype_maps[nseq_type].abbrv_dict;

    def check_nseqby_seqpos(self, nseq_type, seqpos):
        """ Checks if there is a noteseq with a given sequence position. """
        sortedseqpos = tuple(sorted(seqpos));
        return sortedseqpos in self.nseqtype_maps[nseq_type].seqpos_dict;    

    def get_nseqby_name(self, name, nseq_type):
        """ Looks up a noteseq by name. """
        return self.nseqtype_maps[nseq_type].name_dict[name];
        
    def get_nseqby_abbrv(self, abbrv, nseq_type):
        """ Looks up a noteseq by abbreviation. """
        return self.nseqtype_maps[nseq_type].abbrv_dict[abbrv];

    def get_nseqby_seqpos(self, seqpos, nseq_type):
        """ Looks up a noteseq by sequence position. """
        sortedseqpos = tuple(sorted(seqpos));
        return self.nseqtype_maps[nseq_type].seqpos_dict[sortedseqpos];

class temperament():
    """ A musical temperament is used to define the possible keys in music, 
        and also the positions of keys relevant to each other. Our primary
        example is the western or chromatic temperament, with 12 possible keys.
        A temperament also defines which of the keys can be represented as
        naturals (like "C", "D" and "E"), and which need to be represented with
        accidentals (like "C#" and "Db".
        
        Doing anything more at the moment is overkill. For example, we are not
        intereted in the possible frequencies for notes of a given key.
    """
    
    def __init__(self, no_keys, nat_keys, nat_key_posn):
        """ Initialiser. It contains the following arguments.
            no_keys: the number of keys in the temperament. 
            nat_keys: an array consisting of the names of the natural 
                (unsharped or unflattened) keys in the temperament.
            nat_key_posn: the position of the natural keys in the temperament.
                These should correspond to the elements in nat_keys.
                Positions are calculated base zero. 
        """
        self.no_keys = no_keys;
        self.nat_keys = nat_keys;
        self.nat_key_posn = nat_key_posn;
        
# Extra variable: no_nat_keys: number of natural keys.        
        
        self.no_nat_keys = min(len(nat_keys), len(nat_key_posn));
        
# Extra variable: nat_key_pos_lookup: key -> position (e.g. A -> 9, C->0).        
        
        self.nat_key_pos_lookup = {};
        
# Extra variable: post_lookup_nat_key: position -> key/None (e.g., 9->A, 0->C).        
        
        self.pos_lookup_nat_key = {};
        
# Extra variable: nat_key_lookup_order: looks up order of nat_keys (e.g, C->0,
# D->1... B->6). Reverse translation available by self.nat_keys[order].

        self.nat_key_lookup_order = {};
        
        for i in range(self.no_nat_keys):
            self.nat_key_pos_lookup[nat_keys[i]] = nat_key_posn[i];
            self.pos_lookup_nat_key[nat_key_posn[i]] = nat_keys[i];
            self.nat_key_lookup_order[nat_keys[i]] = i;
        self.parsenote = re.compile(RE_NOTEPARSE);
        
#  Useful for dictionary lookup later.       
        
        self.seq_maps = seq_dict([NSEQ_SCALE, NSEQ_CHORD], self);

    def note_parse(self, key):
        """ Parses the name of a key into the tuple (natural key name,
            sharp_or_flat count). For example "C#" is parsed into ("C", 1),
            "Db" is parsed into ("D", -1), and "E" is parsed into ("E", 0).
            As the reader may gather, negative numbers are used for 
            flattened notes.
        """
        noteMatch = self.parsenote.match(key);
        if noteMatch == None:
            return None;
        noteGroup = noteMatch.group('basenotename');
        baseNote = noteGroup[0];
        flatSharpAdj = 0;
        for i in noteGroup[1:]:
            if i in (M_FLAT + MNU_FLAT):
                flatSharpAdj = flatSharpAdj - 1;
            if i in (M_SHARP + MNU_SHARP):
                flatSharpAdj = flatSharpAdj + 1;
        return (baseNote, flatSharpAdj,);            

    def get_pos_of_key(self, key):
        """ Returns the position of the key in the temperament. """
        key_parsed = self.note_parse(key); 
        return self.nat_key_pos_lookup[key_parsed[0]] + key_parsed[1];

    def get_key_of_pos(self, pos, desired_nat_note = None, 
            sharp_not_flat = True):
        """ Given a position in the temperament, this function attempts to
            return the "best" key name corresponding to it. This is not a 
            straight-forward reverse of get_pos_of_key, as there may be two
            different keynames for the same position, with one preferred. For
            example, "C# and "Db" are the same key, but "C# is preferred in an
            A major scale. Fortunately, arguments are provided to indicate the
            programmer's preference.
            pos: the position inside the temperament.
            desired_nat_note: the preferred natural key to start the key name.
                For example "C" makes the function return "C#" instead of "Db".
                If None, then the preference depends on...
            sharp_not_flat: if True, returns the sharpened accidental form 
                (e.g., "C#"); if False, returns the flattened accidental form
                (i.e., "Db").
        """

# accdtls: number of sharps or flats to add to the output string.

        if desired_nat_note:
            accdtls = pos - self.nat_key_pos_lookup[desired_nat_note];
            accdtls = (accdtls % self.no_keys);
            if accdtls > (self.no_keys / 2):
                accdtls = accdtls - self.no_keys;
            if accdtls > 0:
                return desired_nat_note + (M_SHARP * accdtls);
            elif accdtls < 0:
                return desired_nat_note + (M_FLAT * (-1 * accdtls));
            else:
                return desired_nat_note;
        else:
            accdtls = 0;
            if (pos % self.no_keys) in self.pos_lookup_nat_key:
                return self.pos_lookup_nat_key[pos % self.no_keys];
            elif sharp_not_flat:
                while accdtls < self.no_keys:
                    accdtls = accdtls - 1;
                    if ((pos + accdtls) % self.no_keys) in self.pos_lookup_nat_key:
                        return self.pos_lookup_nat_key[(pos + accdtls) 
                            % self.no_keys] + (M_SHARP * (-1 * accdtls));
            else:
                while accdtls < self.no_keys:
                    accdtls = accdtls + 1;
                    if ((pos + accdtls) % self.no_keys) in self.pos_lookup_nat_key:
                        return self.pos_lookup_nat_key[(pos + accdtls) 
                            % self.no_keys] + (M_FLAT * accdtls);
        return None;

    def get_note_sequence(self, key, pos_seq, nat_pos_seq = None, 
        sharp_not_flat = True):
        """ This function takes a key and a sequence of positions relative to
            it. It returns a sequence of notes. Arguments:
            key: the starting key; examples are "C", "C#" and "Db".
            pos_seq: a list of positions relative to it in the temperament 
                base 0). For example, in the Western/Chromatic temperament, a
                key of "C" and a sequence of [0, 1] returns ["C", "C#].
            nat_pos_seq: a list of numbers. These are used to calculate the
                desired natural notes produced from the corresponding positions
                in pos_seq. A number of 0 means to use the same natural note as
                in key, a number of 1 means using the next natural note, and so
                on. Examples for the Western/Chromatic scale:
                    get_note_sequence("C", [0, 1], [0, 0]) -> ["C", "C#"]
                    get_note_sequence("C", [0, 1], [0, 1]) -> ["C", "Dbb"]
                    get_note_sequence("C", [0, 1], [0, 6]) -> ["C", "B##"]
            sharp_not_flat: if nat_pos_seq is None, indicates whether notes 
                with accidentals are preferred to have sharps (True) or flats
                (False)
        """
        base_key_pos = self.get_pos_of_key(key); 
        result_pos_seq = [((base_key_pos + i) % self.no_keys) for i in pos_seq];
        if nat_pos_seq:
            result_seq = [];
            desired_nat_key = self.note_parse(key)[0];
            desired_nat_key_posn = self.nat_keys.index(desired_nat_key);
            for i in range(min(len(nat_pos_seq), len(result_pos_seq))):
                des_nat_note = self.nat_keys[(desired_nat_key_posn
                    + nat_pos_seq[i]) % self.no_nat_keys];
                des_key = self.get_key_of_pos(result_pos_seq[i], des_nat_note);
                result_seq.append(des_key);
            return(result_seq);
        else:
            return [self.get_key_of_pos(i, None, sharp_not_flat) for i in
                result_pos_seq];

    def get_keyseq_notes(self, note_seq):
        """ The reverse of get_note_sequence. This takes a sequence of notes
            (note_seq) and converts it into [base key, position sequence];
            this is returned by the function.
        """
        base_key = note_seq[0];
        base_pos = self.get_pos_of_key(base_key);
        pos_seq = [((self.get_pos_of_key(i) - base_pos) % self.no_keys) for
            i in note_seq];
        return [base_key, pos_seq];

# The next few functions are rip-offs of seq_dict functions.

    def add_elem(self, elem, nseq_type, name_s, abbrv_s, seqpos):
        """ This adds an element to the dictionary inside the temperament. The
            arguments:
            elem: the element to add  to the dictionary .
            nseq_type: the type of the elemenet (such as scale or chord).
            name_s: a string, or a sequence of strings. This provides names
                as keys that map onto elem.
            abbrv_s: a string, or a sequence of strings. This provides 
                abbreviations as keys that map onto elem.
            seqpos: a sequence. A tuple form will be used as a key that
                maps onto elem.
                
            If nseq_type is not associated with any of the sub-dictionaries in
            the dictionary in the temperament, then this function exits.
        """
        return self.seq_maps.add_elem(elem, nseq_type, name_s, abbrv_s, 
            seqpos);

    def check_nseqby_subdict(self, nseq_type):
        """ Checks if there is a subdictionary associated with nseq_type. """
        return self.seq_maps.check_nseqby_subdict(nseq_type);

    def check_nseqby_name(self, nseq_type, name):
        """ Checks if there is a noteseq with a given name. """
        return self.seq_maps.check_nseqby_name(nseq_type, name);
        
    def check_nseqby_abbrv(self, nseq_type, abbrv):
        """ Checks if there is a noteseq with a given abbreviation. """
        return self.seq_maps.check_nseqby_abbrv(nseq_type, abbrv);

    def check_nseqby_seqpos(self, nseq_type, seqpos):
        """ Checks if there is a noteseq with a given sequence position. """
        return self.seq_maps.check_nseqby_seqpos(nseq_type, seqpos); 

    def get_nseqby_name(self, name, nseq_type):
        """ Looks up a noteseq (or anything else) by name. """
        if self.seq_maps.check_nseqby_name(nseq_type, name):
            return self.seq_maps.get_nseqby_name(name, nseq_type);
        else:
            return None;
        
    def get_nseqby_abbrv(self, abbrv, nseq_type):
        """ Looks up a noteseq (or anything else) by abbreviation. """
        if self.seq_maps.check_nseqby_abbrv(nseq_type, abbrv):
            return self.seq_maps.get_nseqby_abbrv(abbrv, nseq_type);
        else:
            return None;

    def get_nseqby_seqpos(self, seqpos, nseq_type):
        """ Looks up a noteseq (or anything else) by sequence position. """
        if self.seq_maps.check_nseqby_seqpos(nseq_type, seqpos):
            return self.seq_maps.get_nseqby_seqpos(seqpos, nseq_type);
        else:
            return None;


# We immediately use this class to define a Western or Chromatic temperament
# object. Apart from useful for understanding the class, the object is used in
# testing.

CHROM_NAT_NOTES = ["C", "D", "E", "F", "G", "A", "B"];
CHROM_NAT_NOTE_POS = [0, 2, 4, 5, 7, 9, 11];
CHROM_SIZE = 12;
WestTemp = temperament(CHROM_SIZE, CHROM_NAT_NOTES, CHROM_NAT_NOTE_POS);

