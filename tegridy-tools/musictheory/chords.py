#!/usr/bin/python 
#-*- coding: UTF-8 -*-
# chords.py: Defines (and contains) the classes for chords.
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

import unittest;

from musutility import enl_seq;
from temperament import temperament, WestTemp, seq_dict, NSEQ_SCALE, \
    NSEQ_CHORD, M_SHARP, M_FLAT;
from scales import noteseq, noteseq_scale;    

class noteseq_chord(noteseq):
    """ A specialisation of noteseq used exclusively for defining chords -
        especially chords associated with heptatonic/diotonic scales.
    """
    def __init__(self, nseq_name, nseq_temp, nseq_posn, nseq_nat_posns, 
        nseq_abbrev):
        """ Arguments:
            nseq_name: name of chord.
            nseq_temp: temperament for chord.
            nseq_posn: position of notes in chord.
            nseq_nat_posns: natural note positions for notes in scales.
            nseq_abbrev: main abbreviation for chord.
        """
        noteseq.__init__(self, nseq_name, NSEQ_CHORD, nseq_temp, nseq_posn, 
            nseq_nat_posns, nseq_abbrev); 

    def __str__(self):
        return str(self.nseq_name)+":"+str(self.nseq_abbrev)+":" \
            + str(self.nseq_posn);

# This dictionary contains a list of different nseq_nat_posns values for 
# different types of chords in the Western tradition. Note that all arrays are
# stored base 0, not base1, so a "seventh" chord is stored as [0, 2, 4, 6] -
# not [1, 3, 5, 7]. 

CHORDTYPE_DICT = {"Triad": [0, 2, 4],
    "Seventh": [0, 2, 4, 6],
    "Ninth": [0, 2, 4, 6, 8],
    "Eleventh": [0, 2, 4, 6, 8, 10],
    "Thirteenth": [0, 2, 4, 6, 8, 10, 12],
    "Added Ninth": [0, 2, 4, 8],
    "Suspended": [0, 3, 4],
    "Suspended Seventh": [0, 3, 4, 6],
    "Sixth": [0, 2, 4, 5],
    "Sixth/Ninth": [0, 2, 4, 5, 8],
    "Added Eleventh": [0, 2, 4, 10],
    "Fifth": [0, 4]};


def generate_west_chords():
    """ This function generates most of the chords in the Western tradition
        (and some that are not really chords at all).
    """
    chordseq = [];
    bases = [[0, 3], [0, 4]];
    triads = enl_seq(bases, [6, 7, 8]);
    
# PKM2014 - add two pseudo suspended chords         
    
    suspendeds = [[0, 5, 6], [0, 5, 7], [0, 5, 8], [0, 6, 7], [0, 6, 8]];
    sixths = enl_seq(triads, [8, 9]);
    sevenths = enl_seq(triads, [10, 11]);
    suspended_sevenths = enl_seq(suspendeds, [9, 10, 11]);
    ninths = enl_seq(sevenths, [13, 14, 15]);
    elevenths = enl_seq(ninths, [17, 18]);
    thirteenths = enl_seq(elevenths, [20, 21]);
    add_ninths = enl_seq(triads, [13, 14, 15]);
    six_ninths = enl_seq(sixths, [13, 14, 15]);
    sevenths_and_above = sevenths + ninths + elevenths + thirteenths + \
        suspended_sevenths;

# We add the power chords.

    chordseq.append(noteseq_chord("Power Fifth", WestTemp, [0, 7], 
        CHORDTYPE_DICT["Fifth"], "5"));
    chordseq.append(noteseq_chord("Tritone", WestTemp, [0, 6], 
        CHORDTYPE_DICT["Fifth"], "T"));
    chordseq.append(noteseq_chord("Power Sharp Fifth", WestTemp, [0, 8], 
        CHORDTYPE_DICT["Fifth"], "+5"));


# We add the diminished chords, as they take special handling.

    chordseq.append(noteseq_chord("Diminish 7th", WestTemp, [0, 3, 6, 9], 
        CHORDTYPE_DICT["Seventh"], "dim7"));
    chordseq.append(noteseq_chord("Diminish 9th", WestTemp, [0, 3, 6, 9, 13], 
        CHORDTYPE_DICT["Ninth"], "dim9"));
    chordseq.append(noteseq_chord("Diminish 11th", WestTemp, 
        [0, 3, 6, 9, 13, 16], CHORDTYPE_DICT["Eleventh"], "dim11"));
    chordseq.append(noteseq_chord("Diminish 13th", WestTemp, 
        [0, 3, 6, 9, 13, 16, 20], CHORDTYPE_DICT["Thirteenth"], "dim13"));

# We loop over the triads and the added ninths together. For this reason, we 
# set up some maps for the ease of iteration.

    triad_names = {3: {6:"Diminish 5th", 7:"Minor", 8:"Minor Sharp 5th"}, 
        4: { 6:"Major Flat 5th", 7:"Major", 8:"Augment"}}; 
    triad_abbrv = {3: {6:"dim5", 7:"min", 8:"min+5"}, 
        4: { 6:"-5", 7:"maj", 8:"+"}}; 
    add9_names = {13:" Add Flat 9th", 14: " Add 9th", 
        15: " Add Sharp 9th"};
    add9_abbrv = {13:" add-9", 14: " add9", 15: " add+9"};

    add11_names = {17:" Add 11th", 18: " Add Sharp 11th"};
    add11_abbrv = {17:" add11", 18: " add+11"};

    add6_names = {20: " Add Flat 6th", 21:" Add 6th"};
    add6_abbrv = {20: " add-6", 21:" add6"};


    for i in [3, 4]:
        for j in [6, 7, 8]:
            our_tname = triad_names[i][j];
            our_tabbr = triad_abbrv[i][j];
            chordseq.append(noteseq_chord(our_tname, WestTemp, 
                [0, i, j], CHORDTYPE_DICT["Triad"], our_tabbr));
            for k in [13, 14, 15]:
                if i == 3 and k == 15:
                    pass;
                else:
                    chordseq.append(noteseq_chord(our_tname + add9_names[k],
                        WestTemp, [0, i, j, k], CHORDTYPE_DICT["Added Ninth"],
                        our_tabbr+add9_abbrv[k]));
            for k in [17, 18]:
                if k == 18 and j == 6:
                    pass;
                else:
                    chordseq.append(noteseq_chord(our_tname + add11_names[k],
                        WestTemp, [0, i, j, k], CHORDTYPE_DICT["Added Eleventh"],
                        our_tabbr+add11_abbrv[k]));
            for k in [20, 21]:
                if i == 3 and j == 6 and k == 21:
                    pass; # Pattern already taken by Diminished 7 chord.
                else:
                    chordseq.append(noteseq_chord(our_tname + add6_names[k],
                        WestTemp, [0, i, j, k], CHORDTYPE_DICT["Sixth"],
                        our_tabbr+add6_abbrv[k]));

                for l in [13, 14, 15]:
                    if i == 3 and l == 15:
                        pass;
                    else:
                        chordseq.append(noteseq_chord(
                            our_tname + add6_names[k] + add9_names[l],
                            WestTemp, [0, i, j, k, l], 
                            CHORDTYPE_DICT["Sixth/Ninth"],
                            our_tabbr+add6_abbrv[k]+add9_abbrv[l]));


# Then we add the suspended chords. They are treated differently from other chords,
# as one can't put 9th, 11th or 13th on there.

    chordseq.append(noteseq_chord("Suspend Flat 5th", WestTemp, [0, 5, 6], 
        CHORDTYPE_DICT["Suspended"], "sus-5"));
    chordseq.append(noteseq_chord("Suspend", WestTemp, [0, 5, 7], 
        CHORDTYPE_DICT["Suspended"], "sus"));
    chordseq.append(noteseq_chord("Suspend Sharp 5th", WestTemp, [0, 5, 8], 
        CHORDTYPE_DICT["Suspended"], "sus+5"));
        
# PKM2014 - add two pseudo suspended chords         
        
    chordseq.append(noteseq_chord("Suspend Sharp 4th", WestTemp, [0, 6, 7], 
        CHORDTYPE_DICT["Suspended"], "sus+4"));
    chordseq.append(noteseq_chord("Suspend Sharp 4th Sharp 5th", WestTemp, [0, 6, 8], 
        CHORDTYPE_DICT["Suspended"], "sus+4+5"));        

# Afterwards, we add the 7th (including sus 7th), 9th, 11th and 13th chords.     
    
    for i in sevenths_and_above:
        suspendsharpfourth = False
        abbrev = "";
        name = "";
        if 5 in i:
            abbrev += "sus / ";
            name += "Suspend "

# PKM2014 - add code for two pseudo suspended 7th chords         

        if len(i) == 4 and 6 in i and ((7 in i) or (8 in i)):
            abbrev += "sus+4 / ";
            name += "Suspend Sharp 4th "
            suspendsharpfourth = True
        if 3 in i:
            if 11 in i:
                abbrev += "maj / min";
                name +="Major / Minor "
            else:
                abbrev += "min";
                name += "Minor ";
        else:
            if 11 in i:
                abbrev += "maj";
                name += "Major "
                
            #PKM2014 - this is for suspended diminished chords.    
                
            elif 9 in i and len(i) == 4 and ((5 in i) or (6 in i and ((7 in i) or (8 in i)))):
                abbrev += "dim";
                name += "Diminish "                
        if 21 in i:
            abbrev += "13";
            name += "13th "
        elif 17 in i:
            abbrev += "11";
            name += "11th "
        elif 14 in i:
            abbrev += "9";
            name += "9th "
        else:
            abbrev += "7";
            name += "7th "
        if 20 in i:
            abbrev += "-13";
            name += "Flat 13th "
        if 18 in i:
            abbrev += "+11";
            name += "Sharp 11th "
        if 15 in i:
            abbrev += "+9";
            name += "Sharp 9th "
        if 13 in i:
            abbrev += "-9";
            name += "Flat 9th "
        if 8 in i:
            abbrev += "+5";
            name += "Sharp 5th "
        if 6 in i and not suspendsharpfourth:
            abbrev += "-5";
            name += "Flat 5th ";
            
# We eliminate chords that consist of the same note seperated by an octave!            
            
        if 3 in i and 15 in i:
            pass;
        elif 5 in i and 17 in i:
            pass;
        elif 6 in i and 18 in i:
            pass;
        elif 8 in i and 20 in i:
            pass;              
        else:
            name = name.rstrip();
            if 5 in i:
                our_chordtype = CHORDTYPE_DICT["Suspended Seventh"];
            elif 20 in i or 21 in i:
                our_chordtype = CHORDTYPE_DICT["Thirteenth"];
            elif 17 in i or 18 in i:
                our_chordtype = CHORDTYPE_DICT["Eleventh"];                
            elif 13 in i or 14 in i or 15 in i:
                our_chordtype = CHORDTYPE_DICT["Ninth"];
            else:
                our_chordtype = CHORDTYPE_DICT["Seventh"];
            chordseq.append(noteseq_chord(name, WestTemp, i, 
                our_chordtype, abbrev));

    return chordseq;
    
ourchords = generate_west_chords();  

# 2020 - the following testing code is commented out, as I don't know
# why it was put together in the first place.
