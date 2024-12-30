#!/usr/bin/python 
#-*- coding: UTF-8 -*-
# chordgenerator.py: Generating tables of chords associated with scales.
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
#
# This software was originally written by Peter Murphy (2008). It has been
# updated in 2011 to use the assoicated musictheory classes. Updated for 2011.
#
# The goal of this utility is to find all the possible chords (major, minor, 
# 7th) in common diatonic scales (such as Major and Harmonic Minor).  The
# results are tables that can be displayed in an HTML file.

import copy, codecs
from musutility import seqtostr;
from temperament import WestTemp, temperament, WestTemp, seq_dict,\
    NSEQ_SCALE, NSEQ_CHORD, M_SHARP, M_FLAT;
from scales import MajorScale, MelMinorScale, HarmMinorScale,\
    HarmMajorScale;
from chords import CHORDTYPE_DICT;

# Used to give a full list of chord types.

CHORDTYPE_ARRAY = ["Fifth", "Triad", "Seventh", "Ninth", "Eleventh", "Thirteenth",
    "Added Ninth", "Suspended", "Suspended Seventh", "Sixth", "Sixth/Ninth",
    "Added Eleventh"];

# Roman numerals are used for the table headers.

# Following routine comes from "Roman Numerals (Python recipe)".
# Author: Paul Winkler on Sun, 14 Oct 2001. See:
# http://code.activestate.com/recipes/81611-roman-numerals/

def int_to_roman(input):
   """
   Convert an integer to Roman numerals.

   Examples:
   >>> int_to_roman(0)
   Traceback (most recent call last):
   ValueError: Argument must be between 1 and 3999

   >>> int_to_roman(-1)
   Traceback (most recent call last):
   ValueError: Argument must be between 1 and 3999

   >>> int_to_roman(1.5)
   Traceback (most recent call last):
   TypeError: expected integer, got <type 'float'>

   >>> for i in range(1, 21): print int_to_roman(i)
   ...
   I
   II
   III
   IV
   V
   VI
   VII
   VIII
   IX
   X
   XI
   XII
   XIII
   XIV
   XV
   XVI
   XVII
   XVIII
   XIX
   XX
   >>> print int_to_roman(2000)
   MM
   >>> print int_to_roman(1999)
   MCMXCIX
   """
   #if type(input) != type(1):
   #   raise TypeError, "expected integer, got %s" % type(input)
   #if not 0 < input < 4000:
   #   raise ValueError, "Argument must be between 1 and 3999"  
   ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
   nums = ('M',  'CM', 'D', 'CD','C', 'XC','L','XL','X','IX','V','IV','I')
   result = ""
   for i in range(len(ints)):
      count = int(input / ints[i])
      result += nums[i] * count
      input -= ints[i] * count
   return result

def make_roman_numeral_list(num):
    """ Make a sequence of roman numerals from 1 to num. """
    return([int_to_roman(i) for i in range(1, num + 1)]);

# In Django, do we print out abbreviations or full chord names?

PRINT_ABBRV = 0;
PRINT_FNAME = 1;
PRINT_BOTH = 2;

class scale_chords():
    """ Represents all the chords associated with a scale. This is
        used to make a tabular representation.
    """
    def __init__ (self, full_name, key, full_notes, table_title):
        self.full_name = full_name;
        self.key = key;
        self.full_notes = full_notes;
        self.table_title = table_title;
        self.rows = [];

class scale_chordrow():
    """ Represents different rows (triads, 7ths, 9ths) in tabular
        representations of scale chords.
    """
    def __init__ (self, chord_type):
        self.chord_type = chord_type;
        self.notes = [];
        
class scale_chordcell():
    """ Represents different cells in the table of chords. """
    def __init__ (self, chordname_1, chordname_2, notes):
        self.chordname_1 = chordname_1;
        self.chordname_2 = chordname_2;
        self.notes = notes;

def makebaserep(notex, base = 0):
    """ Used for converting "C" -> 1, "Db"-> 2b, etc. """
    notexparsed = WestTemp.note_parse(notex);
    pos_rep = str(WestTemp.nat_key_lookup_order[notexparsed[0]] + base);
    if notexparsed[1] > 0:
        return pos_rep + (M_SHARP * notexparsed[1]);
    elif notexparsed[1] < 0:
        return pos_rep + (M_FLAT * (-1 * notexparsed[1]));
    else:
        return pos_rep;

def populate_scale_chords(scale_name, key, possiblechords):
    """ Returns an instance of scale_chords using the following inputs:

        scale_name: a name of a scale like "Dorian".
        key: generally a standard music key like "C".
        possiblechords: a sequence of chord types like "Seventh" and "Ninth".
    """
    our_scale = WestTemp.get_nseqby_name(scale_name, NSEQ_SCALE);
    num_elem = len(our_scale.nseq_posn);
    try: 
        int_of_key = int(key)
        is_key_an_int = True;
    except ValueError:
        is_key_an_int = False;
        int_of_key = None;
    if is_key_an_int:
        our_scale_notes = [makebaserep(x, int_of_key) for x in 
            our_scale.get_notes_for_key("C")];
    else:
        our_scale_notes = our_scale.get_notes_for_key(key); 
    our_chord_data = scale_chords(scale_name, key, our_scale_notes,
        make_roman_numeral_list(num_elem));
    for i in possiblechords:
        ourchordrow = scale_chordrow(i);
        our_chord_data.rows.append(ourchordrow);
        for j in range(num_elem):
            our_slice = CHORDTYPE_DICT[i];
            if is_key_an_int:
                our_chord_notes = [makebaserep(x, int_of_key) for x in
                    our_scale.get_notes_for_key("C", j, our_slice)];
            else:
                our_chord_notes = our_scale.get_notes_for_key(key, j, 
                    our_slice); 
            our_posn = our_scale.get_posn_for_offset(j, our_slice, True);
            our_chord = WestTemp.get_nseqby_seqpos(our_posn, NSEQ_CHORD);
            if our_chord:
                ourchordrow.notes.append(scale_chordcell(our_chord.nseq_name,
                    our_chord.nseq_abbrev, our_chord_notes));
            else:
                ourchordrow.notes.append(scale_chordcell("", "", 
                    our_chord_notes));
    return our_chord_data;    

def chordgentable(scales):
    """ Generates a table representation of a series of scales,
        represented by a scale_chords instance. """
    startrow  = "<tr>\n";
    endrow = "</tr>\n";
    thestring = "";
    for scale in scales:
        thestring += "<h2>%s</h2>\n" % (scale.key + " " +scale.full_name);
        thestring += ("<table id=\"%s\" class=\"chordtable\">\n" % 
            (scale.key + ""));
        thestring += "<caption>%s</caption>\n" % (scale.key + " " + scale.full_name +
            ": " + seqtostr(scale.full_notes));
        thestring += "<thead>\n" + startrow + "<th>Chord Types</th>\n";
        for q in range(7):
            thestring += "<th>%s</th>\n" % str(int_to_roman(q+1));
        thestring += endrow + "</thead>\n<tbody>\n";

        for i in scale.rows:
            thestring += startrow;    
            thestring += "<td>%s</td>\n" % i.chord_type;
            for j in i.notes:
                if not j.chordname_1:
                    thestring += ("<td><p>%s<br />" % (str(j.chordname_1)));
                    thestring += ("<i>%s</i><br />" % (str(j.chordname_2)));
                else:
                    thestring += ("<td><p>%s<br />" % (j.notes[0]+" "+str(j.chordname_1)));
                    thestring += ("<i>%s</i><br />" % (j.notes[0]+str(j.chordname_2)));
                    
                thestring += ("<b>%s</b></p></td>" % seqtostr(j.notes));                
            thestring += endrow;    
        thestring += "\n</tbody>\n</table>\n";    
    return thestring;

# Code for 2020 - commented out, because relative imports won't work as well in Python 3.

#if __name__ == "__main__":
#
#    def xhtmlchordgencontent(filename, content):
#        """ Generates an XHTML file using the output of chordgenerator. This
#        function is called only when executing this module as __main__. """
#        ffilename = codecs.open(filename, "w", "utf-8");
#        bplate = """<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" 
#        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
#        <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
#        <head>\n<meta http-equiv="Content-Type" content="text/html; 
#        charset=utf-8" />\n<title>Chord Generator</title>
#        <style type="text/css" media="screen"> h1 {color: blue;
#        text-align: center;} table {border: 1px solid #000000; 
#        border-collapse: collapse;} td {border: 1px solid #000000;}
#        th {border: 1px solid #000000;}</style>\n
#        </head><body>\n<h1>Chord Instances</h1>\n%s"</body>\n</html>\n""";
#        output = bplate % content;
#        ffilename.write(output);
#        ffilename.close();
#
#    MajorStuff = populate_scale_chords("Major", "C♯", CHORDTYPE_ARRAY); 
#    MelMinorStuff = populate_scale_chords("Melodic Minor", "C♯", 
#        CHORDTYPE_ARRAY); 
#    HarmMinStuff = populate_scale_chords("Harmonic Minor", "C♯", 
#        CHORDTYPE_ARRAY); 
#    HarmMajStuff = populate_scale_chords("Harmonic Major", "C♯", 
#        CHORDTYPE_ARRAY); 
#
#    tabledata = chordgentable([MajorStuff, MelMinorStuff, HarmMinStuff,
#        HarmMajStuff]);
#    xhtmlchordgencontent("chordgenerator.html", tabledata);
#    print(list(WestTemp.seq_maps.nseqtype_maps[NSEQ_SCALE].name_dict.keys()))
#