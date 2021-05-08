#    This file is part of PyChoReLib.
#
#    PyChoReLib is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    PyChoReLib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyChoReLib; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from IntervalName import IntervalName
from Interval import Interval
from Exceptions import NoMatch

class IntervalRecognizer(object):
        def __init__ (self):
                """ Interval knowledge consists of interval names and an interval type.
                    The interval name is a string (e.g. "third")
                    The interval type is either 1 or 2. 
                    Intervals of type 1 can have modifiers: perfect, augmented, diminished, doubly augmented or doubly diminished.
                    Intervals of type 2 can have modifiers: major, minor, augmented, diminished, doubly augmented or doubly diminished.
                """
                self.IntervalNameLookup = { 0 : ["unison", 1],
                                            1 : ["second", 2],
                                            2 : ["third" , 2],
                                            3 : ["fourth", 1],
                                            4 : ["fifth",  1],
                                            5 : ["sixth",  2],
                                            6 : ["seventh",2],
                                            7 : ["octave", 1],
                                          }
                                            
                self.MajorScale = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
                                           
                self.ModifierKnowledgeType1 = { -4 : "quadruply diminished",
                                                -3 : "triply diminished",
                                                -2 : "doubly diminished",
                                                -1 : "diminished",
                                                 0 : "perfect",
                                                 1 : "augmented",
                                                 2 : "doubly augmented",
                                                 3 : "triply augmented",
                                                 4 : "quadruply augmented"}
                                                 
                self.ModifierKnowledgeType2 = { -4 : "triply diminished",
                                                -3 : "doubly diminished",
                                                -2 : "diminished",
                                                -1 : "minor",
                                                 0 : "major",
                                                 1 : "augmented",
                                                 2 : "doubly augmented", 
                                                 3 : "triply augmented",
                                                 4 : "quadruply augmented"}
                                                 
                self.Modifiers = { 1 : self.ModifierKnowledgeType1, 
                                   2 : self.ModifierKnowledgeType2 }

                self.NominalHalfSteps = {}
                for Note in self.MajorScale:
                        self.NominalHalfSteps[Interval(self.MajorScale[0],Note).GetNoteNameDistance()] = Interval(self.MajorScale[0],Note).GetDistance()

        def RecognizeInterval(self, Intval):
                """ Gives a name to an interval """
                NoteDistance = Intval.GetNoteNameDistance()
                LookupValue = self.IntervalNameLookup[NoteDistance]
                Difference = Intval.GetDistance() - self.NominalHalfSteps[NoteDistance]
                if Difference > 6:
                        LookupValue = self.IntervalNameLookup[7]
                        Difference = Difference - 12
                        
                try:
                        Modifier = self.Modifiers[LookupValue[1]][Difference]                
                        return IntervalName(LookupValue[0], Modifier)
                except KeyError:
                        raise NoMatch

