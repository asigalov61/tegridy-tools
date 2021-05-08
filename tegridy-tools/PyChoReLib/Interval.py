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

from Exceptions import InvalidInput, NoMatch
from HelperFunctions import Callable

class Interval(object):
        """ Class that defines an interval ("2-note chord") """
        #static method
        def SmallerThan(Note1,Note2):
                d1 = Interval(Note1,Note2).GetDistance()
                d2 = Interval(Note2,Note1).GetDistance()
                if d1 < d2:
                        return -1
                elif d1 == d2:
                        return 0
                else:
                        return 1
                     
                
        SmallerThan = Callable(SmallerThan)

        """ Normal class methods """
        def __init__(self, NoteName1, NoteName2):
                self.NoteName1 = NoteName1
                self.NoteName2 = NoteName2
                self.ChromaticScale = [    ['c' , 'b#', 'dbb'], # one row contains all synonyms (i.e. synonym for our purpose)
                                           ['c#', 'bx', 'db' ], # # denotes a sharp, b denotes a moll, x denotes a double sharp, bb denotes a double moll 
                                           ['d' , 'cx', 'ebb'],
                                           ['d#', 'eb'       ],
                                           ['e' , 'dx', 'fb' ],
                                           ['f' , 'e#', 'gbb'],
                                           ['f#', 'ex', 'gb' ],
                                           ['g' , 'fx','abb' ],
                                           ['g#', 'ab'       ],
                                           ['a' , 'gx','bbb' ],
                                           ['a#', 'bb','cbb' ],
                                           ['b' , 'ax','cb'  ] ]
                                           
                self.NoteNames = [ 'a', 'b', 'c', 'd', 'e', 'f', 'g']
                
                self.IntervalDictionary = {}
                self.InitializeIntervalDictionary(self.ChromaticScale)

        def GetNoteName1(self):
                return self.NoteName1

        def GetNoteName2(self):
                return self.NoteName2

        def InitializeIntervalDictionary(self, ChromaticScale):
                Idx = 0
                for NoteList in ChromaticScale:
                        for Note in NoteList:
                                self.IntervalDictionary[Note] = Idx
                        Idx = Idx+1

        def GetNoteNameDistance(self):
                """
                Returns the NoteName distance of a chord.
                e.g. Interval('c','b').GetNoteNameDistance() should return 6
                     Interval('b','c').GetNoteNameDistance() should return 1
                     Interval('cx','bbb').GetNoteNameDistance() should also return 6
                     Interval('d','a').GetNoteNameDistance() should return 4
                     Interval('c','c').GetNoteNameDistance() should return 0
                """
                try:    
                        IdxNote1 = self.NoteNames.index(self.NoteName1[0])
                        IdxNote2 = self.NoteNames.index(self.NoteName2[0])
                        if (IdxNote1 > IdxNote2):
                                return IdxNote2+7-IdxNote1
                        else:
                                return IdxNote2-IdxNote1
                        
                except ValueError:
                        raise InvalidInput

        def GetDistance(self):
                """
                Returns the number of half steps between two notes.
                Notes are ordered ascending.
                e.g.
                Interval('c','b').GetDistance() should return 11
                Interval('b','c').GetDistance() should return 1
                Interval('c','c').GetDistance() should return 0
                
                Raises InvalidInput upon InvalidInput
                
                """
                try:
                        IdxNote1 = self.IntervalDictionary[self.NoteName1]
                        IdxNote2 = self.IntervalDictionary[self.NoteName2]
                        Difference = IdxNote2 - IdxNote1
                        if (Difference < 0):
                                return Difference + 12
                        else:
                                return Difference
                                
                except KeyError:
                        raise InvalidInput

        def TransposeTo(self, NewRootNote):
                """ Returns a new interval, which is the current
                    interval transposed to the new root note
                """
                NewNoteIdx =  self.NoteNames.index(self.NoteName2[0]) + Interval(self.NoteName1, NewRootNote).GetNoteNameDistance()
                if NewNoteIdx > 6:
                        NewNoteIdx = NewNoteIdx - 7
                Modifiers = { -2 : "bb", -1 : "b", 0 : "", 1 : "#", 2 : "x" }
                SecondNote = self.NoteNames[NewNoteIdx]
                try:
                        return Interval(NewRootNote, SecondNote + Modifiers[Interval(self.NoteName1, self.NoteName2).GetDistance() - Interval(NewRootNote,SecondNote).GetDistance()])
                except KeyError:
                        raise NoMatch

        def __repr__(self):
                return "['" + self.NoteName1 + "', '" + self.NoteName2 + "']"
