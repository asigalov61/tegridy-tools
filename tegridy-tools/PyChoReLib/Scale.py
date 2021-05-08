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
from Interval import Interval
from Exceptions import InvalidInput

class Scale(object):
        def __init__(self, ListOfNotesInScale):
                self.Notes = ListOfNotesInScale[:]
                if len(self.Notes) < 2:
                        raise InvalidInput 

        def ToIntervalPattern(self):
                """
                Function that takes a list of ascending note names,
                and returns a list of numeric intervals indicating the number 
                of half steps between two successive notes. 
                e.g.
                ToIntervalPattern(['c', 'e', 'g']) => [4, 3]
                """
                if len(self.Notes) == 0:
                        raise InvalidInput
                elif len(self.Notes) == 1:
                        return [0]
                else:
                        Idx = 0
                        Result = []
                        while (Idx+1) < len(self.Notes):
                                Result.append(Interval(self.Notes[Idx], self.Notes[Idx+1]).GetDistance())
                                Idx = Idx+1
                        return Result

        def WithoutDuplicateNotes(self):
                """ Returns a version of itself with all duplicate notenames removed """
                AlreadyFoundNote = {}
                NewList = []
                if len(self.Notes) == 0:
                        raise InvalidInput
                for Note in self.Notes:
                        if AlreadyFoundNote.has_key(Note):
                                pass
                        else:
                                AlreadyFoundNote[Note] = 1
                                NewList.append(Note)
                return NewList

        def SortInPlaceRelativeToFirstNote(self):
                if len(self.Notes) < 2: # already sorted!
                        return
                NotesCopy = self.Notes[:]
                NotesCopy.sort()                     # first step: sort alphabetically -> this allows to use inversions instead of permutations,
                                                     # avoiding combinatorial explosion
                NotesCopy.sort(Interval.SmallerThan) # second step: resort musically  (both steps should be combined)
                IdxFirst = NotesCopy.index(self.Notes[0])
                self.Notes = NotesCopy[IdxFirst:] + NotesCopy[:IdxFirst]

        def CreateAllInversions(self):
                """ Returns all inversions of this scale """
                Idx = 0
                Result = []
                if len(self.Notes) == 0:
                        raise InvalidInput
                while Idx < len(self.Notes):
                        Result.append( [self.Notes[Idx]] + self.Notes[Idx+1:] + self.Notes[:Idx])
                        Idx = Idx + 1
                return Result

        def TransposeTo(self, NewRootName):
                NewNotes = [NewRootName]
                for Note in self.Notes[1:]:
                        NewNotes.append(Interval(self.Notes[0],Note).TransposeTo(NewRootName).GetNoteName2())
                return NewNotes

        def FindCanonicForm(self):
                """ Rewrites a scale into its canonical form ("canonical" probably having another meaning outside this software). 
                    The canonic form for scales allows to distinguish between different modes.
                """
                SimpleScale = Scale(self.WithoutDuplicateNotes())
                SimpleScale.SortInPlaceRelativeToFirstNote()
                return SimpleScale.Notes
              
                
        def __repr__(self):
                """ Returns a representation of a scale """
                return self.Notes.__repr__()
        
        def Print(self):
                """ Prints a representation of a scale """
                print self.__repr__()


