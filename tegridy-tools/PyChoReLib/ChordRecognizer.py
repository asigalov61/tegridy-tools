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
from KnowledgeBasedRecognizer import KnowledgeBasedRecognizer
from Chord import Chord
from ChordName import ChordName
from HelperFunctions import P

class ChordRecognizer(KnowledgeBasedRecognizer):
                
        def RegisterChordDefinitionByExample(self, AChord, AChordName, Verbosity):
                """
                Method to register a new chord type, by showing an example of such a chord.

                First:  the chord is canonicalised, i.e.
                        - duplicate notes are removed
                        - notes are rearranged to minimize the intervals between them
                Second: - chords are stored in a database as an interval pattern, 
                          making the representation independent of the chord's root note
                        - also store the offset of the root note in the canonicalised representation, 
                          needed for unparsing from interval pattern to chord name

                e.g. a chord ['c','g','c','eb'] --- duplicate notes removed         ---> ['c','g','eb']
                                                --- rearrange to minimize intervals ---> ['c','eb','g']
                                                --- use as key an interval pattern  ---> [4, 3] (interval here is nr of half steps between 2 adjacent notes)
                                                --- store chordname and offset of root note ---> Database entry with key ([4,3]) is (ChordName,0)
                """
                SimplifiedChord = AChord.FindCanonicForm()
                if SimplifiedChord != AChord.Notes:
                        if Verbosity != 0:
                                print "*** Warning: registration of non-canonical chord", AChord.Notes, "as ", SimplifiedChord
                                pass
                        
                Intervals= Chord(SimplifiedChord).ToIntervalPattern()
                DBKey = tuple(Intervals)
                if self.KnowledgeBase.has_key(DBKey):
                        if Verbosity != 0:
                                print "*** Warning: duplicate Chord definition overrules earlier defined chord. Old: ",self.KnowledgeBase[DBKey][0].Print(),"New: ",AChordName.Print()
                BaseNoteIndex = SimplifiedChord.index(AChord.Notes[0])
                self.KnowledgeBase[DBKey] = tuple([AChordName, BaseNoteIndex])

        def InitializeKnowledgeBase(self,Verbosity=0):
                """ Teach all chord types to the system. This may take some time. 
                    Alternatively, a generated database could be serialized to disk, and loaded on start-up
                """
                P(Verbosity,"Start teach chord types. This may take some time")
                self.RegisterChordDefinitionByExample(Chord(['c','e','g']),                  ChordName('c','','c'),             Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','g','a']),                  ChordName('c','6','c'),            Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','b']),              ChordName('c','Maj7','c'),         Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','f#','b']),             ChordName('c','Maj7(#11)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','d']),              ChordName('c','add(9)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','d','e','b']),              ChordName('c','Maj7(9)','c'),      Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','d','e','a']),              ChordName('c','6(9)','c'),         Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g#']),                 ChordName('c','aug or +','c'),     Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g']),                 ChordName('c','m','c'),            Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','d']),             ChordName('c','madd(9)','c'),      Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','bb']),            ChordName('c','m7','c'),           Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','d','eb','bb']),            ChordName('c','m7(9)','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','b']),             ChordName('c','mMaj7','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','d','eb','b']),             ChordName('c','mMaj7(9)','c'),     Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','f#']),                ChordName('c','dim','c'),          Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','f#','a']),            ChordName('c','dim7','c'),         Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb']),             ChordName('c','7','c'),            Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','bb']),                 ChordName('c','7','c'),            Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb']),             ChordName('c','7sus4','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','f#','bb']),            ChordName('c','7(b5)','c'),        Verbosity)  # or 7(#11)'))
                self.RegisterChordDefinitionByExample(Chord(['c','d','e','bb']),             ChordName('c','7(9)','c'),         Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','a','bb']),             ChordName('c','7(13)','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','c#','e','bb']),            ChordName('c','7(b9)','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g#','bb']),            ChordName('c','7(b13)','c'),       Verbosity) # or 7aug or 7(#5)'))
                self.RegisterChordDefinitionByExample(Chord(['c','eb','e','bb']),            ChordName('c','7(#9)','c'),        Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g#','b']),             ChordName('c','Maj7aug','c'),      Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','f','g']),                  ChordName('c','sus4','c'),         Verbosity)
#                self.RegisterChordDefinitionByExample(Chord(['c','d','g']),                  ChordName('c','1+2+5','c'),        Verbosity)  # or add(9)omit(3)'))
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','a','d']),          ChordName('c','6/9','c'),          Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g','a','b']),          ChordName('c','Maj7add(13)','c'),  Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','b','d']),          ChordName('c','Maj9','c'),         Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g','b','d','a']),      ChordName('c','Maj13','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d']),         ChordName('c','9','c'),            Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d','a']),     ChordName('c','13','c'),           Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','a']),             ChordName('c','m6','c'),           Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','a','d']),         ChordName('c','m6/9','c'),         Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','a','d']),             ChordName('c','m6/9','c'),         Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','eb','g','bb','f']),        ChordName('c','m7add(11)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','a','bb']),        ChordName('c','m7add(13)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','bb','d']),        ChordName('c','m9','c'),           Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','bb','d','f']),    ChordName('c','m11','c'),          Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','bb','d','f','a']),ChordName('c','m13','c'),          Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','g','b','d']),         ChordName('c','m9Maj7','c'),       Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','eb','gb','bb']),           ChordName('c','m7(b5)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','gb','bb','d']),       ChordName('c','m9(b5)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','eb','gb','bb','d','f']),   ChordName('c','m11(b5)','c'),      Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','eb','gb','a']),            ChordName('c','o7','c'),           Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','eb','gb','a','b']),        ChordName('c','o7add(Maj7)','c'),  Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb']),             ChordName('c','sus7','c'),         Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb','d']),         ChordName('c','sus9','c'),         Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb','d','a']),     ChordName('c','sus13','c'),        Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','g','bb']),                 ChordName('c','sus7','c'),         Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','gb','b']),             ChordName('c','Maj7(b5)','c'),     Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g#','b']),             ChordName('c','Maj7(#5)','c'),     Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','b','f#']),         ChordName('c','Maj7(#11)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','b','d','f#']),     ChordName('c','Maj9(#11)','c'),    Verbosity)
#                self.RegisterChordDefinitionByExample(Chord(['c','e','g','b','d','f#','a']), ChordName('c','Maj13(#11)','c'),   Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g#','bb']),            ChordName('c','7(#5)','c'),        Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g#','bb','d']),        ChordName('c','9(#5)','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','db']),        ChordName('c','7(b9)','c'),        Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d#']),        ChordName('c','7(#9)','c'),        Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','gb','bb','db']),       ChordName('c','7(b5)(b9)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g#','bb','d#']),       ChordName('c','7(#5)(#9)','c'),    Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','e','g#','bb','db']),       ChordName('c','7(#5)(b9)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','f#']),        ChordName('c','7(#11)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d','f#']),    ChordName('c','9(#11)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','db','f#']),   ChordName('c','7(b9)(#11)','c'),   Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d#','f#']),   ChordName('c','7(#9)(#11)','c'),   Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','gb','bb','d','a']),    ChordName('c','13(b5)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','db','a']),    ChordName('c','13(b9)','c'),       Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','g','bb','d','f#','a']),ChordName('c','13(#11)','c'),      Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb','db']),        ChordName('c','sus7(b9)','c'),     Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','f','g','bb','db','a']),    ChordName('c','sus13(b9)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','g','bb']),                 ChordName('c','7(omit3)','c'),     Verbosity)
#               self.RegisterChordDefinitionByExample(Chord(['c','eb','bb']),                ChordName('c','m7(omit5)','c'),    Verbosity)
                self.RegisterChordDefinitionByExample(Chord(['c','e','b']),                  ChordName('c','Maj7(omit5)','c'),  Verbosity)
                P(Verbosity,"Done teaching")

        def ChordToKey(self, AChord):
                """ Method that takes a Chord AChord, and transforms it to a key into the knowledge base """
                SimplifiedChord = Chord(AChord.FindCanonicForm())
                Intervals = SimplifiedChord.ToIntervalPattern()
                return tuple(Intervals)
        
        def RecognizeChord(self, AChord):
                """ Method to lookup a Chord AChord in the knowledge base """
                DBKey = self.ChordToKey(AChord)
                if self.KnowledgeBase.has_key(DBKey):
                        MatchedChord = self.KnowledgeBase[DBKey][0]
                        SimplifiedChord = Chord(AChord.FindCanonicForm())
                        RootName = SimplifiedChord.Notes[self.KnowledgeBase[DBKey][1]]
                        MatchedChord.SetRootName(RootName)
                        SlashNote = AChord.Notes[0]
                        MatchedChord.SetSlash(SlashNote)
                        return MatchedChord
                else:
                        raise NoMatch

