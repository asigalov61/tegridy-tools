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
from Interval import Interval
from IntervalName import IntervalName
from IntervalRecognizer import IntervalRecognizer
from Chord import Chord
from ChordName import ChordName
from ChordRecognizer import ChordRecognizer
from Scale import Scale
from ScaleName import ScaleName
from ScaleRecognizer import ScaleRecognizer
import unittest

### TESTS       
        
class TestInterval(unittest.TestCase):
        ExpectedValues = ((['c','c']   ,0 ),
                          (['b','c']   ,1 ),
                          (['c','b']   ,11),
                          (['c','b#']  ,0 ),
                          (['b#','c']  ,0 ),
                          (['dbb','c'] ,0 ),
                          (['dbb','b#'],0 ),
                          (['gx','eb'], 6 ))

        ExpectedNoteDist = ((['c','b'],    6),
                            (['b','c'],    1),
                            (['c','c'],    0),
                            (['cx','bbb'], 6),
                            (['d','a'],    4))

        TranspositionTest = ( ((['c','e']  , 'f' ), (['f' ,'a' ])),
                              ((['b','f']  , 'bb'), (['bb','fb'])),
                              ((['ab','f#'], 'e' ), (['e' ,'cx'])),
                            )

        
        def testGetDistance(self):
                for Ch, Dst in self.ExpectedValues:
                        Distance = Interval(Ch[0],Ch[1]).GetDistance()
                        self.assertEqual(Dst, Distance)

        def testGetNoteNameDistance(self):
                for Ch, NDst in self.ExpectedNoteDist:
                        Distance = Interval(Ch[0],Ch[1]).GetNoteNameDistance()
                        self.assertEqual(NDst, Distance)

        def testTransposeTo(self):
                for Ch, Int in self.TranspositionTest:
                        TransposedInterval = Interval(Ch[0][0],Ch[0][1]).TransposeTo(Ch[1])
                        self.assertEqual(TransposedInterval.__repr__(), Int.__repr__())
                        print Interval(Ch[0][0],Ch[0][1]),"transposed to ",Ch[1]," gives",TransposedInterval


class TestRecognizeInterval(unittest.TestCase):
        ExpectedValues = ((['c','c'], IntervalName('unison','perfect')),
                          (['c','e'], IntervalName('third','major')),
                          (['e','g'], IntervalName('third','minor')),
                          (['g','d'], IntervalName('fifth','perfect')),
                          (['gx','dbb'], IntervalName('fifth','quadruply diminished')),
                          (['d','bb'], IntervalName('sixth','minor')),
                          (['c','f'], IntervalName('fourth','perfect')),
                          (['c','e#'], IntervalName('third','augmented')),
                          (['dbb','f'], IntervalName('third','augmented')),
                          (['dbb','e#'], IntervalName('second','triply augmented')),
                          (['a#','a'], IntervalName('octave','diminished')),
                          (['ax','a'], IntervalName('octave','doubly diminished')),
                          )

        def testRecognizeInterval(self):
                IR = IntervalRecognizer()
                for Ch, In in self.ExpectedValues:
                        IName = IR.RecognizeInterval(Interval(Ch[0],Ch[1]))
                        self.assertEqual(IName.__repr__(), In.__repr__())
                        print "Interval ",Ch," is found to be a ",IName


class TestChord(unittest.TestCase):
        KnownIntervalPatterns = ((['c','e','g'],        [4, 3   ]),
                                 (['e','g','c'],        [3, 5   ]),
                                 (['g','c','e'],        [5, 4   ]),
                                 (['c','e','g','c'],    [4, 3, 5]))

        KnownRemovedDuplicates = ((['c','e','c'],               ['c','e']),
                                  (['c','e','g','c','g','b'],   ['c','e','g','b']),
                                  (['c','c','c'],               ['c']))

        KnownInversions = ((['c','e','g'], [['c', 'e', 'g'], ['e','g','c'], ['g','c','e']]),)

        KnownTranspositions = ( ((['c','e','g'],'f'), ['f','a','c']),
                                ((['c','e','g'],'e#'),['e#','gx','b#']),
                              )
        KnownImpossibleTranspositions = ( ((['c','e','g'],'ex'),NoMatch), 
                                        )


                                  
        def testToIntervalPattern(self):
                for Ch, Pat in self.KnownIntervalPatterns:
                        IntervalPattern = Chord(Ch).ToIntervalPattern()
                        self.assertEqual(Pat, IntervalPattern)

        def testWithoutDuplicateNotes(self):
                for Ch, Pat in self.KnownRemovedDuplicates:
                        DuplicatesRemoved = Chord(Ch).WithoutDuplicateNotes()
                        self.assertEqual(DuplicatesRemoved, Pat)

        def testCreateAllInversions(self):
                for Ch, Pat in self.KnownInversions:
                        Inversions = Chord(Ch).CreateAllInversions()
                        Nr = 0
                        for P in Inversions:   # test if all inversions are present
                                self.assert_(P in Pat)
                                Nr = Nr+1
                        self.assertEqual(len(Pat), Nr) # test if no more inversions are present

        def testFindCanonicForm(self):
                for C in Chord(['c','e','g']).CreateAllPermutations():
                        self.assertEqual(Chord(C).FindCanonicForm(), ['c', 'e', 'g'])
                for C in Chord(['e','g','c','e']).CreateAllPermutations():
                        self.assertEqual(Chord(C).FindCanonicForm(), ['c', 'e', 'g']) 
                for C in Chord(['f', 'g', 'c']).CreateAllPermutations():
                        self.assertEqual(Chord(C).FindCanonicForm(), ['f', 'g', 'c'])
                for C in Chord(['c','f','g','b','d']).CreateAllPermutations():
                        self.assertEqual(Chord(C).FindCanonicForm(), ['b','c','d','f','g'])

        def testTransposeTo(self):
                for InputData, Result in self.KnownTranspositions:
                        self.assertEqual(Chord(InputData[0]).TransposeTo(InputData[1]), Result)
                        print "Chord ",InputData[0], " transposed to ",InputData[1],"gives ",Result
                for InputData, Result in self.KnownImpossibleTranspositions:
                        self.assertRaises(NoMatch, Chord(InputData[0]).TransposeTo, InputData[1])


class TestRecognizeChord(unittest.TestCase):
        FILENAME = ''
        VERBOSE = 1
        CR = ChordRecognizer(FILENAME,VERBOSE)

        def setUp(self):
                self.Test = ChordName('c','','c')
                self.KnownChords = ((['c','e','g','c']     , ChordName('c', ''           ,'c' )),
                                    (['c','g','e','c']     , ChordName('c' ,''           ,'c' )),
                                    (['e','g#','b','d']    , ChordName('e' ,'7'          ,'e' )),
                                    (['g#','b','d','e']    , ChordName('e' ,'7'          ,'g#')),
                                    (['a','c','e']         , ChordName('a' ,'m'          ,'a' )),
                                    (['d','f','a','c']     , ChordName('d' ,'m7'         ,'d' )),
                                    (['d','f','c']         , ChordName('f' ,'6'          ,'d' )),
                                    (['b','d','f']         , ChordName('b' ,'dim'        ,'b' )),
                                    (['e','a','d']         , ChordName('a' ,'sus4'       ,'e' )),
                                    (['c','g','a']         , ChordName('c' ,'6'          ,'c' )),
                                    (['g','a','c']         , ChordName('c' ,'6'          ,'g' )),
                                    (['g','a','c']         , ChordName('c' ,'6'          ,'g' )), 
                                    (['b#','e#','gx']      , ChordName('e#',''           ,'b#')),
                                    (['d','eb','g']        , ChordName('eb','Maj7(omit5)','d' )),
                                    (['c','db','f']        , ChordName('db','Maj7(omit5)','c' )),
                                    (['f','gb','bb']       , ChordName('gb','Maj7(omit5)','f' )),
                                    (['c','f','bb']        , ChordName('f' ,'sus4'       ,'c' )),
                                    (['c','f','g','bb','d'], ChordName('bb','6/9'        ,'c' )),
                                    (['c','g','bb']        , ChordName('c' ,'7(omit3)'   ,'c' )),
                                    (['f','a','c','e']     , ChordName('f' ,'Maj7'       ,'f' )))
                                       
        def testRecognizeChord(self):
                for ChNot, ChNam in self.KnownChords:
                        self.assertEqual( self.CR.RecognizeChord(Chord(ChNot)).__repr__(), ChNam.__repr__() )
                        print ChNot,"is a ",ChNam.__repr__(),"chord"


        def testWriteReadChordRecognitionKnowledgeBase(self):
                print "Writing chord recognition knowledge base to disk"
                self.CR.WriteRecognitionKnowledgeBaseToFile('Stefaan.ChordDefs')
                print "Done writing."
                
                print "Loading chord recognition knowledge base from disk"
                self.CR.ReadRecognitionKnowledgeBaseFromFile('Stefaan.ChordDefs')
                print "Done loading."
        
                print "Test loaded definitions"
                self.testRecognizeChord()
                print "Done testing"
                
class TestScale(unittest.TestCase):
        KnownScale = (( ['c','d','e','f','g','a','b'], [2,2,1,2,2,2] ), )
        
        KnownTransposedScale = ( (( ['c','d','e','f','g','a','b']   , 'f'), ['f' ,'g','a','bb','c','d','e']), 
                                 (( ['c','d','eb','f','g','ab','bb'], 'f#'),['f#','g#','a','b','c#','d','e']),
                                 (( ['c','d','eb','f','g','ab','bb'], 'gb'),['gb','ab','bbb','cb','db','ebb','fb'])
                               )

        KnownScaleSort = ( ( ['c','e','d','f','g','a','b'], ['c','d','e','f','g','a','b']), 
                           ( ['c','e','dx','fb','d'], ['c','d','dx','e','fb']),
                           ( ['c','f','g','b','d'],   ['c','d','f','g','b']),
                         )

        def testScale(self):
                for S,R in self.KnownScale:
                        P = Scale(S).ToIntervalPattern()
                        self.assertEqual(P,R)
                        print S,"to interval pattern yields",R
                        
        def testTransposedScale(self):
                for S,R in self.KnownTransposedScale:
                        T = Scale(S[0]).TransposeTo(S[1])
                        self.assertEqual(T, R)
                        print S[0],"transposed to",S[1],"yields",T

        def testScaleSort(self):
                for S,R in self.KnownScaleSort:
                        T = Scale(S)
                        T.SortInPlaceRelativeToFirstNote()
                        print "Sorting scale",S,"yields ",T.Notes
                        self.assertEqual(T.Notes, R)
                        
class TestRecognizeScale(unittest.TestCase):
        FILENAME = ''
        VERBOSE = 1
        CR = ScaleRecognizer(FILENAME,VERBOSE)

        def setUp(self):
                self.KnownScales = ((['c','d','e','f','g','a','b'] ,       ScaleName('c', 'major', 'c', 'ionian' )),
                                    (['d','e','f','g','a','b','c'] ,       ScaleName('c', 'major', 'd', 'dorian' )),  
                                    (['d','e','f#','g','a','b','c#'],      ScaleName('d', 'major', 'd', 'ionian' )),
                                    (['e','f#','g#','a','b','c#','d'],     ScaleName('a', 'major', 'e', 'mixolydian' )),
                                    (['c','d','e','f#','g#','a#'],         ScaleName('c', 'whole tone','','')),
                                    (['d','e','f#','g#','a#','b#'],        ScaleName('d', 'whole tone','','')),
                                    (['c','d','eb','f','gb','ab','a','b'], ScaleName('c', 'diminished', 'c', 'whole step/half step')),
                                    (['e','f','g#','a','b','c','d'],       ScaleName('a', 'harmonic minor', 'e', 'harmonic minor mode 5')),
                                    (['db','d','e','f','g','ab','bb','b'], ScaleName('db','diminished', 'db', 'half step/whole step')),
                                   )
                                       
        def testRecognizeScale(self):
                for ChNot, ChNam in self.KnownScales:
                        self.assertEqual( self.CR.RecognizeScale(Scale(ChNot)).__repr__(), ChNam.__repr__() )
                        print ChNot,"is a ",ChNam.__repr__(),"scale"


        def testWriteReadScaleRecognitionKnowledgeBase(self):
                print "Writing chord recognition knowledge base to disk"
                self.CR.WriteRecognitionKnowledgeBaseToFile('Stefaan.ScaleDefs')
                print "Done writing."
                
                print "Loading chord recognition knowledge base from disk"
                self.CR.ReadRecognitionKnowledgeBaseFromFile('Stefaan.ScaleDefs')
                print "Done loading."
        
                print "Test loaded definitions"
                self.testRecognizeScale()
                print "Done testing"
                
                
def main():
        try:
                import psyco
                psyco.full()
        except ImportError:
                pass
                
        unittest.main()

if __name__ == '__main__':
        main()
        
