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
from Scale import Scale
from ScaleName import ScaleName
from HelperFunctions import P

class ScaleRecognizer(KnowledgeBasedRecognizer):
                
        def RegisterScaleDefinitionByExample(self, AScale, AScaleName, Verbosity):
                """
                Method to register a new scale type, by showing an example of such a scale.
                """
                SimplifiedScale = AScale.FindCanonicForm()
                if SimplifiedScale != AScale.Notes:
                        if Verbosity != 0:
                                print "*** Warning: registration of non-canonical scale", AScale.Notes, "as ", SimplifiedScale
                                pass
                        
                Intervals= Scale(SimplifiedScale).ToIntervalPattern()
                DBKey = tuple(Intervals)
                if self.KnowledgeBase.has_key(DBKey):
                        if Verbosity != 0:
                                print "*** Warning: duplicate Scale definition overrules earlier defined scale. Old: ",self.KnowledgeBase[DBKey][0].Print(),"New: ",AScaleName.Print()
                RootNoteIndex = SimplifiedScale.index(AScaleName.GetRootName())
                ModeRootNoteIndex = SimplifiedScale.index(AScaleName.GetModeRootName())
                self.KnowledgeBase[DBKey] = tuple([AScaleName, RootNoteIndex, ModeRootNoteIndex])

        def RegisterScaleSetByExample(self, ScaleType, ListOfScaleNotes, ModeNames, Verbosity):
                """
                Convenience method to register a whole set of modes formed by starting AScale on successive notes.
                """
                for Mode in ModeNames:
                        if not (Mode is None):
                                StartNoteIdx = ModeNames.index(Mode)
                                TheScale = Scale(ListOfScaleNotes[StartNoteIdx:] + ListOfScaleNotes[:StartNoteIdx])
                                TheScaleName = ScaleName(ListOfScaleNotes[0], ScaleType, ListOfScaleNotes[StartNoteIdx], Mode)
                                self.RegisterScaleDefinitionByExample(TheScale, TheScaleName, Verbosity)
                        

        def InitializeKnowledgeBase(self,Verbosity=0):
                """ Teach all scale types to the system. This may take some time. 
                    Alternatively, a generated database can be serialized to disk, and loaded on start-up
                """
                ExampleMajorScale = ['c','d','e','f','g','a','b']
                MajorScaleModeNames = ['ionian','dorian','phrygian','lydian','mixolydian','aeolian','locrian']
                
                ExampleHarmonicMinorScale = ['a','b','c','d','e','f','g#']
                HarmonicMinorScaleModeNames = ['harmonic minor mode 1','harmonic minor mode 2','harmonic minor mode 3','harmonic minor mode 4','harmonic minor mode 5','harmonic minor mode 6','harmonic minor mode 7']
                
                ExampleMelodicMinorScale = ['a','b','c','d','e','f#','g#']
                MelodicMinorScaleModeNames = ['minor-major','minor-major mode 2','lydian augmented','lydian dominant','minor-major mode 5','half-diminished','altered']

                ExampleDiminishedScale = ['g','ab','a#','b','c#','d','e','f']
                DiminishedScaleModeNames = ['half step/whole step',]

                ExampleDiminishedScale2= ['c','d','eb','f','gb','ab','a','b']
                DiminishedScaleModeNames2 = ['whole step/half step',]

                ExampleWholeToneScale = ['c','d','e','f#','g#','a#']
                WholeToneScaleModeNames = ['',] # there is only one mode, so it is not given a name...
                
                ExampleBluesScale = ['c','eb','f','f#','g','bb']
                BluesScaleModeNames = ['blues mode 1','blues mode 2','blues mode 3','blues mode 4','blues mode 5','blues mode 6']
                
                ExampleBebopScale =   ['c','d','e','f','g','a','bb','b']
                BebopScaleModeNames = ['bebop dominant','bebop dominant mode 2','bebop dominant mode 3','bebop dominant mode 4','bebop dorian','bebop dominant mode 6','bebop dominant mode 7' ,'bebop dominant mode 8']

                ExampleBebopScale2 = ['c','d','e','f','g','g#','a','b']
                BebopScaleModeNames2=['bebop major','bebop major mode 2','bebop major mode 3','bebop major mode 4','bebop major mode 5','bebop major mode 6','bebop major mode 7','bebop major mode 8']

                ExampleBebopScale3 = ['c','d','eb','f','g','g#','a','b']
                BebopScaleModeNames3=['bebop melodic minor','bebop melodic minor mode 2','bebop melodic minor mode 3' ,'bebop melodic minor mode 4','bebop melodic minor mode 5','bebop melodic minor mode 6' ,'bebop melodic minor mode 7','bebop melodic minor mode 8']

                ExampleChromaticScale = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
                ChromaticScaleModeNames = ['',] # only one mode

                ExamplePentatonicScale = ['c','d','e','g','a']
                PentatonicScaleModeNames = ['pentatonic mode 1','pentatonic mode 2','pentatonic mode 3','pentatonic mode 4',None]

                ExampleMinorPentatonicScale = ['c','eb','f','g','bb']
                MinorPentatonicScaleModeNames = ['minor pentatonic',] 
                
                ExampleAlteredPentatonicScale = ['e','f','a','b','c#']
                AlteredPentatonicScaleModeNames = ['altered pentatonic','altered pentatonic mode 2','altered pentatonic mode 3','altered pentatonic mode 4','altered pentatonic mode 5']
                
                ExampleInSenScale = ['e','f','a','b','d']
                InSenModeNames = ['In-sen','In-sen mode 2','In-sen mode 3','In-sen mode 4','In-sen mode 5']

                P(Verbosity,"Start teach scale types. This may take some time")
                self.RegisterScaleSetByExample('major',              ExampleMajorScale,            MajorScaleModeNames,            Verbosity)
                self.RegisterScaleSetByExample('harmonic minor',     ExampleHarmonicMinorScale,    HarmonicMinorScaleModeNames,    Verbosity)
                self.RegisterScaleSetByExample('melodic minor',      ExampleMelodicMinorScale,     MelodicMinorScaleModeNames,     Verbosity)
                self.RegisterScaleSetByExample('diminished',         ExampleDiminishedScale,       DiminishedScaleModeNames,       Verbosity)
                self.RegisterScaleSetByExample('diminished',         ExampleDiminishedScale2,      DiminishedScaleModeNames2,      Verbosity)
                self.RegisterScaleSetByExample('whole tone',         ExampleWholeToneScale,        WholeToneScaleModeNames,        Verbosity)
                self.RegisterScaleSetByExample('blues',              ExampleBluesScale,            BluesScaleModeNames,            Verbosity)
                self.RegisterScaleSetByExample('bebop dominant',     ExampleBebopScale,            BebopScaleModeNames,            Verbosity)
                self.RegisterScaleSetByExample('bebop major',        ExampleBebopScale2,           BebopScaleModeNames2,           Verbosity)
                self.RegisterScaleSetByExample('bebop melodic minor',ExampleBebopScale3,           BebopScaleModeNames3,           Verbosity)
                self.RegisterScaleSetByExample('chromatic',          ExampleChromaticScale,        ChromaticScaleModeNames,        Verbosity)
                self.RegisterScaleSetByExample('pentatonic',         ExamplePentatonicScale,       PentatonicScaleModeNames,       Verbosity)
                self.RegisterScaleSetByExample('minor pentatonic',   ExampleMinorPentatonicScale,  MinorPentatonicScaleModeNames,  Verbosity)
                self.RegisterScaleSetByExample('altered pentatonic', ExampleAlteredPentatonicScale,AlteredPentatonicScaleModeNames,Verbosity)
                self.RegisterScaleSetByExample('In-sen',             ExampleInSenScale,            InSenModeNames,                 Verbosity)
                P(Verbosity,"Done teaching")

        def ScaleToKey(self, AScale):
                """ Method that takes a Scale AScale, and transforms it to a key into the knowledge base """
                SimplifiedScale = Scale(AScale.FindCanonicForm())
                Intervals = SimplifiedScale.ToIntervalPattern()
                return tuple(Intervals)
        
        def RecognizeScale(self, AScale):
                """ Method to lookup a Scale AScale in the knowledge base """
                DBKey = self.ScaleToKey(AScale)
                if self.KnowledgeBase.has_key(DBKey):
                        MatchedScale = self.KnowledgeBase[DBKey][0]
                        SimplifiedScale = Scale(AScale.FindCanonicForm())
                        RootName = SimplifiedScale.Notes[self.KnowledgeBase[DBKey][1]]
                        MatchedScale.SetRootName(RootName)
                        ModeRootName = SimplifiedScale.Notes[self.KnowledgeBase[DBKey][2]]
                        MatchedScale.SetModeRootName(ModeRootName)
                        return MatchedScale
                else:
                        raise NoMatch

