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
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301   USA

class ScaleName(object):

        def __init__ (self, RootName, Type, ModeRootName='', Mode=''):
                """ ScaleName initialization: RootName, Type, Mode 

                    RootName = name of root note of base scale, e.g. 'c'
                    Type = type of scale, e.g. 'major', 'minor'
                    ModeRootName = rootname for mode (e.g. d for d dorian)
                    Mode     = mode (e.g. 'ionian','dorian','phrygian','lydian','mixolydian','aeolian','locrian', ...)
                """
                self.RootName = RootName
                self.Type = Type
                self.ModeRootName = ModeRootName
                self.Mode = Mode
                if self.Mode == '':
                        self.Mode = self.Type
                        self.ModeRootName = self.RootName
                
        def GetRootName(self):
                return self.RootName
        def GetType(self):
                return self.Type
        def GetModeRootName(self):
                return self.ModeRootName
        def GetMode(self):
                return self.Mode
                
        def SetRootName(self,RootName):
                self.RootName = RootName
        def SetType(self,Type):
                self.Type = Type
        def SetModeRootName(self,ModeRootName):
                self.ModeRootName = ModeRootName
        def SetMode(self,Mode):
                self.Mode = Mode

        def __repr__(self):
                RName = self.RootName.__repr__().strip("'")         # for D dorian, this is 'c'
#                print "RName = ",RName,"expected for d dorian: c"
                MRName= self.ModeRootName.__repr__().strip("'")     # for D dorian, this is 'd'
#                print "MRName = ",MRName,"expected for d dorian: d"
                RtName = RName[0].upper() + RName[1:] + " "         # for D dorian, this is 'C'
#                print "RtName = ",RtName,"expected for d dorian: C"
                MRtName= MRName[0].upper() + MRName[1:] + " "       # for D dorian, this is 'D'
#                print "MRtName = ",MRtName, "expected for d dorian: D" 
                Repr = MRtName                                      # for D dorian, this is 'D'
#                print "MRtName = ","expected for d dorian: D"
                MdName = self.Mode.__repr__().strip("'")            # for D dorian, this is 'dorian'
#                print "MdName = ","expected for d dorian: dorian"
                TName = self.Type.__repr__().strip("'")             # for D dorian, this is 'major'
#                print "TName = ","expected for d dorian: major"

                if self.Mode == '': 
                        Repr = RtName + TName                       # for D dorian, this would be 'C major'
                else:
                        Repr = MRtName  + MdName + " (based on " + RtName + TName + ")"
                        #for D dorian, this would be 'D dorian (based on C major scale)

                return Repr
                
        def Print(self):
                print self.__repr__()
                

