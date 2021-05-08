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

class ChordName(object):

        def __init__ (self, RootName, Modifier, Slash):
                """ ChordName initialization: RootName, Modifier, Slash 

                    RootName = name of root note of chord, e.g. 'a'
                    Modifier = type of chord, e.g. 'Maj7'
                    Slash    = bass note to indicate inversion, e.g. 'e'
                """
                self.RootName = RootName
                self.Modifier = Modifier
                self.Slash = Slash
                if Slash == '':
                        self.Slash = RootName
                
        def GetRootName(self):
                return self.RootName
        def GetModifier(self):
                return self.Modifier
        def GetSlash(self):
                return self.Slash
                
        def SetRootName(self,RootName):
                self.RootName = RootName
        def SetModifier(self,Modifier):
                self.Modifier = Modifier
        def SetSlash(self,Slash):
                self.Slash = Slash

        def __repr__(self):
                RName = self.RootName.__repr__().strip("'")
                RtName = RName[0].upper() + RName[1:]
                Repr = RtName + self.Modifier.__repr__().strip("'")
                
                if self.Slash != self.RootName:
                        SName = self.Slash.__repr__().strip("'")
                        SlName = SName[0].upper() + SName[1:]
                        Repr = Repr  + ' / ' + SlName 
                return Repr
                
        def Print(self):
                print self.__repr__()
                

