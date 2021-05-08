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

class IntervalName(object):
        def __init__ (self, IntervalName, Modifier):
                self.IntervalName = IntervalName
                self.Modifier = Modifier

        def SetIntervalName(self, IntervalName):
                self.IntervalName = IntervalName
        def SetModifier(self, Modifier):
                self.Modifier = Modifier
        def GetIntervalName(self):
                return self.IntervalName
        def GetModifier(self):
                return self.Modifier
        def __repr__(self):
                return self.Modifier.__repr__().strip("'") + " " + self.IntervalName.__repr__().strip("'")
        def Print(self):
                print self.__repr__()

