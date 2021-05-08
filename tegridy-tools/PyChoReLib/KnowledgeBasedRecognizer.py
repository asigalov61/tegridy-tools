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
from HelperFunctions import P

class KnowledgeBasedRecognizer(object):
        """ Class that matches note names to a scale database """
        def __init__ (self, DefinitionFile='', Verbosity = 0):
                self.KnowledgeBase = {}
                if DefinitionFile == '':
                        self.InitializeKnowledgeBase(Verbosity)
                else:
                        try:
                                self.ReadRecognitionKnowledgeBaseFromFile(DefinitionFile)
                        except:
                                P("Warning! Could not open scale definition file"+DefinitionFile,Verbosity)
                                P("Falling back to initialization from scratch",Verbosity)
                                self.InitializeKnowledgeBase(Verbosity)
                

        def InitializeKnowledgeBase(self,Verbosity=0):
                """
                Should be overridden in a derived class...
                """
                pass

        def WriteRecognitionKnowledgeBaseToFile(self, FileName, Verbosity=0):
                try:
                        import cPickle
                        cPickle.dump(self.KnowledgeBase,open(FileName,"w"))
                        
                except ImportError:
                        P("*** Warning! Could not create knowledge base file due to missing cPickle module. This indicates a problem with your Python installation. ",Verbosity)

        def ReadRecognitionKnowledgeBaseFromFile(self, FileName, Verbosity=0):
                try:
                        import cPickle
                        self.KnowledgeBae = {}
                        self.KnowledgeBase = cPickle.load(open(FileName))

                except ImportError:
                        P("*** Warning! Could not read the knowledge base file due to Python installation problem.",Verbosity)
                        
