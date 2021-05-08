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

# helper function to generate permutations
def xcomb(items, n):
    "Generator for permuting items in a list."
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xcomb(items[:i]+items[i+1:],n-1):
                yield [items[i]]+cc

def perm(items):
    "Return all permutations of items in a list."
    return list(xcomb(items, len(items)))

def P(Verbosity, Arg):
        """ Prints Arg if Verbosity > 0 """
        if Verbosity != 0:
                print Arg

class Callable:
        def __init__(self, anycallable):
                self.__call__ = anycallable
