#=================================================================================================
#
# musicpy_pop_generator Python module
#
# Partial musicpy code as a stand-alone Python module
#
# Version 1.0
#
# Original source code courtesy of Rainbow Dreamer
# https://github.com/Rainbow-Dreamer/musicpy
#
# Original source code retrieved on 06/06/2025
# Commit 3ca486a
#
# Project Los Angeles
# Tegridy Code 2025
#
#=================================================================================================
#
# Critical dependencies
#
# !pip install mido
#
#=================================================================================================
#
# Simple use example
#
# from musicpy_pop_generator import *
#
# write_pop()
#
#=================================================================================================

import os
import math
import struct
from io import BytesIO
from copy import deepcopy as copy

from fractions import Fraction
from dataclasses import dataclass
import functools

import itertools

import random

from difflib import SequenceMatcher

import mido

#=================================================================================================
# database.py
#=================================================================================================


class match:

    def __init__(self, current_dict):
        # keys and values should both be a list/tuple/set of data,
        # and they should have the same counts
        # if the key itself is given as a dict, then just use it
        if isinstance(current_dict, dict):
            self.dic = current_dict
        else:
            raise ValueError('a dictionary is required')

    def __call__(self, key, mode=0, index=None):
        # unlike __getitem__, this treat key as a whole to match(mode == 0)
        # when mode == 1, the same as __getitem__,
        # and you can set which index to return in the finding results,
        # if the index is set to None (as default), then return whole results.
        if mode == 0:
            result = self.dic[key]
            if index is None:
                return result
            else:
                return result[index]
        elif mode == 1:
            result = self[key[0]]
            if index is None:
                return result
            else:
                return result[index]

    def __getitem__(self, key):
        dic = self.dic
        for i in dic:
            if key in i:
                return dic[i]
        raise KeyError(key)

    def __contains__(self, obj):
        return any(obj in i for i in self.dic)

    def search_all(self, key):
        result = []
        dic = self.dic
        for i in dic:
            if key in i:
                result.append(dic[i])
        return result

    def keys(self):
        return self.dic.keys()

    def values(self):
        return self.dic.values()

    def items(self):
        return self.dic.items()

    def __iter__(self):
        return self.dic.__iter__()

    def keynames(self):
        return [x[0] for x in self.dic.keys()]

    def valuenames(self):
        return [x[0] for x in self.dic.values()]

    def reverse(self, mode=0):
        dic = self.dic
        return match({
            ((tuple(j), ) if (not isinstance(j, tuple) or mode == 1) else j):
            i
            for i, j in dic.items()
        })

    def __repr__(self):
        return str(self.dic)

    def update(self, key, value=None):
        if isinstance(key, dict):
            self.dic.update(key)
        elif isinstance(key, match):
            self.dic.update(key.dic)
        else:
            if not isinstance(key, (list, tuple, set)):
                key = (key, )
            self.dic[tuple(key)] = value

    def delete(self, key):
        for i in self.dic:
            if key in i:
                del self.dic[i]
                return


class Interval:

    def __init__(self, number, quality, name=None, direction=1):
        self.number = number
        self.quality = quality
        self.name = name
        self.direction = direction
        self.value = self.get_value()

    def __repr__(self):
        return f'{"-" if self.direction == -1 else ""}{self.quality}{self.number}'

    def get_value(self):
        if len(self.quality) > 1 and len(set(self.quality)) == 1:
            current_quality = self.quality[0]
        else:
            current_quality = self.quality
        if current_quality not in quality_dict:
            raise ValueError(
                f'quality {self.quality} is not a valid quality, should be one of {list(quality_dict.keys())} or multiples of each'
            )
        if self.number not in interval_number_dict:
            raise ValueError(
                f'number {self.number} is not a valid number, should be one of {list(interval_number_dict.keys())}'
            )
        times = len(self.quality)
        quality_number = quality_dict[current_quality]
        if current_quality == 'd' and self.number % 7 in [1, 4, 5]:
            quality_number += 1
        if quality_number != 0:
            quality_number_sign = int(quality_number / abs(quality_number))
        else:
            quality_number_sign = 0
        current_value = interval_number_dict[
            self.number] + quality_number + quality_number_sign * (times - 1)
        current_value *= self.direction
        return current_value

    def change_direction(self, direction):
        self.direction = direction
        self.value = self.get_value()

    def __add__(self, other):
        
        if isinstance(other, note):
            current_pitch_name = other.name.upper()[0]
            current_pitch_name_ind = standard_pitch_name.index(
                current_pitch_name)
            new_other_name = standard_pitch_name[(current_pitch_name_ind +
                                                  self.direction *
                                                  (self.number - 1)) %
                                                 len(standard_pitch_name)]
            result = degree_to_note(degree=other.degree + self.value,
                                       duration=other.duration,
                                       volume=other.volume,
                                       channel=other.channel)
            result.name = relative_note(result.name, new_other_name)
        else:
            result = self.value + other
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        
        if isinstance(other, note):
            current_pitch_name = other.name.upper()[0]
            current_pitch_name_ind = standard_pitch_name.index(
                current_pitch_name)
            new_other_name = standard_pitch_name[(current_pitch_name_ind -
                                                  self.direction *
                                                  (self.number - 1)) %
                                                 len(standard_pitch_name)]
            result = degree_to_note(degree=other.degree - self.value,
                                       duration=other.duration,
                                       volume=other.volume,
                                       channel=other.channel)
            result.name = relative_note(result.name, new_other_name)
        else:
            result = other - self.value
        return result

    def __neg__(self):
        result = copy(self)
        result.change_direction(-result.direction)
        return result

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.value == other.value
        else:
            return self.value == other

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __divmod__(self, other):
        return divmod(self.value, other.value)

    def __rdivmod__(self, other):
        return divmod(other, self.value)

    def __hash__(self):
        return id(self)

    def sharp(self, unit=1):
        if unit == 0:
            return self
        if unit > 1:
            result = self
            for i in range(unit):
                result = result.sharp()
            return result
        current_interval_number_mod = self.number % 7
        if current_interval_number == 1:
            current_quality = ['P', 'A']
        elif current_interval_number_mod in [1, 4, 5]:
            current_quality = ['d', 'P', 'A']
        elif current_interval_number_mod in [0, 2, 3, 6]:
            current_quality = ['d', 'm', 'M', 'A']
        if self.quality not in current_quality and self.quality[
                0] == current_quality[0]:
            return Interval(self.number, self.quality[:-1])
        elif self.quality[0] == current_quality[-1]:
            return Interval(self.number, self.quality + current_quality[-1])
        elif self.quality in current_quality:
            current_quality_ind = current_quality.index(self.quality)
            return Interval(self.number,
                            current_quality[current_quality_ind + 1])

    def flat(self, unit=1):
        if unit == 0:
            return self
        if unit > 1:
            result = self
            for i in range(unit):
                result = result.flat()
            return result
        current_interval_number_mod = self.number % 7
        if current_interval_number == 1:
            current_quality = ['P', 'A']
        elif current_interval_number_mod in [1, 4, 5]:
            current_quality = ['d', 'P', 'A']
        elif current_interval_number_mod in [0, 2, 3, 6]:
            current_quality = ['d', 'm', 'M', 'A']
        if self.quality not in current_quality and self.quality[
                0] == current_quality[-1]:
            return Interval(self.number, self.quality[:-1])
        elif self.quality[0] == current_quality[0]:
            return Interval(self.number, self.quality + current_quality[0])
        elif self.quality in current_quality:
            current_quality_ind = current_quality.index(self.quality)
            return Interval(self.number,
                            current_quality[current_quality_ind - 1])

    def inverse(self):
        
        root_note = N('C')
        notes = root_note, root_note + self
        new_notes = notes[1], notes[0]
        while new_notes[0].degree >= new_notes[1].degree:
            new_notes[0].num -= 1
        result = get_pitch_interval(*new_notes)
        return result


quality_dict = {'P': 0, 'M': 0, 'm': -1, 'd': -2, 'A': 1, 'dd': -3, 'AA': 2}

quality_name_dict = {
    'P': 'perfect',
    'M': 'major',
    'm': 'minor',
    'd': 'diminished',
    'A': 'augmented',
    'dd': 'doubly-diminished',
    'AA': 'doubly-augmented'
}

interval_number_dict = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 9,
    7: 11,
    8: 12,
    9: 14,
    10: 16,
    11: 17,
    12: 19,
    13: 21,
    14: 23,
    15: 24,
    16: 26,
    17: 28
}

interval_number_name_list = [
    'unison', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
    'octave', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth',
    'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth'
]

interval_dict = {}
for i, each in enumerate(interval_number_name_list):
    current_interval_number = i + 1
    if current_interval_number == 1:
        current_quality = ['P', 'A', 'AA']
    else:
        current_interval_number_mod = current_interval_number % 7
        if current_interval_number_mod in [1, 4, 5]:
            current_quality = ['P', 'A', 'd', 'AA', 'dd']
        elif current_interval_number_mod in [0, 2, 3, 6]:
            current_quality = ['M', 'm', 'A', 'd', 'AA', 'dd']
    for each_quality in current_quality:
        current_interval_name = f'{quality_name_dict[each_quality]}_{each}'
        current_interval = Interval(number=current_interval_number,
                                    quality=each_quality,
                                    name=current_interval_name)
        interval_dict[current_interval_name] = current_interval
        interval_dict[str(current_interval)] = current_interval

globals().update(interval_dict)

tritone = d5
octave = perfect_octave = P8
tritave = P12
double_octave = P15

semitone = halfstep = 1
wholetone = wholestep = tone = 2

accidentals = ['b', '#', 'x', '♮']

INTERVAL = {
    0: 'perfect unison',
    1: 'minor second',
    2: 'major second',
    3: 'minor third',
    4: 'major third',
    5: 'perfect fourth',
    6: 'diminished fifth',
    7: 'perfect fifth',
    8: 'minor sixth',
    9: 'major sixth',
    10: 'minor seventh',
    11: 'major seventh',
    12: 'perfect octave',
    13: 'minor ninth',
    14: 'major ninth',
    15: 'minor third / augmented ninth',
    16: 'major third / major tenth',
    17: 'perfect eleventh',
    18: 'augmented eleventh',
    19: 'perfect twelfth',
    20: 'minor thirteenth',
    21: 'major thirteenth'
}
NAME_OF_INTERVAL = {j: i for i, j in INTERVAL.items()}

standard = {
    'C': 0,
    'C#': 1,
    'D': 2,
    'D#': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A': 9,
    'A#': 10,
    'B': 11,
    'Bb': 10,
    'Eb': 3,
    'Ab': 8,
    'Db': 1,
    'Gb': 6
}

standard_lowercase = {
    'c': 0,
    'c#': 1,
    'd': 2,
    'd#': 3,
    'e': 4,
    'f': 5,
    'f#': 6,
    'g': 7,
    'g#': 8,
    'a': 9,
    'a#': 10,
    'b': 11,
    'bb': 10,
    'eb': 3,
    'ab': 8,
    'db': 1,
    'gb': 6
}

standard.update(standard_lowercase)

standard2 = {
    'C': 0,
    'C#': 1,
    'D': 2,
    'D#': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'G': 7,
    'G#': 8,
    'A': 9,
    'A#': 10,
    'B': 11
}

standard_dict = {'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#', 'Gb': 'F#'}

standard_dict2 = {
    i: (i.upper() if not (len(i) == 2 and i[1] == 'b') else
        standard_dict[i[0].upper() + i[1]])
    for i in standard_lowercase
}

reverse_standard_dict = {j: i for i, j in standard_dict.items()}

standard_dict.update(standard_dict2)

reverse_standard_dict.update({
    i: reverse_standard_dict[standard_dict2[i]]
    for i in standard_dict2 if standard_dict2[i] in reverse_standard_dict
})

standard_pitch_name = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

scaleTypes = match({
    ('major', ): [M2, M2, m2, M2, M2, M2, m2],
    ('minor', ): [M2, m2, M2, M2, m2, M2, M2],
    ('melodic minor', ): [M2, m2, M2, M2, M2, M2, m2],
    ('harmonic minor', ): [M2, m2, M2, M2, m2, m3, m2],
    ('lydian', ): [M2, M2, M2, m2, M2, M2, m2],
    ('dorian', ): [M2, m2, M2, M2, M2, m2, M2],
    ('phrygian', ): [m2, M2, M2, M2, m2, M2, M2],
    ('mixolydian', ): [M2, M2, m2, M2, M2, m2, M2],
    ('locrian', ): [m2, M2, M2, m2, M2, M2, M2],
    ('whole tone', ): [M2, M2, M2, M2, M2, M2],
    ('12', ): [m2, m2, m2, m2, m2, m2, m2, m2, m2, m2, m2, m2],
    ('major pentatonic', ): [M2, M2, m3, M2, m3],
    ('minor pentatonic', ): [m3, M2, M2, m3, M2]
})
diatonic_modes = [
    'major', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'minor', 'locrian'
]
# you can sort the chord types from most commonly used to least commonly used
# to get better chord detection results
chordTypes = match({
    ('major', 'M', 'maj', 'majorthird'): (M3, P5),
    ('minor', 'm', 'minorthird', 'min', '-'): (m3, P5),
    ('maj7', 'M7', 'major7th', 'majorseventh'): (M3, P5, M7),
    ('m7', 'min7', 'minor7th', 'minorseventh', '-7'): (m3, P5, m7),
    ('7', 'seven', 'seventh', 'dominant seventh', 'dom7', 'dominant7'):
    (M3, P5, m7),
    ('germansixth', ): (M3, P5, A6),
    ('minormajor7', 'minor major 7', 'mM7'): (m3, P5, M7),
    ('dim', 'o'): (m3, d5),
    ('dim7', 'o7'): (m3, d5, d7),
    ('half-diminished7', 'ø7', 'ø', 'half-diminished', 'half-dim', 'm7b5'):
    (m3, d5, m7),
    ('aug', 'augmented', '+', 'aug3', '+3'): (M3, A5),
    ('aug7', 'augmented7', '+7'): (M3, A5, m7),
    ('augmaj7', 'augmented-major7', '+maj7', 'augM7'): (M3, A5, M7),
    ('aug6', 'augmented6', '+6', 'italian-sixth'): (M3, A6),
    ('frenchsixth', ): (M3, d5, A6),
    ('aug9', '+9'): (M3, A5, m7, M9),
    ('sus', 'sus4'): (P4, P5),
    ('sus2', ): (M2, P5),
    ('9', 'dominant9', 'dominant-ninth', 'ninth'): (M3, P5, m7, M9),
    ('maj9', 'major-ninth', 'major9th', 'M9'): (M3, P5, M7, M9),
    ('m9', 'minor9', 'minor9th', '-9'): (m3, P5, m7, M9),
    ('augmaj9', '+maj9', '+M9', 'augM9'): (M3, A5, M7, M9),
    ('add6', '6', 'sixth'): (M3, P5, M6),
    ('m6', 'minorsixth'): (m3, P5, M6),
    ('add2', '+2'): (M2, M3, P5),
    ('add9', ): (M3, P5, M9),
    ('madd2', 'm+2'): (M2, m3, P5),
    ('madd9', ): (m3, P5, M9),
    ('7sus4', '7sus'): (P4, P5, m7),
    ('7sus2', ): (M2, P5, m7),
    ('maj7sus4', 'maj7sus', 'M7sus4'): (P4, P5, M7),
    ('maj7sus2', 'M7sus2'): (M2, P5, M7),
    ('9sus4', '9sus'): (P4, P5, m7, M9),
    ('9sus2', ): (M2, P5, m7, M9),
    ('maj9sus4', 'maj9sus', 'M9sus', 'M9sus4'): (P4, P5, M7, M9),
    ('11', 'dominant11', 'dominant 11'): (M3, P5, m7, M9, P11),
    ('maj11', 'M11', 'eleventh', 'major 11', 'major eleventh'):
    (M3, P5, M7, M9, P11),
    ('m11', 'minor eleventh', 'minor 11'): (m3, P5, m7, M9, P11),
    ('13', 'dominant13', 'dominant 13'): (M3, P5, m7, M9, P11, M13),
    ('maj13', 'major 13', 'M13'): (M3, P5, M7, M9, P11, M13),
    ('m13', 'minor 13'): (m3, P5, m7, M9, P11, M13),
    ('13sus4', '13sus'): (P4, P5, m7, M9, M13),
    ('13sus2', ): (M2, P5, m7, P11, M13),
    ('maj13sus4', 'maj13sus', 'M13sus', 'M13sus4'): (P4, P5, M7, M9, M13),
    ('maj13sus2', 'M13sus2'): (M2, P5, M7, P11, M13),
    ('add4', '+4'): (M3, P4, P5),
    ('madd4', 'm+4'): (m3, P4, P5),
    ('maj7b5', 'M7b5'): (M3, d5, M7),
    ('maj7#11', 'M7#11'): (M3, P5, M7, A11),
    ('maj9#11', 'M9#11'): (M3, P5, M7, M9, A11),
    ('69', '6/9', 'add69'): (M3, P5, M6, M9),
    ('m69', 'madd69'): (m3, P5, M6, M9),
    ('6sus4', '6sus'): (P4, P5, M6),
    ('6sus2', ): (M2, P5, M6),
    ('5', 'power chord'): (P5, ),
    ('5(+octave)', 'power chord(with octave)'): (P5, P8),
    ('maj13#11', 'M13#11'): (M3, P5, M7, M9, A11, M13),
    ('13#11', ): (M3, P5, m7, M9, A11, M13),
    ('fifth_9th', ): (P5, M9),
    ('minormajor9', 'minor major 9', 'mM9'): (m3, P5, M7, M9),
    ('dim(Maj7)', ): (m3, d5, M7)
})
standard_reverse = {j: i for i, j in standard2.items()}
detectScale = scaleTypes.reverse()
detectTypes = chordTypes.reverse(mode=1)

degree_match = {
    '1': [perfect_unison],
    '2': [major_second, minor_second],
    '3': [minor_third, major_third],
    '4': [perfect_fourth],
    '5': [perfect_fifth],
    '6': [major_sixth, minor_sixth],
    '7': [minor_seventh, major_seventh],
    '9': [major_ninth, minor_ninth],
    '11': [perfect_eleventh],
    '12': [octave],
    '13': [major_thirteenth, minor_thirteenth]
}

reverse_degree_match = match({tuple(j): i for i, j in degree_match.items()})

precise_degree_match = {
    '1': perfect_unison,
    'b2': minor_second,
    '2': major_second,
    'b3': minor_third,
    '3': major_third,
    '4': perfect_fourth,
    '#4': diminished_fifth,
    'b5': diminished_fifth,
    '5': perfect_fifth,
    '#5': minor_sixth,
    'b6': minor_sixth,
    '6': major_sixth,
    'b7': minor_seventh,
    '7': major_seventh,
    'b9': minor_ninth,
    '9': major_ninth,
    '#9': augmented_ninth,
    'b11': diminished_eleventh,
    '11': perfect_eleventh,
    '#11': augmented_eleventh,
    '12': octave,
    'b13': minor_thirteenth,
    '13': major_thirteenth,
    '#13': augmented_thirteenth
}

reverse_precise_degree_match = {
    perfect_unison: '1',
    minor_second: 'b2',
    major_second: '2',
    minor_third: 'b3',
    major_third: '3',
    perfect_fourth: '4',
    diminished_fifth: 'b5/#4',
    perfect_fifth: '5',
    minor_sixth: 'b6/#5',
    major_sixth: '6',
    minor_seventh: 'b7',
    major_seventh: '7',
    minor_ninth: 'b9',
    major_ninth: '9',
    augmented_ninth: '#9',
    diminished_eleventh: 'b11',
    perfect_eleventh: '11',
    augmented_eleventh: '#11',
    octave: '12',
    minor_thirteenth: 'b13',
    major_thirteenth: '13',
    augmented_thirteenth: '#13'
}

reverse_precise_degree_number_match = {
    i.value: j
    for i, j in reverse_precise_degree_match.items()
}

INSTRUMENTS = {
    'Acoustic Grand Piano': 1,
    'Bright Acoustic Piano': 2,
    'Electric Grand Piano': 3,
    'Honky-tonk Piano': 4,
    'Electric Piano 1': 5,
    'Electric Piano 2': 6,
    'Harpsichord': 7,
    'Clavi': 8,
    'Celesta': 9,
    'Glockenspiel': 10,
    'Music Box': 11,
    'Vibraphone': 12,
    'Marimba': 13,
    'Xylophone': 14,
    'Tubular Bells': 15,
    'Dulcimer': 16,
    'Drawbar Organ': 17,
    'Percussive Organ': 18,
    'Rock Organ': 19,
    'Church Organ': 20,
    'Reed Organ': 21,
    'Accordion': 22,
    'Harmonica': 23,
    'Tango Accordion': 24,
    'Acoustic Guitar (nylon)': 25,
    'Acoustic Guitar (steel)': 26,
    'Electric Guitar (jazz)': 27,
    'Electric Guitar (clean)': 28,
    'Electric Guitar (muted)': 29,
    'Overdriven Guitar': 30,
    'Distortion Guitar': 31,
    'Guitar harmonics': 32,
    'Acoustic Bass': 33,
    'Electric Bass (finger)': 34,
    'Electric Bass (pick)': 35,
    'Fretless Bass': 36,
    'Slap Bass 1': 37,
    'Slap Bass 2': 38,
    'Synth Bass 1': 39,
    'Synth Bass 2': 40,
    'Violin': 41,
    'Viola': 42,
    'Cello': 43,
    'Contrabass': 44,
    'Tremolo Strings': 45,
    'Pizzicato Strings': 46,
    'Orchestral Harp': 47,
    'Timpani': 48,
    'String Ensemble 1': 49,
    'String Ensemble 2': 50,
    'SynthStrings 1': 51,
    'SynthStrings 2': 52,
    'Choir Aahs': 53,
    'Voice Oohs': 54,
    'Synth Voice': 55,
    'Orchestra Hit': 56,
    'Trumpet': 57,
    'Trombone': 58,
    'Tuba': 59,
    'Muted Trumpet': 60,
    'French Horn': 61,
    'Brass Section': 62,
    'SynthBrass 1': 63,
    'SynthBrass 2': 64,
    'Soprano Sax': 65,
    'Alto Sax': 66,
    'Tenor Sax': 67,
    'Baritone Sax': 68,
    'Oboe': 69,
    'English Horn': 70,
    'Bassoon': 71,
    'Clarinet': 72,
    'Piccolo': 73,
    'Flute': 74,
    'Recorder': 75,
    'Pan Flute': 76,
    'Blown Bottle': 77,
    'Shakuhachi': 78,
    'Whistle': 79,
    'Ocarina': 80,
    'Lead 1 (square)': 81,
    'Lead 2 (sawtooth)': 82,
    'Lead 3 (calliope)': 83,
    'Lead 4 (chiff)': 84,
    'Lead 5 (charang)': 85,
    'Lead 6 (voice)': 86,
    'Lead 7 (fifths)': 87,
    'Lead 8 (bass + lead)': 88,
    'Pad 1 (new age)': 89,
    'Pad 2 (warm)': 90,
    'Pad 3 (polysynth)': 91,
    'Pad 4 (choir)': 92,
    'Pad 5 (bowed)': 93,
    'Pad 6 (metallic)': 94,
    'Pad 7 (halo)': 95,
    'Pad 8 (sweep)': 96,
    'FX 1 (rain)': 97,
    'FX 2 (soundtrack)': 98,
    'FX 3 (crystal)': 99,
    'FX 4 (atmosphere)': 100,
    'FX 5 (brightness)': 101,
    'FX 6 (goblins)': 102,
    'FX 7 (echoes)': 103,
    'FX 8 (sci-fi)': 104,
    'Sitar': 105,
    'Banjo': 106,
    'Shamisen': 107,
    'Koto': 108,
    'Kalimba': 109,
    'Bag pipe': 110,
    'Fiddle': 111,
    'Shanai': 112,
    'Tinkle Bell': 113,
    'Agogo': 114,
    'Steel Drums': 115,
    'Woodblock': 116,
    'Taiko Drum': 117,
    'Melodic Tom': 118,
    'Synth Drum': 119,
    'Reverse Cymbal': 120,
    'Guitar Fret Noise': 121,
    'Breath Noise': 122,
    'Seashore': 123,
    'Bird Tweet': 124,
    'Telephone Ring': 125,
    'Helicopter': 126,
    'Applause': 127,
    'Gunshot': 128
}

reverse_instruments = {j: i for i, j in INSTRUMENTS.items()}

mode_check_parameters = [['major', [1, 3, 5]], ['dorian', [2, 4, 7]],
                         ['phrygian', [3, 5, 4]], ['lydian', [4, 6, 7]],
                         ['mixolydian', [5, 7, 4]], ['minor', [6, 1, 3]],
                         ['locrian', [7, 2, 4]]]

chord_functions_roman_numerals = {
    1: 'I',
    2: 'II',
    3: 'III',
    4: 'IV',
    5: 'V',
    6: 'VI',
    7: 'VII',
}

roman_numerals_dict = match({
    ('I', 'i', '1'): 1,
    ('II', 'ii', '2'): 2,
    ('III', 'iii', '3'): 3,
    ('IV', 'iv', '4'): 4,
    ('V', 'v', '5'): 5,
    ('VI', 'vi', '6'): 6,
    ('VII', 'vii', '7'): 7
})

chord_function_dict = {
    'major': [0, ''],
    'minor': [1, ''],
    'maj7': [0, 'M7'],
    'm7': [1, '7'],
    '7': [0, '7'],
    'minormajor7': [1, 'M7'],
    'dim': [1, 'o'],
    'dim7': [1, 'o7'],
    'half-diminished7': [1, 'ø7'],
    'aug': [0, '+'],
    'aug7': [0, '+7'],
    'augmaj7': [0, '+M7'],
    'aug6': [0, '+6'],
    'frenchsixth': [0, '+6(french)'],
    'aug9': [0, '+9'],
    'sus': [0, 'sus4'],
    'sus2': [0, 'sus2'],
    '9': [0, '9'],
    'maj9': [0, 'M9'],
    'm9': [1, '9'],
    'augmaj9': [0, '+M9'],
    'add6': [0, 'add6'],
    'm6': [1, 'add6'],
    'add2': [0, 'add2'],
    'add9': [0, 'add9'],
    'madd2': [1, 'add2'],
    'madd9': [1, 'add9'],
    '7sus4': [0, '7sus4'],
    '7sus2': [0, '7sus2'],
    'maj7sus4': [0, 'M7sus4'],
    'maj7sus2': [0, 'M7sus2'],
    '9sus4': [0, '9sus4'],
    '9sus2': [0, '9sus2'],
    'maj9sus4': [0, 'M9sus4'],
    '13sus4': [0, '13sus4'],
    '13sus2': [0, '13sus2'],
    'maj13sus4': [0, 'M13sus4'],
    'maj13sus2': [0, 'M13sus2'],
    'add4': [0, 'add4'],
    'madd4': [1, 'add4'],
    'maj7b5': [0, 'M7b5'],
    'maj7#11': [0, 'M7#11'],
    'maj9#11': [0, 'M9#11'],
    '69': [0, '69'],
    'm69': [1, '69'],
    '6sus4': [1, '6sus4'],
    '6sus2': [1, '6sus2'],
    '5': [1, '5'],
    'maj11': [0, 'M11'],
    'm11': [1, '11'],
    '11': [0, '11'],
    '13': [0, '13'],
    'maj13': [0, 'M13'],
    'm13': [1, '13'],
    'maj13#11': [0, 'M13#11'],
    '13#11': [0, '13#11'],
    'fifth_9th': [0, '5/9'],
    'minormajor9': [1, 'M9']
}

chord_notation_dict = {
    'major': '',
    'minor': '-',
    'maj7': 'M7',
    'm7': '-7',
    '7': '7',
    'minormajor7': 'mM7',
    'dim': 'o',
    'dim7': 'o7',
    'half-diminished7': 'ø',
    'aug': '+',
    'aug7': '+7',
    'augmaj7': '+M7',
    'aug6': '+6',
    'frenchsixth': '+6(french)',
    'aug9': '+9',
    'sus': 'sus4',
    'sus2': 'sus2',
    '9': '9',
    'maj9': 'M9',
    'm9': [1, '9'],
    'augmaj9': '+M9',
    'add6': '6',
    'm6': 'm6',
    'add2': 'add2',
    'add9': 'add9',
    'madd2': 'madd2',
    'madd9': 'madd9',
    '7sus4': '7sus4',
    '7sus2': '7sus2',
    'maj7sus4': 'M7sus4',
    'maj7sus2': 'M7sus2',
    '9sus4': '9sus4',
    '9sus2': '9sus2',
    'maj9sus4': 'M9sus4',
    '13sus4': '13sus4',
    '13sus2': '13sus2',
    'maj13sus4': 'M13sus4',
    'maj13sus2': 'M13sus2',
    'add4': 'add4',
    'madd4': 'madd4',
    'maj7b5': 'M7b5',
    'maj7#11': 'M7#11',
    'maj9#11': 'M9#11',
    '69': '69',
    'm69': 'm69',
    '6sus4': '6sus4',
    '6sus2': '6sus2',
    '5': '5',
    'maj11': 'M11',
    'm11': 'm11',
    '11': '11',
    '13': '13',
    'maj13': 'M13',
    'm13': 'm13',
    'maj13#11': 'M13#11',
    '13#11': '13#11',
    'fifth_9th': '5/9',
    'minormajor9': 'M9'
}

drum_types = {
    27: 'High Q',
    28: 'Slap',
    29: 'Stratch Push',
    30: 'Stratch Pull',
    31: 'Sticks',
    32: 'Square Click',
    33: 'Metronome Click',
    34: 'Metronome Bell',
    35: 'Acoustic Bass Drum',
    36: 'Electric Bass Drum',
    37: 'Side Stick',
    38: 'Acoustic Snare',
    39: 'Hand Clap',
    40: 'Electric Snare',
    41: 'Low Floor Tom',
    42: 'Closed Hi-hat',
    43: 'High Floor Tom',
    44: 'Pedal Hi-hat',
    45: 'Low Tom',
    46: 'Open Hi-hat',
    47: 'Low-Mid Tom',
    48: 'Hi-Mid Tom',
    49: 'Crash Cymbal 1',
    50: 'High Tom',
    51: 'Ride Cymbal 1',
    52: 'Chinese Cymbal',
    53: 'Ride Bell',
    54: 'Tambourine',
    55: 'Splash Cymbal',
    56: 'Cowbell',
    57: 'Crash Cymbal 2',
    58: 'Vibra Slap',
    59: 'Ride Cymbal 2',
    60: 'High Bongo',
    61: 'Low Bongo',
    62: 'Mute High Conga',
    63: 'Open High Conga',
    64: 'Low Conga',
    65: 'High Timbale',
    66: 'Low Timbale',
    67: 'High Agogô',
    68: 'Low Agogô',
    69: 'Cabasa',
    70: 'Maracas',
    71: 'Short Whistle',
    72: 'Long Whistle',
    73: 'Short Guiro',
    74: 'Long Guiro',
    75: 'Claves',
    76: 'High Woodblock',
    77: 'Low Woodblock',
    78: 'Mute Cuica',
    79: 'Open Cuica',
    80: 'Mute Triangle',
    81: 'Open Triangle',
    82: 'Shaker',
    83: 'Jingle Bell',
    84: 'Belltree',
    85: 'Castanets',
    86: 'Mute Surdo',
    87: 'Open Surdo'
}

drum_mapping = {
    'K': 36,
    'H': 42,
    'S': 40,
    'S2': 38,
    'OH': 46,
    'PH': 44,
    'HC': 39,
    'K2': 35,
    'C': 57,
    'C2': 49,
    '0': -1,
    '-': -2
}

drum_mapping2 = {
    '0': 36,
    '1': 42,
    '2': 40,
    '3': 38,
    '4': 46,
    '5': 44,
    '6': 39,
    '7': 35,
    '8': 57,
    '9': 49,
    'x': -1,
    '-': -2
}

drum_set_dict = {
    1: 'Standard',
    9: 'Room Kit',
    17: 'Power Kit',
    25: 'Electronic Kit',
    26: 'TR-808 Kit',
    33: 'Jazz Kit',
    41: 'Brush Kit',
    49: 'Orchestra Kit',
    57: 'Sound FX Kit'
}
drum_set_dict_reverse = {j: i for i, j in drum_set_dict.items()}

drum_keywords = [
    'r', 'd', 'a', 't', 'l', 'n', 's', 'v', 'dl', 'di', 'dv', 'al', 'ai', 'av',
    'b'
]

guitar_standard_tuning = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

choose_chord_progressions_list = [
    '6451', '3456', '4536', '14561451', '1564', '4156', '4565', '4563', '6341',
    '6345', '6415', '15634145'
]

default_choose_melody_rhythms = [('b b b 0 b b b b', 1)]

default_choose_drum_beats = [
    'K, H, S, H, K, H, S, H, K, H, S, H, K, K, S, H, t:2',
    'K;H, 0, OH, 0, K;S;H, 0, OH, 0, K;H, 0, OH, 0, K;S;H, 0, OH, S, t:1'
]

default_choose_bass_rhythm = [('b b b b', 1 / 2)]

default_choose_bass_playing_techniques = ['octaves', 'root']

non_standard_intervals = [major_sixth, minor_sixth, minor_second]

#=================================================================================================
# structures.py
#=================================================================================================

class note:
    '''
    This class represents a single note.
    '''

    def __init__(self,
                 name,
                 num=4,
                 duration=1 / 4,
                 volume=100,
                 channel=None,
                 accidental=None):
        if not name:
            raise ValueError('note name is empty')
        if name[0] not in standard:
            raise ValueError(f"Invalid note name '{name}'")
        if accidental is not None:
            self.base_name = name
            self.accidental = accidental
        else:
            self.base_name = name[0]
            accidental = name[1:]
            if not accidental:
                accidental = None
            self.accidental = accidental
        self.num = num
        self.duration = duration
        volume = int(volume)
        self.volume = volume
        self.channel = channel

    @property
    def name(self):
        return f'{self.base_name}{self.accidental if self.accidental is not None else ""}'

    @name.setter
    def name(self, value):
        self.base_name = value[0]
        accidental = value[1:]
        if not accidental:
            accidental = None
        self.accidental = accidental

    @property
    def degree(self):
        return standard.get(
            self.name, standard.get(
                self.standard_name())) + 12 * (self.num + 1)

    @degree.setter
    def degree(self, value):
        self.name = standard_reverse[value % 12]
        self.num = (value // 12) - 1

    def __repr__(self):
        return f'{self.name}{self.num}'

    def __eq__(self, other):
        return type(other) is note and self.same_note(
            other) and self.duration == other.duration

    def __lt__(self, other):
        return self.degree < other.degree

    def __le__(self, other):
        return self.degree <= other.degree

    def to_standard(self):
        temp = copy(self)
        tename = standardize_note(tename)
        return temp

    def standard_name(self):
        return standardize_note(self.name)

    def get_number(self):
        return standard.get(
            self.name, standard.get(self.standard_name()))

    def same_note_name(self, other):
        return self.get_number() == other.get_number()

    def same_note(self, other):
        return self.same_note_name(other) and self.num == other.num

    def __matmul__(self, other):
        if isinstance(other, rhythm):
            return self.from_rhythm(other)

    def set_volume(self, vol):
        vol = int(vol)
        self.volume = vol

    def set(self, duration=None, volume=None, channel=None):
        if duration is None:
            duration = copy(self.duration)
        if volume is None:
            volume = copy(self.volume)
        if channel is None:
            channel = copy(self.channel)
        return note(self.name, self.num, duration, volume, channel)

    def __mod__(self, obj):
        return self.set(*obj)

    def accidental(self):
        result = ''
        if self.name in standard:
            if '#' in self.name:
                result = '#'
            elif 'b' in self.name:
                result = 'b'
        return result

    def join(self, other, ind, interval):
        if isinstance(other, str):
            other = to_note(other)
        if isinstance(other, note):
            return chord([copy(self), copy(other)], interval=interval)
        if isinstance(other, chord):
            temp = copy(other)
            teinsert(ind, copy(self))
            temp.interval.insert(ind, interval)
            return temp

    def up(self, unit=1):
        if isinstance(unit, Interval):
            return self + unit
        else:
            if unit == 0:
                return copy(self)
            return degree_to_note(self.degree + unit, self.duration,
                                     self.volume, self.channel)

    def down(self, unit=1):
        return self.up(-unit)

    def sharp(self, unit=1):
        temp = self
        for i in range(unit):
            temp += A1
        return temp

    def flat(self, unit=1):
        temp = self
        for i in range(unit):
            temp -= A1
        return temp

    def __pos__(self):
        return self.up()

    def __neg__(self):
        return self.down()

    def __invert__(self):
        name = self.name
        if name in standard_dict:
            if '#' in name:
                return self.reset(name=reverse_standard_dict[name])
            else:
                return self.reset(name=standard_dict[name])
        elif name in reverse_standard_dict:
            return self.reset(name=reverse_standard_dict[name])
        else:
            return self.reset(name=name)

    def flip_accidental(self):
        return ~self

    def __add__(self, obj):
        if isinstance(obj, int):
            return self.up(obj)
        elif isinstance(obj, Interval):
            return obj + self
        if not isinstance(obj, note):
            obj = to_note(obj)
        return chord([copy(self), copy(obj)])

    def __sub__(self, obj):
        if isinstance(obj, int):
            return self.down(obj)
        elif isinstance(obj, Interval):
            return obj.__rsub__(self)

    def __call__(self, obj=''):
        return C(self.name + obj, self.num)

    def with_interval(self, interval):
        result = chord([copy(self), self + interval])
        return result

    def get_chord_by_interval(start,
                              interval1,
                              duration=1 / 4,
                              interval=0,
                              cumulative=True):
        return get_chord_by_interval(start, interval1, duration, interval,
                                        cumulative)

    def dotted(self, num=1):
        temp = copy(self)
        if num == 0:
            return temp
        teduration = teduration * sum([(1 / 2)**i
                                             for i in range(num + 1)])
        return temp

    def reset_octave(self, num):
        temp = copy(self)
        tenum = num
        return temp

    def reset_pitch(self, name):
        temp = copy(self)
        tename = name
        return temp

    def reset_name(self, name):
        temp = to_note(name)
        teduration = self.duration
        tevolume = self.volume
        temp.channel = self.channel
        return temp

    def set_channel(self, channel):
        self.channel = channel

    def with_channel(self, channel):
        temp = copy(self)
        temp.channel = channel
        return temp

    def from_rhythm(self, current_rhythm, set_duration=True):
        return get_chords_from_rhythm(chords=self,
                                         current_rhythm=current_rhythm,
                                         set_duration=set_duration)


class chord:
    '''
    This class represents a collection of notes with relative distances.
    '''

    def __init__(self,
                 notes,
                 duration=None,
                 interval=None,
                 volume=None,
                 rootpitch=4,
                 other_messages=[],
                 start_time=None,
                 default_duration=1 / 4,
                 default_interval=0,
                 default_volume=100,
                 tempos=None,
                 pitch_bends=None):
        standardize_msg = False
        if isinstance(notes, str):
            notes = notes.replace(' ', '').split(',')
            if all(not any(i.isdigit() for i in j) for j in notes):
                standardize_msg = True
        elif isinstance(notes, list) and all(
                not isinstance(i, note)
                for i in notes) and all(not any(j.isdigit() for j in i)
                                        for i in notes if isinstance(i, str)):
            standardize_msg = True
        notes_msg = _read_notes(note_ls=notes,
                                rootpitch=rootpitch,
                                default_duration=default_duration,
                                default_interval=default_interval,
                                default_volume=default_volume)
        notes, current_intervals, current_start_time = notes_msg
        if current_intervals and interval is None:
            interval = current_intervals
        if standardize_msg and notes:
            root = notes[0]
            notels = [root]
            for i in range(1, len(notes)):
                last_note = notels[i - 1]
                current_note = notes[i]
                if isinstance(current_note, note):
                    current = note(name=current_note.name,
                                   num=last_note.num,
                                   duration=current_note.duration,
                                   volume=current_note.volume,
                                   channel=current_note.channel)
                    if current.get_number() <= last_note.get_number():
                        current.num += 1
                    notels.append(current)
                else:
                    notels.append(current_note)
            notes = notels
        self.notes = notes
        # interval between each two notes one-by-one
        self.interval = [0 for i in range(len(notes))]
        if interval is not None:
            self.change_interval(interval)
        if duration is not None:
            if isinstance(duration, (int, float)):
                for t in self.notes:
                    t.duration = duration
            else:
                for k in range(len(duration)):
                    self.notes[k].duration = duration[k]
        if volume is not None:
            self.set_volume(volume)
        if start_time is None:
            self.start_time = current_start_time
        else:
            self.start_time = start_time
        self.other_messages = other_messages
        self.tempos = tempos if tempos is not None else []
        self.pitch_bends = pitch_bends if pitch_bends is not None else []

    def get_duration(self):
        return [i.duration for i in self.notes]

    def get_volume(self):
        return [i.volume for i in self.notes]

    def get_degree(self):
        return [i.degree for i in self]

    def names(self, standardize_note=False):
        result = [i.name for i in self]
        if standardize_note:
            result = [standardize_note(i) for i in result]
        return result

    def __eq__(self, other):
        return type(
            other
        ) is chord and self.notes == other.notes and self.interval == other.interval

    def get_msg(self, types):
        return [i for i in self.other_messages if i.type == types]

    def cut(self,
            ind1=0,
            ind2=None,
            start_time=0,
            cut_extra_duration=False,
            cut_extra_interval=False,
            round_duration=False,
            round_cut_interval=False):
        # get parts of notes between two bars
        temp = copy(self)
        find_start = False
        if ind1 <= start_time:
            find_start = True
            actual_start_time = start_time - ind1
        else:
            actual_start_time = 0
        if ind2 is None:
            ind2 = tebars(mode=0, start_time=start_time)

        new_tempos = []
        new_pitch_bends = []
        new_other_messages = []
        adjust_time = ind1
        cut_bar_length = ind2 - ind1
        for each in temp.tempos:
            each.start_time -= adjust_time
            if 0 <= each.start_time < cut_bar_length:
                new_tempos.append(each)
        for each in temp.pitch_bends:
            each.start_time -= adjust_time
            if 0 <= each.start_time < cut_bar_length:
                new_pitch_bends.append(each)
        for each in temp.other_messages:
            each.start_time -= adjust_time
            if 0 <= each.start_time < cut_bar_length:
                new_other_messages.append(each)

        if ind2 <= start_time:
            result = chord([], start_time=ind2 - ind1)
            result.tempos = new_tempos
            result.pitch_bends = new_pitch_bends
            result.other_messages = new_other_messages
            return result

        current_bar = start_time
        notes = temp.notes
        intervals = temp.interval
        length = len(notes)
        start_ind = 0
        to_ind = length
        for i in range(length):
            current_bar += intervals[i]
            if round_cut_interval:
                current_bar = float(Fraction(current_bar).limit_denominator())
            if (not find_start) and current_bar >= ind1:
                start_ind = i + 1
                find_start = True
                actual_start_time = current_bar - ind1
            elif current_bar >= ind2:
                to_ind = i + 1
                break
        if not find_start:
            start_ind = to_ind
        result = temp[start_ind:to_ind]
        result.tempos = new_tempos
        result.pitch_bends = new_pitch_bends
        result.other_messages = new_other_messages
        result.start_time = actual_start_time
        if cut_extra_duration:
            current_bar = result.start_time
            new_notes = []
            new_intervals = []
            for i in range(len(result.notes)):
                current_note = result.notes[i]
                current_interval = result.interval[i]
                current_duration = current_note.duration
                new_bar_with_duration = current_bar + current_duration
                if new_bar_with_duration > cut_bar_length:
                    current_note.duration -= (new_bar_with_duration -
                                              cut_bar_length)
                    if round_duration:
                        current_note.duration = float(
                            Fraction(
                                current_note.duration).limit_denominator())
                    if current_note.duration > 0:
                        new_notes.append(current_note)
                        new_intervals.append(current_interval)
                    else:
                        if new_intervals:
                            new_intervals[-1] += current_interval
                else:
                    new_notes.append(current_note)
                    new_intervals.append(current_interval)
                current_bar += current_interval
            result.notes = new_notes
            result.interval = new_intervals
        if cut_extra_interval:
            if result.interval:
                current_bar = result.bars(mode=0, start_time=result.start_time)
                if current_bar > cut_bar_length:
                    result.interval[-1] -= (current_bar - cut_bar_length)
        return result

    def cut_time(self,
                 bpm,
                 time1=0,
                 time2=None,
                 start_time=0,
                 normalize_tempo=False,
                 cut_extra_duration=False,
                 cut_extra_interval=False,
                 round_duration=False,
                 round_cut_interval=False):
        if normalize_tempo:
            temp = copy(self)
            tenormalize_tempo(bpm)
            return tecut_time(bpm=bpm,
                                 time1=time1,
                                 time2=time2,
                                 start_time=start_time,
                                 normalize_tempo=False,
                                 cut_extra_duration=cut_extra_duration,
                                 cut_extra_interval=cut_extra_interval,
                                 round_duration=round_duration,
                                 round_cut_interval=round_cut_interval)
        bar_left = time1 / ((60 / bpm) * 4)
        bar_right = time2 / ((60 / bpm) * 4) if time2 is not None else None
        result = self.cut(ind1=bar_left,
                          ind2=bar_right,
                          start_time=start_time,
                          cut_extra_duration=cut_extra_duration,
                          cut_extra_interval=cut_extra_interval,
                          round_duration=round_duration,
                          round_cut_interval=round_cut_interval)
        return result

    def last_note_standardize(self):
        self.interval[-1] = self.notes[-1].duration

    def bars(self, start_time=0, mode=1, audio_mode=0, bpm=None):
        if mode == 0:
            max_length = sum(self.interval)
        elif mode == 1:
            temp = self.only_notes()
            if audio_mode == 1:
                from pydub import AudioSegment
                temp = teset(duration=[
                    real_time_to_bar(len(i), bpm) if isinstance(
                        i, AudioSegment) else i.duration for i in temp.notes
                ])
            current_durations = temp.get_duration()
            if not current_durations:
                return 0
            current_intervals = temp.interval
            max_length = current_durations[0]
            current_length = 0
            for i in range(1, len(temp)):
                current_duration = current_durations[i]
                last_interval = current_intervals[i - 1]
                current_length += last_interval + current_duration
                if current_length > max_length:
                    max_length = current_length
                current_length -= current_duration
        elif mode == 2:
            result = self.bars(start_time=start_time,
                               mode=1,
                               audio_mode=audio_mode,
                               bpm=bpm)
            last_extra_interval = self.interval[-1] - self.notes[-1].duration
            if last_extra_interval > 0:
                result += last_extra_interval
            return result
        else:
            raise ValueError('Invalid bars mode')
        return start_time + max_length

    def firstnbars(self, n, start_time=0):
        return self.cut(0, n, start_time)

    def get_bar(self, n, start_time=0):
        return self.cut(n, n + 1, start_time)

    def split_bars(self, start_time=0):
        bars_length = int(self.bars(start_time))
        result = []
        for i in range(bars_length):
            result.append(self.cut(i, i + 1, start_time))
        return result

    def count(self, note1, mode='name'):
        if isinstance(note1, str):
            if any(i.isdigit() for i in note1):
                mode = 'note'
            note1 = to_note(note1)
        if mode == 'name':
            return self.names().count(note1.name)
        elif mode == 'note':
            return self.notes.count(note1)

    def standard_notation(self):
        temp = copy(self)
        for each in temp.notes:
            if each.name in standard_dict:
                each.name = standard_dict[each.name]
        return temp

    def most_appear(self, choices=None, mode='name', as_standard=False):
        test_obj = self
        if as_standard:
            test_obj = self.standard_notation()
        if not choices:
            return max([i for i in standard2],
                       key=lambda s: test_obj.count(s))
        else:
            choices = [
                to_note(i) if isinstance(i, str) else i for i in choices
            ]
            if mode == 'name':
                return max([i.name for i in choices],
                           key=lambda s: test_obj.count(s))
            elif mode == 'note':
                return max(choices,
                           key=lambda s: test_obj.count(s, mode='note'))

    def count_appear(self, choices=None, as_standard=True, sort=False):
        test_obj = self
        if as_standard:
            test_obj = self.standard_notation()
        if not choices:
            choices = copy(standard2) if as_standard else copy(
                standard)
        else:
            choices = [
                to_note(i).name if isinstance(i, str) else i.name
                for i in choices
            ]
        result = {i: test_obj.count(i) for i in choices}
        if sort:
            result = [[i, result[i]] for i in result]
            result.sort(key=lambda s: s[1], reverse=True)
        return result

    def eval_time(self,
                  bpm,
                  ind1=None,
                  ind2=None,
                  mode='seconds',
                  start_time=0,
                  normalize_tempo=False,
                  audio_mode=0):
        if normalize_tempo:
            temp = copy(self)
            tenormalize_tempo(bpm)
            return teeval_time(bpm,
                                  ind1,
                                  ind2,
                                  start_time,
                                  mode=mode,
                                  audio_mode=audio_mode)
        if ind1 is None:
            whole_bars = self.bars(start_time, audio_mode=audio_mode, bpm=bpm)
        else:
            if ind2 is None:
                ind2 = self.bars(start_time, audio_mode=audio_mode, bpm=bpm)
            whole_bars = ind2 - ind1
        result = (60 / bpm) * whole_bars * 4
        if mode == 'seconds':
            result = round(result, 3)
            return f'{result}s'
        elif mode == 'hms':
            hours = int(result / 3600)
            minutes = int((result - 3600 * hours) / 60)
            seconds = round(result - 3600 * hours - 60 * minutes, 3)
            if hours:
                return f'{hours} hours, {minutes} minutes, {seconds} seconds'
            else:
                return f'{minutes} minutes, {seconds} seconds'
        elif mode == 'number':
            return result

    def count_bars(self, ind1, ind2, bars_range=True):
        bars_length = self[ind1:ind2].bars()
        if bars_range:
            start = self[:ind1].bars(mode=0)
            return [start, start + bars_length]
        else:
            return bars_length

    def clear_pitch_bend(self, value='all', cond=None):
        pitch_bends = self.pitch_bends
        length = len(pitch_bends)
        if cond is None:
            if value == 'all':
                self.pitch_bends.clear()
                return
            else:
                inds = [
                    i for i in range(length) if pitch_bends[i].value != value
                ]
        else:
            inds = [i for i in range(length) if not cond(pitch_bends[i])]
        self.pitch_bends = [pitch_bends[k] for k in inds]

    def clear_tempo(self, cond=None):
        if cond is None:
            self.tempos.clear()
        else:
            tempos = self.tempos
            length = len(tempos)
            inds = [i for i in range(length) if not cond(tempos[i])]
            self.tempos = [tempos[k] for k in inds]

    def only_notes(self):
        temp = copy(self)
        temp.clear_tempo()
        temp.clear_pitch_bend()
        return temp

    def __mod__(self, alist):
        if isinstance(alist, (list, tuple)):
            return self.set(*alist)
        elif isinstance(alist, (str, note)):
            return self.on(alist)

    def standardize(self, standardize_note=True):
        temp = self.only_notes()
        temp.names = temp.names()
        intervals = temp.interval
        durations = temp.get_duration()
        if standardize_note:
            names_standard = [standardize_note(i) for i in temp.names]
        else:
            names_standard = temp.names
        names_offrep = []
        new_duration = []
        new_interval = []
        for i in range(len(names_standard)):
            current = names_standard[i]
            if current not in names_offrep:
                if current is not None:
                    names_offrep.append(current)
                else:
                    names_offrep.append(temp.notes[i])
                new_interval.append(intervals[i])
                new_duration.append(durations[i])
        temp.notes = chord(names_offrep,
                           rootpitch=temp[0].num,
                           duration=new_duration).notes
        temp.interval = new_interval
        return temp

    def standardize_note(self):
        temp = copy(self)
        for i in temp:
            i.name = standardize_note(i.name)
        return temp

    def sortchord(self):
        temp = self.copy()
        temp.notes.sort(key=lambda x: x.degree)
        return temp

    def set(self, duration=None, interval=None, volume=None, ind='all'):
        if interval is None:
            interval = copy(self.interval)
        result = chord(copy(self.notes),
                       duration,
                       interval,
                       start_time=copy(self.start_time))
        if volume is not None:
            result.set_volume(volume, ind)
        return result

    def special_set(self,
                    duration=None,
                    interval=None,
                    volume=None,
                    ind='all'):
        if interval is None:
            interval = copy(self.interval)
        result = chord(copy(self.notes),
                       duration,
                       interval,
                       start_time=copy(self.start_time))
        result.interval = [
            0 if hasattr(self.notes[i], 'keep_same_time')
            and self.notes[i].keep_same_time else result.interval[i]
            for i in range(len(self))
        ]
        result.set_volume(volume)
        if volume is not None:
            result.set_volume(volume, ind)
        return result

    def change_interval(self, newinterval):
        if isinstance(newinterval, (int, float)):
            self.interval = [newinterval for i in range(len(self.notes))]
        else:
            if len(newinterval) == len(self.interval):
                self.interval = newinterval
            else:
                raise ValueError(
                    'please ensure the intervals between notes has the same numbers of the notes'
                )

    def __repr__(self):
        return self.show()

    def show(self, limit=10):
        if limit is None:
            limit = len(self.notes)
        current_notes_str = ', '.join([str(i) for i in self.notes[:limit]])
        if len(self.notes) > limit:
            current_notes_str += ', ...'
        current_interval_str = ', '.join([
            str(Fraction(i).limit_denominator()) for i in self.interval[:limit]
        ])
        if len(self.interval) > limit:
            current_interval_str += ', ...'
        result = f'chord(notes=[{current_notes_str}], interval=[{current_interval_str}], start_time={self.start_time})'
        return result

    def __contains__(self, note1):
        if not isinstance(note1, note):
            note1 = to_note(note1)
            if note1.name in standard_dict:
                note1.name = standard_dict[note1.name]
        return note1 in self.same_accidentals().notes

    def __add__(self, obj):
        if isinstance(obj, (int, list, Interval)):
            return self.up(obj)
        elif isinstance(obj, tuple):
            if isinstance(obj[0], chord):
                return self | obj
            else:
                return self.up(*obj)
        elif isinstance(obj, rest):
            return self.rest(obj.get_duration())
        temp = copy(self)
        if isinstance(obj, note):
            temp.notes.append(copy(obj))
            temp.interval.append(temp.interval[-1])
        elif isinstance(obj, str):
            return temp + to_note(obj)
        elif isinstance(obj, chord):
            temp |= obj
        return temp

    def __radd__(self, obj):
        if isinstance(obj, (rest, float)):
            temp = copy(self)
            temp.start_time += (obj if not isinstance(obj, rest) else
                                obj.get_duration())
            return temp
        elif isinstance(obj, (int, Interval)):
            return self + obj

    def __ror__(self, obj):
        if isinstance(obj, (int, float, rest)):
            temp = copy(self)
            temp.start_time += (obj if not isinstance(obj, rest) else
                                obj.get_duration())
            return temp

    def __pos__(self):
        return self.up()

    def __neg__(self):
        return self.down()

    def __invert__(self):
        return self.reverse()

    def __or__(self, obj):
        if isinstance(obj, (int, float)):
            return self.rest(obj)
        elif isinstance(obj, str):
            obj = trans(obj)
        elif isinstance(obj, tuple):
            first = obj[0]
            start = obj[1] if len(obj) == 2 else 0
            if isinstance(first, int):
                temp = copy(self)
                for k in range(first - 1):
                    temp |= (self, start)
                return temp
            elif isinstance(first, rest):
                return self.rest(first.get_duration(),
                                 ind=obj[1] if len(obj) == 2 else None)
            else:
                return self.add(first, start=start, mode='after')
        elif isinstance(obj, list):
            return self.rest(*obj)
        elif isinstance(obj, rest):
            return self.rest(obj.get_duration())
        return self.add(obj, mode='after')

    def __xor__(self, obj):
        if isinstance(obj, int):
            return self.inversion_highest(obj)
        if isinstance(obj, note):
            name = obj.name
        else:
            name = obj
        temp.names = self.names()
        if name in temp.names and name != temp.names[-1]:
            return self.inversion_highest(temp.names.index(name))
        else:
            return self + obj

    def __truediv__(self, obj):
        if isinstance(obj, int):
            if obj > 0:
                return self.inversion(obj)
            else:
                return self.inversion_highest(-obj)
        elif isinstance(obj, list):
            return self.sort(obj)
        else:
            if not isinstance(obj, chord):
                if isinstance(obj, str):
                    obj = trans_note(obj)
                temp.names = self.names()
                if obj.name not in standard2:
                    obj.name = standard_dict[obj.name]
                if obj.name in temp.names and obj.name != temp.names[0]:
                    return self.inversion(temp.names.index(obj.name))
            return self.on(obj)

    def __and__(self, obj):
        if isinstance(obj, tuple):
            if len(obj) == 2:
                first = obj[0]
                if isinstance(first, int):
                    temp = copy(self)
                    for k in range(first - 1):
                        temp &= (self, (k + 1) * obj[1])
                    return temp
                else:
                    return self.add(obj[0], start=obj[1], mode='head')
            else:
                return
        elif isinstance(obj, int):
            return self & (obj, 0)
        else:
            return self.add(obj, mode='head')

    def __matmul__(self, obj):
        if type(obj) is list:
            return self.get(obj)
        elif isinstance(obj, int):
            return self.inv(obj)
        elif isinstance(obj, str):
            return self.inv(self.names().index(
                standard_dict.get(obj, obj)))
        elif isinstance(obj, rhythm):
            return self.from_rhythm(obj)
        else:
            if isinstance(obj, tuple):
                return negative_harmony(obj[0], self, *obj[1:])
            else:
                return negative_harmony(obj, self)

    def negative_harmony(self, *args, **kwargs):
        return negative_harmony(current_chord=self, *args, **kwargs)

    def __call__(self, obj):
        # deal with the chord's sharp or flat notes, or to omit some notes
        # of the chord
        temp = copy(self)
        commands = obj.split(',')
        for each in commands:
            each = each.replace(' ', '')
            first = each[0]
            if first in ['#', 'b']:
                degree = each[1:]
                if degree in degree_match:
                    degree_ls = degree_match[degree]
                    found = False
                    for i in degree_ls:
                        current_note = temp[0] + i
                        if current_note in temp:
                            ind = temp.notes.index(current_note)
                            temp.notes[ind] = temp.notes[ind].sharp(
                            ) if first == '#' else temp.notes[ind].flat()
                            found = True
                            break
                    if not found:
                        if first == '#':
                            new_note = (temp[0] + degree_ls[0]).sharp()
                        else:
                            new_note = (temp[0] + degree_ls[0]).flat()
                        temp += new_note
                else:
                    self_names = temp.names()
                    if degree in self_names:
                        ind = temp.names().index(degree)
                        temp.notes[ind] = temp.notes[ind].sharp(
                        ) if first == '#' else temp.notes[ind].flat()
            elif each.startswith('omit') or each.startswith('no'):
                degree = each[4:] if each.startswith('omit') else each[2:]
                if degree in degree_match:
                    degree_ls = degree_match[degree]
                    for i in degree_ls:
                        current_note = temp[0] + i
                        if current_note in temp:
                            ind = temp.notes.index(current_note)
                            del temp.notes[ind]
                            del temp.interval[ind]
                            break
                else:
                    self_names = temp.names()
                    if degree in self_names:
                        temp = teomit(degree)
            elif each.startswith('sus'):
                num = each[3:]
                if num.isdigit():
                    num = int(num)
                else:
                    num = 4
                temp.notes = tesus(num).notes
            elif each.startswith('add'):
                degree = each[3:]
                if degree in degree_match:
                    degree_ls = degree_match[degree]
                    temp += (temp[0] + degree_ls[0])
            else:
                raise ValueError(f'{obj} is not a valid chord alternation')
        return temp

    def get(self, ls):
        temp = copy(self)
        result = []
        result_interval = []
        for each in ls:
            if isinstance(each, int):
                result.append(temp[each - 1])
                result_interval.append(temp.interval[each - 1])
            elif isinstance(each, float):
                num, pitch = [int(j) for j in str(each).split('.')]
                if num > 0:
                    current_note = temp[num - 1] + pitch * octave
                else:
                    current_note = temp[-num - 1] - pitch * octave
                result.append(current_note)
                result_interval.append(temp.interval[abs(num) - 1])
        return chord(result,
                     interval=result_interval,
                     start_time=temp.start_time)

    def pop(self, ind=None):
        if ind is None:
            result = self.notes.pop()
            self.interval.pop()
        else:
            result = self.notes.pop(ind)
            self.interval.pop(ind)
        return result

    def __sub__(self, obj):
        if isinstance(obj, (int, list, Interval)):
            return self.down(obj)
        elif isinstance(obj, tuple):
            return self.down(*obj)
        if not isinstance(obj, note):
            obj = to_note(obj)
        temp = copy(self)
        if obj in temp:
            ind = temp.notes.index(obj)
            del temp.notes[ind]
            del temp.interval[ind]
        return temp

    def __mul__(self, num):
        if isinstance(num, tuple):
            return self | num
        else:
            temp = copy(self)
            for i in range(num - 1):
                temp |= self
            return temp

    def __rmul__(self, num):
        return self * num

    def reverse(self, start=None, end=None, cut=False, start_time=0):
        temp = copy(self)
        if start is None:
            temp2 = teonly_notes()
            length = len(temp2)
            bar_length = temp2.bars()
            tempos = []
            pitch_bends = []
            for each in temp.tempos:
                each.start_time -= start_time
                each.start_time = bar_length - each.start_time
                tempos.append(each)
            for each in temp.pitch_bends:
                each.start_time -= start_time
                each.start_time = bar_length - each.start_time
                pitch_bends.append(each)
            if temp2.notes:
                last_interval = temp2.interval[-1]
                end_events = []
                current_start_time = 0
                for i in range(len(temp2.notes)):
                    current_note = temp2.notes[i]
                    current_end_time = current_start_time + current_note.duration
                    current_end_event = (current_note, current_end_time, i)
                    end_events.append(current_end_event)
                    current_start_time += temp2.interval[i]
                end_events.sort(key=lambda s: (s[1], s[2]), reverse=True)
                new_notes = [i[0] for i in end_events]
                new_interval = [
                    end_events[j][1] - end_events[j + 1][1]
                    for j in range(len(end_events) - 1)
                ]
                new_interval.append(last_interval)
                temp2.notes = new_notes
                temp2.interval = new_interval
            temp2.tempos = tempos
            temp2.pitch_bends = pitch_bends
            return temp2
        else:
            if end is None:
                result = temp[:start] + temp[start:].reverse(
                    start_time=start_time)
                return result[start:] if cut else result
            else:
                result = temp[:start] + temp[start:end].reverse(
                    start_time=start_time) + temp[end:]
                return result[start:end] if cut else result

    def reverse_chord(self, start_time=0):
        temp = copy(self)
        bar_length = tebars()
        temp.notes = temp.notes[::-1]
        temp.interval = temp.interval[::-1]
        if temp.interval:
            temp.interval.append(temp.interval.pop(0))
        for each in temp.tempos:
            each.start_time -= start_time
            each.start_time = bar_length - each.start_time
        for each in temp.pitch_bends:
            each.start_time -= start_time
            each.start_time = bar_length - each.start_time
        return temp

    def intervalof(self, cumulative=True, translate=False):
        degrees = self.get_degree()
        N = len(degrees)
        if not cumulative:
            if not translate:
                result = [degrees[i] - degrees[i - 1] for i in range(1, N)]
            else:
                result = [
                    get_pitch_interval(self.notes[i - 1], self.notes[i])
                    for i in range(1, N)
                ]
        else:
            if not translate:
                root = degrees[0]
                others = degrees[1:]
                result = [i - root for i in others]
            else:
                result = [
                    get_pitch_interval(self.notes[0], i)
                    for i in self.notes[1:]
                ]
        return result

    def add(self,
            note1=None,
            mode='after',
            start=0,
            duration=1 / 4,
            adjust_msg=True):
        if self.is_empty():
            result = copy(note1)
            shift = start
            if mode == 'after':
                shift += self.start_time
            elif mode == 'head':
                if result.interval:
                    last_note_diff = self.start_time - (
                        result.start_time + shift + sum(result.interval[:-1]))
                    result.interval[-1] = max(result.interval[-1],
                                              last_note_diff)
            result.start_time += shift
            if adjust_msg:
                result.apply_start_time_to_changes(shift, msg=True)
            if not result.notes:
                result.start_time = max(result.start_time, self.start_time)
            return result
        temp = copy(self)
        if note1.is_empty():
            if mode == 'after':
                if note1.start_time > 0:
                    temp = terest(note1.start_time)
            elif mode == 'head':
                if temp.interval:
                    last_note_diff = (note1.start_time + start) - (
                        temp.start_time + sum(temp.interval[:-1]))
                    temp.interval[-1] = max(temp.interval[-1], last_note_diff)
                else:
                    temp.start_time = max(temp.start_time,
                                          note1.start_time + start)
            return temp
        if mode == 'tail':
            note1 = copy(note1)
            adjust_interval = sum(temp.interval)
            temp.notes += note1.notes
            temp.interval += note1.interval
            if adjust_msg:
                note1.apply_start_time_to_changes(adjust_interval, msg=True)
            temp.other_messages += note1.other_messages
            temp.tempos += note1.tempos
            temp.pitch_bends += note1.pitch_bends
            return temp
        elif mode == 'head':
            note1 = copy(note1)
            if isinstance(note1, str):
                note1 = chord([to_note(note1, duration=duration)])
            elif isinstance(note1, note):
                note1 = chord([note1])
            elif isinstance(note1, list):
                note1 = chord(note1)
            # calculate the absolute distances of all of the notes of the chord to add and self,
            # and then sort them, make differences between each two distances
            apply_msg_chord = note1
            if temp.notes:
                note1_start_time = note1.start_time + start
                if note1_start_time < 0:
                    current_add_start_time = temp.start_time - note1_start_time
                    note1.start_time = temp.start_time + note1_start_time
                    temp, note1 = note1, temp
                    apply_msg_chord = temp
                else:
                    if note1_start_time < temp.start_time:
                        current_add_start_time = temp.start_time - note1_start_time
                        note1.start_time = note1_start_time
                        temp, note1 = note1, temp
                        apply_msg_chord = temp
                    else:
                        current_add_start_time = note1_start_time - temp.start_time

                if not temp.notes:
                    new_notes = note1.notes
                    new_interval = note1.interval
                    current_start_time = note1.start_time
                else:
                    distance = []
                    intervals1 = temp.interval
                    intervals2 = note1.interval
                    current_start_time = temp.start_time

                    if current_add_start_time != 0:
                        note1.notes.insert(0, temp.notes[0])
                        intervals2.insert(0, current_add_start_time)
                    counter = 0
                    for i in range(len(intervals1)):
                        distance.append([counter, temp.notes[i]])
                        counter += intervals1[i]
                    counter = 0
                    for j in range(len(intervals2)):
                        if not (j == 0 and current_add_start_time != 0):
                            distance.append([counter, note1.notes[j]])
                        counter += intervals2[j]
                    distance.sort(key=lambda s: s[0])
                    new_notes = [each[1] for each in distance]
                    new_interval = [each[0] for each in distance]
                    new_interval = [
                        new_interval[i] - new_interval[i - 1]
                        for i in range(1, len(new_interval))
                    ] + [distance[-1][1].duration]
            else:
                new_notes = note1.notes
                new_interval = note1.interval
                current_start_time = note1.start_time + start
            if adjust_msg:
                apply_msg_chord.apply_start_time_to_changes(start, msg=True)
            return chord(new_notes,
                         interval=new_interval,
                         start_time=current_start_time,
                         other_messages=temp.other_messages +
                         note1.other_messages,
                         tempos=temp.tempos + note1.tempos,
                         pitch_bends=temp.pitch_bends + note1.pitch_bends)
        elif mode == 'after':
            return self.rest(start + note1.start_time).add(note1, mode='tail')

    def inversion(self, num=1):
        if not 1 <= num < len(self.notes):
            raise ValueError(
                'the number of inversion is out of range of the notes in this chord'
            )
        else:
            temp = copy(self)
            for i in range(num):
                temp.notes.append(temp.notes.pop(0) + octave)
            return temp

    def inv(self, num=1, interval=None):
        temp = self.copy()
        if isinstance(num, str):
            return self @ num
        if not 1 <= num < len(self.notes):
            raise ValueError(
                'the number of inversion is out of range of the notes in this chord'
            )
        while temp[num].degree >= temp[num - 1].degree:
            temp[num] = temp[num].down(octave)
        current_interval = copy(temp.interval)
        teinsert(0, tepop(num))
        temp.interval = current_interval
        return temp

    def sort(self, indlist, rootpitch=None):
        temp = self.copy()
        names = [temp[i - 1].name for i in indlist]
        if rootpitch is None:
            rootpitch = temp[indlist[0] - 1].num
        elif rootpitch == 'same':
            rootpitch = temp[0].num
        new_interval = [temp.interval[i - 1] for i in indlist]
        return chord(names,
                     rootpitch=rootpitch,
                     interval=new_interval,
                     start_time=temp.start_time)

    def voicing(self, rootpitch=None):
        if rootpitch is None:
            rootpitch = self[0].num
        duration, interval = [i.duration for i in self.notes], self.interval
        temp.names = self.names()
        return [
            chord(i,
                  rootpitch=rootpitch).standardize().set(duration, interval)
            for i in perm(temp.names)
        ]

    def inversion_highest(self, ind):
        if not 1 <= ind < len(self):
            raise ValueError(
                'the number of inversion is out of range of the notes in this chord'
            )
        temp = self.copy()
        ind -= 1
        while temp[ind].degree < temp[-1].degree:
            temp[ind] = temp[ind].up(octave)
        temp.notes.append(temp.notes.pop(ind))
        return temp

    def inoctave(self):
        temp = self.copy()
        root = self[0].degree
        for i in range(1, len(temp)):
            while temp[i].degree - root > octave:
                temp[i] = temp[i].down(octave)
        temp.notes.sort(key=lambda x: x.degree)
        return temp

    def on(self, root, duration=1 / 4, interval=None, each=0):
        temp = copy(self)
        if each == 0:
            if isinstance(root, chord):
                return root & self
            if isinstance(root, str):
                root = to_note(root)
                root.duration = duration
            temp.notes.insert(0, root)
            if interval is not None:
                temp.interval.insert(0, interval)
            else:
                temp.interval.insert(0, self.interval[0])
            return temp
        else:
            if isinstance(root, chord):
                root = list(root)
            else:
                root = [to_note(i) for i in root]
            return [self.on(x, duration, interval) for x in root]

    def up(self, unit=1, ind=None, ind2=None):
        temp = copy(self)
        if not isinstance(unit, (int, Interval)):
            temp.notes = [temp.notes[k].up(unit[k]) for k in range(len(unit))]
            return temp
        if not isinstance(ind, (int, Interval)) and ind is not None:
            temp.notes = [
                temp.notes[i].up(unit) if i in ind else temp.notes[i]
                for i in range(len(temp.notes))
            ]
            return temp
        if ind2 is None:
            if ind is None:
                temp.notes = [each.up(unit) for each in temp.notes]
            else:
                temp[ind] = temp[ind].up(unit)
        else:
            temp.notes = temp.notes[:ind] + [
                each.up(unit) for each in temp.notes[ind:ind2]
            ] + temp.notes[ind2:]
        return temp

    def down(self, unit=1, ind=None, ind2=None):
        if not isinstance(unit, (int, Interval)):
            unit = [-i for i in unit]
            return self.up(unit, ind, ind2)
        return self.up(-unit, ind, ind2)

    def sharp(self, unit=1):
        temp = copy(self)
        temp.notes = [i.sharp(unit=unit) for i in temp.notes]
        return temp

    def flat(self, unit=1):
        temp = copy(self)
        temp.notes = [i.flat(unit=unit) for i in temp.notes]
        return temp

    def omit(self, ind, mode=0):
        '''
        mode == 0: omit note as pitch interval with the first note
        mode == 1: omit note as number of semitones with the first note
        mode == 2: omit note as index of current chord
        '''
        if not isinstance(ind, list):
            ind = [ind]
        if mode == 0:
            ind = [self.interval_note(i) for i in ind]
        elif mode == 1:
            ind = [self.notes[0] + i for i in ind]
        if ind:
            if isinstance(ind[0], int):
                temp = copy(self)
                length = len(temp)
                temp.notes = [
                    temp.notes[k] for k in range(length) if k not in ind
                ]
                temp.interval = [
                    temp.interval[k] for k in range(length) if k not in ind
                ]
                return temp
            elif isinstance(ind[0], note) or (isinstance(ind[0], str) and any(
                    i for i in ind[0] if i.isdigit())):
                temp = self.same_accidentals()
                ind = chord(ind).same_accidentals().notes
                current_ind = [
                    k for k in range(len(temp)) if temp.notes[k] in ind
                ]
                return self.omit(current_ind, mode=2)
            elif isinstance(ind[0],
                            str) and not any(i for i in ind[0] if i.isdigit()):
                temp = self.standardize_note()
                self_temp.names = temp.names()
                ind = chord(ind).standardize_note().names()
                current_ind = [
                    k for k in range(len(self_temp.names))
                    if self_temp.names[k] in ind
                ]
                return self.omit(current_ind, mode=2)
            else:
                return self
        else:
            return self

    def sus(self, num=4):
        temp = self.copy()
        first_note = temp[0]
        if num == 4:
            temp.notes = [
                temp.notes[0] +
                P4 if abs(i.degree - first_note.degree) %
                octave
                in [major_third, minor_third] else i
                for i in temp.notes
            ]
        elif num == 2:
            temp.notes = [
                temp.notes[0] +
                M2 if abs(i.degree - first_note.degree) %
                octave
                in [major_third, minor_third] else i
                for i in temp.notes
            ]
        return temp

    def __setitem__(self, ind, value):
        if isinstance(value, str):
            value = to_note(value)
        self.notes[ind] = value
        if isinstance(value, chord):
            self.interval[ind] = value.interval

    def __delitem__(self, ind):
        del self.notes[ind]
        del self.interval[ind]

    def index(self, value):
        if isinstance(value, str):
            if value not in standard:
                value = to_note(value)
                if value not in self:
                    return -1
                return self.notes.index(value)
            else:
                note_names = self.names()
                if value not in note_names:
                    return -1
                return note_names.index(value)
        else:
            return self.index(str(value))

    def remove(self, note1):
        if isinstance(note1, str):
            note1 = to_note(note1)
        if note1 in self:
            inds = self.notes.index(note1)
            self.notes.remove(note1)
            del self.interval[inds]

    def append(self, value, interval=0):
        if isinstance(value, str):
            value = to_note(value)
        self.notes.append(value)
        self.interval.append(interval)

    def extend(self, values, intervals=0):
        if isinstance(values, chord):
            self.notes.extend(values.notes)
            self.interval.extend(values.interval)
        else:
            values = [
                to_note(value) if isinstance(value, str) else value
                for value in values
            ]
            if isinstance(intervals, int):
                intervals = [intervals for i in range(len(values))]
            self.notes.extend(values)
            self.interval.extend(intervals)

    def delete(self, ind):
        del self.notes[ind]
        del self.interval[ind]

    def insert(self, ind, value, interval=0):
        if isinstance(value, chord):
            self.notes[ind:ind] = value.notes
            self.interval[ind:ind] = value.interval
        else:
            if isinstance(value, str):
                value = to_note(value)
            self.notes.insert(ind, value)
            self.interval.insert(ind, interval)

    def replace_chord(self, ind1, ind2=None, value=None, mode=0):
        if not isinstance(value, chord):
            value = chord(value)
        if ind2 is None:
            ind2 = ind1 + len(value)
        if mode == 0:
            self.notes[ind1:ind2] = value.notes
            self.interval[ind1:ind2] = value.interval
        elif mode == 1:
            N = len(self.notes)
            for i in range(ind1, ind2):
                current_value = value.notes[i - ind1]
                if i < N:
                    current = self.notes[i]
                    current.name = current_value.name
                    current.num = current_value.num
                else:
                    self.notes[i:i] = [current_value]
                    self.interval[i:i] = [value.interval[i - ind1]]

    def drops(self, ind):
        temp = self.copy()
        dropnote = temp.notes.pop(-ind).down(octave)
        dropinterval = temp.interval.pop(-ind)
        temp.notes.insert(0, dropnote)
        temp.interval.insert(0, dropinterval)
        return temp

    def rest(self, length, dotted=None, ind=None):
        temp = copy(self)
        if dotted is not None and dotted != 0:
            length = length * sum([(1 / 2)**i for i in range(dotted + 1)])
        if not temp.notes:
            temp.start_time += length
            return temp
        if ind is None:
            last_interval = temp.interval[-1]
            if last_interval != 0:
                temp.interval[-1] += length
            else:
                temp.interval[-1] += (temp.notes[-1].duration + length)
        else:
            if ind == len(temp) - 1:
                last_interval = temp.interval[-1]
                if last_interval != 0:
                    temp.interval[-1] += length
                else:
                    temp.interval[-1] += (temp.notes[-1].duration + length)
            else:
                temp.interval[ind] += length
        return temp

    def modulation(self, old_scale, new_scale):
        # change notes (including both of melody and chords) in the given piece
        # of music from a given scale to another given scale, and return
        # the new changing piece of music

        # this modulation function only supports modulate from a scale with equal or more notes to another scale
        temp = copy(self)
        old_scale_names = [
            i if i not in standard_dict else standard_dict[i]
            for i in old_scale.names()
        ]
        new_scale_names = [
            i if i not in standard_dict else standard_dict[i]
            for i in new_scale.names()
        ]
        old_scale_names_len = len(old_scale_names)
        new_scale_names_len = len(new_scale_names)
        if new_scale_names_len < old_scale_names_len:
            new_scale_names += new_scale_names[-(old_scale_names_len -
                                                 new_scale_names_len):]
            new_scale_names.sort(key=lambda s: standard[s])
        number = len(new_scale_names)
        transdict = {
            old_scale_names[i]: new_scale_names[i]
            for i in range(number)
        }
        transdict = {
            standardize_note(i): standardize_note(j)
            for i, j in transdict.items()
        }
        for k in range(len(temp)):
            current = temp.notes[k]
            if current.name in standard_dict:
                current_name = standard_dict[current.name]
            else:
                current_name = current.name
            if current_name in transdict:
                current_note = closest_note(transdict[current_name],
                                               current)
                temp.notes[k] = current.reset(name=current_note.name,
                                              num=current_note.num)
        return temp

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return self.__getslice__(ind.start, ind.stop)
        return self.notes[ind]

    def __iter__(self):
        for i in self.notes:
            yield i

    def __getslice__(self, i, j):
        temp = copy(self)
        temp.notes = temp.notes[i:j]
        temp.interval = temp.interval[i:j]
        return temp

    def __len__(self):
        return len(self.notes)

    def set_volume(self, vol, ind='all'):
        if isinstance(ind, int):
            each = self.notes[ind]
            each.set_volume(vol)
        elif isinstance(ind, list):
            if isinstance(vol, list):
                for i in range(len(ind)):
                    current = ind[i]
                    each = self.notes[current]
                    each.set_volume(vol[i])
            elif isinstance(vol, (int, float)):
                vol = int(vol)
                for i in range(len(ind)):
                    current = ind[i]
                    each = self.notes[current]
                    each.set_volume(vol)
        elif ind == 'all':
            if isinstance(vol, list):
                for i in range(len(vol)):
                    current = self.notes[i]
                    current.set_volume(vol[i])
            elif isinstance(vol, (int, float)):
                vol = int(vol)
                for each in self.notes:
                    each.set_volume(vol)

    def move(self, x):
        # x could be a dict or list of (index, move_steps)
        temp = self.copy()
        if isinstance(x, dict):
            for i in x:
                temp.notes[i] = temp.notes[i].up(x[i])
            return temp
        if isinstance(x, list):
            for i in x:
                temp.notes[i[0]] = temp.notes[i[0]].up(i[1])
            return temp

    def clear_at(self, duration=0, interval=None, volume=None):
        temp = copy(self)
        i = 0
        while i < len(temp):
            current = temp[i]
            if duration is not None:
                if current.duration <= duration:
                    tedelete(i)
                    continue
            if interval is not None:
                if temp.interval[i] <= interval:
                    tedelete(i)
                    continue
            if volume is not None:
                if current.volume <= volume:
                    tedelete(i)
                    continue
            i += 1
        return temp

    def retrograde(self):
        temp = self.copy()
        tempo_changes = temp.tempos
        if tempo_changes:
            tenormalize_tempo(tempo_changes[0].bpm)
        result = tereverse()
        return result

    def pitch_inversion(self):
        pitch_bend_changes = self.pitch_bends
        temp = self.copy()
        temp.clear_pitch_bend()
        tempo_changes = temp.tempos
        if tempo_changes:
            tenormalize_tempo(tempo_changes[0].bpm)
        volumes = teget_volume()
        pitch_intervals = temp.intervalof(cumulative=False)
        result = get_chord_by_interval(temp[0],
                                          [-i for i in pitch_intervals],
                                          temp.get_duration(), temp.interval,
                                          False)
        result.set_volume(volumes)
        result.pitch_bends += pitch_bend_changes
        return result

    def normalize_tempo(self,
                        bpm,
                        start_time=0,
                        pan_msg=None,
                        volume_msg=None,
                        original_bpm=None):
        # choose a bpm and apply to all of the notes, if there are tempo
        # changes, use relative ratios of the chosen bpms and changes bpms
        # to re-calculate the notes durations and intervals
        if original_bpm is not None:
            self.tempos.append(tempo(bpm=original_bpm, start_time=0))
        if all(i.bpm == bpm for i in self.tempos):
            self.clear_tempo()
            return
        if start_time > 0:
            self.notes.insert(0, note('C', 5, duration=0))
            self.interval.insert(0, start_time)
        tempo_changes = copy(self.tempos)
        tempo_changes.insert(0, tempo(bpm=bpm, start_time=0))
        self.clear_tempo()
        tempo_changes.sort(key=lambda s: s.start_time)
        new_tempo_changes = [tempo_changes[0]]
        for i in range(len(tempo_changes) - 1):
            current_tempo = tempo_changes[i]
            next_tempo = tempo_changes[i + 1]
            if next_tempo.start_time == current_tempo.start_time:
                new_tempo_changes[-1] = next_tempo
            else:
                new_tempo_changes.append(next_tempo)
        tempo_changes_ranges = [
            (new_tempo_changes[i].start_time,
             new_tempo_changes[i + 1].start_time, new_tempo_changes[i].bpm)
            for i in range(len(new_tempo_changes) - 1)
        ]
        tempo_changes_ranges.append(
            (new_tempo_changes[-1].start_time, self.bars(mode=1),
             new_tempo_changes[-1].bpm))
        pitch_bend_msg = copy(self.pitch_bends)
        self.clear_pitch_bend()
        _process_normalize_tempo(self, tempo_changes_ranges, bpm)
        other_types = pitch_bend_msg + self.other_messages
        if pan_msg:
            other_types += pan_msg
        if volume_msg:
            other_types += volume_msg
        other_types.sort(key=lambda s: s.start_time)
        other_types.insert(0, pitch_bend(value=0, start_time=0))
        other_types_interval = [
            other_types[i + 1].start_time - other_types[i].start_time
            for i in range(len(other_types) - 1)
        ]
        other_types_interval.append(0)
        other_types_chord = chord([])
        other_types_chord.notes = other_types
        other_types_chord.interval = other_types_interval
        _process_normalize_tempo(other_types_chord,
                                 tempo_changes_ranges,
                                 bpm,
                                 mode=1)
        new_pitch_bends = []
        new_pan = []
        new_volume = []
        new_other_messages = []
        for i in range(len(other_types_chord.notes)):
            each = other_types_chord.notes[i]
            current_start_time = sum(other_types_chord.interval[:i])
            each.start_time = current_start_time
        del other_types_chord[0]
        for each in other_types_chord.notes:
            if isinstance(each, pitch_bend):
                new_pitch_bends.append(each)
            elif isinstance(each, pan):
                new_pan.append(each)
            elif isinstance(each, volume):
                new_volume.append(each)
            else:
                new_other_messages.append(each)
        self.pitch_bends.extend(new_pitch_bends)
        result = [new_other_messages]
        if new_pan or new_volume:
            result += [new_pan, new_volume]
        if start_time > 0:
            start_time = self.interval[0]
            del self.notes[0]
            del self.interval[0]
        return result, start_time

    def place_shift(self, time=0, pan_msg=None, volume_msg=None):
        temp = copy(self)
        for i in temp.tempos:
            i.start_time += time
            if i.start_time < 0:
                i.start_time = 0
        for i in temp.pitch_bends:
            i.start_time += time
            if i.start_time < 0:
                i.start_time = 0
        if pan_msg:
            for each in pan_msg:
                each.start_time += time
                if each.start_time < 0:
                    each.start_time = 0
        if volume_msg:
            for each in volume_msg:
                each.start_time += time
                if each.start_time < 0:
                    each.start_time = 0
        for each in temp.other_messages:
            each.start_time += time
            if each.start_time < 0:
                each.start_time = 0
        if pan_msg or volume_msg:
            return temp, pan_msg, volume_msg
        else:
            return temp

    def info(self, **detect_args):
        chord_type = self.detect(get_chord_type=True, **detect_args)
        return chord_type

    def same_accidentals(self, mode='#'):
        temp = copy(self)
        for each in temp.notes:
            each.name = standardize_note(each.name)
            if mode == '#':
                if len(each.name) > 1 and each.name[-1] == 'b':
                    each.name = standard_dict[each.name]
            elif mode == 'b':
                if each.name[-1] == '#':
                    each.name = reverse_standard_dict[each.name]
        return temp

    def filter(self, cond, action=None, mode=0, action_mode=0):
        temp = self.copy()
        available_inds = [k for k in range(len(temp)) if cond(temp.notes[k])]
        if mode == 1:
            return available_inds
        if action is None:
            if available_inds:
                new_interval = []
                N = len(available_inds) - 1
                for i in range(N):
                    new_interval.append(
                        sum(temp.interval[available_inds[i]:available_inds[i + 1]]))
                new_interval.append(sum(temp.interval[available_inds[-1]:]))
                new_notes = [temp.notes[j] for j in available_inds]
                result = chord(new_notes, interval=new_interval)
                start_time = sum(temp.interval[:available_inds[0]])
            else:
                result = chord([])
                start_time = 0
            return result, start_time
        else:
            if action_mode == 0:
                for each in available_inds:
                    temp.notes[each] = action(temp.notes[each])
            elif action_mode == 1:
                for each in available_inds:
                    action(temp.notes[each])
            return temp

    def pitch_filter(self, x='A0', y='C8'):
        if isinstance(x, str):
            x = trans_note(x)
        if isinstance(y, str):
            y = trans_note(y)
        if all(x.degree <= i.degree <= y.degree for i in self.notes):
            return self, 0
        temp = self.copy()
        available_inds = [
            k for k in range(len(temp))
            if x.degree <= temp.notes[k].degree <= y.degree
        ]
        if available_inds:
            new_interval = []
            N = len(available_inds) - 1
            for i in range(N):
                new_interval.append(
                    sum(temp.interval[available_inds[i]:available_inds[i +
                                                                       1]]))
            new_interval.append(sum(temp.interval[available_inds[-1]:]))
            new_notes = [temp.notes[j] for j in available_inds]
            start_time = sum(temp.interval[:available_inds[0]])
            temp.notes = new_notes
            temp.interval = new_interval

        else:
            temp.notes.clear()
            temp.interval.clear()
            start_time = 0
        return temp, start_time

    def interval_note(self, interval, mode=0):
        interval = str(interval)
        if mode == 0:
            if interval in degree_match:
                self_notes_degrees = [i.degree for i in self.notes]
                degrees = degree_match[interval]
                for each in degrees:
                    current_note = self.notes[0] + each
                    if current_note.degree in self_notes_degrees:
                        return current_note
            if interval in precise_degree_match:
                self_notes_degrees = [i.degree for i in self.notes]
                degrees = precise_degree_match[interval]
                current_note = self.notes[0] + degrees
                if current_note.degree in self_notes_degrees:
                    return current_note
        elif mode == 1:
            if interval in precise_degree_match:
                interval = precise_degree_match[interval]
                return self.notes[0] + interval

    def note_interval(self, current_note, mode=0):
        if isinstance(current_note, str):
            if not any(i.isdigit() for i in current_note):
                current_note = to_note(current_note)
                if standard[self[0].name] == standard[
                        current_note.name]:
                    current_interval = 0
                else:
                    current_chord = chord([self[0].name, current_note.name])
                    current_interval = current_chord[1].degree - current_chord[
                        0].degree
            else:
                current_note = to_note(current_note)
                current_interval = current_note.degree - self[0].degree
        else:
            current_interval = current_note.degree - self[0].degree
        if mode == 0:
            if current_interval in reverse_precise_degree_number_match:
                return reverse_precise_degree_number_match[
                    current_interval]
        elif mode == 1:
            return INTERVAL[current_interval]

    def get_voicing(self, voicing, mode=0):
        notes = [self.interval_note(i, mode=mode).name for i in voicing]
        pitch = self.notes[self.names().index(notes[0])].num
        return chord(notes, rootpitch=pitch)

    def near_voicing(self,
                     other,
                     keep_root=True,
                     standardize=True,
                     choose_nearest=False,
                     get_distance=False):
        if choose_nearest:
            result1, distance1 = self.near_voicing(other,
                                                   keep_root=True,
                                                   standardize=standardize,
                                                   choose_nearest=False,
                                                   get_distance=True)
            result2, distance2 = self.near_voicing(other,
                                                   keep_root=False,
                                                   standardize=standardize,
                                                   choose_nearest=False,
                                                   get_distance=True)
            result = result2 if distance2 < distance1 else result1
            return result if not get_distance else (result,
                                                    min(distance1, distance2))
        if standardize:
            temp = self.standardize()
            other = other.standardize()
        else:
            temp = copy(self)
        original_duration = temp.get_duration()
        original_volume = teget_volume()
        if keep_root:
            root_note = temp.notes[0]
            other_root_note = other.notes[0]
            new_root_note, current_distance = closest_note(
                root_note, other_root_note, get_distance=True)
            remain_notes = []
            current_other_notes = other.notes[1:]
            total_distance = current_distance
            for each in temp.notes[1:]:
                current_closest_note, current_distance = closest_note_from_chord(
                    each, current_other_notes, get_distance=True)
                total_distance += current_distance
                current_other_notes.remove(current_closest_note)
                new_note = closest_note(each, current_closest_note)
                remain_notes.append(new_note)
            remain_notes.insert(0, new_root_note)
        else:
            remain_notes = []
            current_other_notes = other.notes
            total_distance = 0
            for each in temp.notes:
                current_closest_note, current_distance = closest_note_from_chord(
                    each, current_other_notes, get_distance=True)
                total_distance += current_distance
                current_other_notes.remove(current_closest_note)
                new_note = closest_note(each, current_closest_note)
                remain_notes.append(new_note)
        temp.notes = remain_notes
        temp = tesortchord()
        temp = teset(duration=original_duration, volume=original_volume)
        return temp if not get_distance else (temp, total_distance)

    def reset_octave(self, num):
        diff = num - self[0].num
        return self + diff * octave

    def reset_pitch(self, pitch):
        if isinstance(pitch, str):
            pitch = to_note(pitch)
        return self + (pitch.degree - self[0].degree)

    def reset_same_octave(self, octave):
        temp = copy(self)
        for each in temp.notes:
            each.num = octave
        return temp

    def reset_same_channel(self, channel=None):
        for each in self.notes:
            each.channel = channel

    def with_same_channel(self, channel=None):
        temp = copy(self)
        tereset_same_channel(channel)
        return temp

    def with_other_messages(self, other_messages):
        temp = copy(self)
        temp.other_messages = other_messages
        return temp

    def clear_program_change(self):
        self.other_messages = [
            i for i in self.other_messages if i.type != 'program_change'
        ]

    def clear_other_messages(self, types=None):
        if types is None:
            self.other_messages.clear()
        else:
            self.other_messages = [
                i for i in self.other_messages if i.type != types
            ]

    def dotted(self, ind=-1, num=1, duration=True, interval=False):
        temp = copy(self)
        if num == 0:
            return temp
        if duration:
            if isinstance(ind, list):
                for each in ind:
                    temp.notes[
                        each].duration = temp.notes[each].duration * sum(
                            [(1 / 2)**i for i in range(num + 1)])
            elif ind == 'all':
                for each in range(len(temp.notes)):
                    temp.notes[
                        each].duration = temp.notes[each].duration * sum(
                            [(1 / 2)**i for i in range(num + 1)])
            else:
                temp.notes[ind].duration = temp.notes[ind].duration * sum(
                    [(1 / 2)**i for i in range(num + 1)])
        if interval:
            if isinstance(ind, list):
                for each in ind:
                    temp.interval[each] = temp.interval[each] * sum(
                        [(1 / 2)**i for i in range(num + 1)])
            elif ind == 'all':
                for each in range(len(temp.notes)):
                    temp.interval[each] = temp.interval[each] * sum(
                        [(1 / 2)**i for i in range(num + 1)])
            else:
                temp.interval[ind] = temp.interval[ind] * sum(
                    [(1 / 2)**i for i in range(num + 1)])
        return temp

    def apply_start_time_to_changes(self, start_time, msg=False):
        for each in self.tempos:
            each.start_time += start_time
            if each.start_time < 0:
                each.start_time = 0
        for each in self.pitch_bends:
            each.start_time += start_time
            if each.start_time < 0:
                each.start_time = 0
        if msg:
            for each in self.other_messages:
                each.start_time += start_time
                if each.start_time < 0:
                    each.start_time = 0

    def with_start(self, start_time):
        temp = copy(self)
        temp.start_time = start_time
        return temp

    def reset_channel(self,
                      channel,
                      reset_msg=True,
                      reset_pitch_bend=True,
                      reset_note=True):
        if reset_msg:
            for i in self.other_messages:
                if hasattr(i, 'channel'):
                    i.channel = channel
        if reset_note:
            for i in self.notes:
                i.channel = channel
        if reset_pitch_bend:
            for i in self.pitch_bends:
                i.channel = channel

    def reset_track(self, track, reset_msg=True, reset_pitch_bend=True):
        if reset_msg:
            for i in self.other_messages:
                i.track = track
        if reset_pitch_bend:
            for i in self.pitch_bends:
                i.track = track

    def pick(self, indlist):
        temp = copy(self)
        whole_notes = temp.notes
        new_interval = []
        whole_interval = temp.interval
        M = len(indlist) - 1
        for i in range(M):
            new_interval.append(sum(whole_interval[indlist[i]:indlist[i + 1]]))
        new_interval.append(sum(whole_interval[indlist[-1]:]))
        start_time = temp[:indlist[0]].bars(mode=0)
        return chord([whole_notes[j] for j in indlist],
                     interval=new_interval,
                     start_time=start_time,
                     other_messages=temp.other_messages)

    def remove_duplicates(self):
        temp = copy(self)
        inds = []
        degrees = []
        notes = []
        intervals = []
        for i, each in enumerate(temp.notes):
            if each.degree not in degrees:
                degrees.append(each.degree)
                notes.append(each)
                intervals.append(temp.interval[i])
        temp.notes = notes
        temp.interval = intervals
        return temp

    def delete_track(self, current_ind):
        self.tempos = [i for i in self.tempos if i.track != current_ind]
        self.pitch_bends = [
            i for i in self.pitch_bends if i.track != current_ind
        ]
        for i in self.tempos:
            if i.track is not None and i.track > current_ind:
                i.track -= 1
        for i in self.pitch_bends:
            if i.track is not None and i.track > current_ind:
                i.track -= 1
        self.other_messages = [
            i for i in self.other_messages if i.track != current_ind
        ]
        for i in self.other_messages:
            if i.track > current_ind:
                i.track -= 1

    def delete_channel(self, current_ind):
        available_inds = [
            i for i, each in enumerate(self.notes)
            if each.channel != current_ind
        ]
        self.notes = [self.notes[i] for i in available_inds]
        self.interval = [self.interval[i] for i in available_inds]
        self.tempos = [i for i in self.tempos if i.channel != current_ind]
        self.pitch_bends = [
            i for i in self.pitch_bends if i.channel != current_ind
        ]
        self.other_messages = [
            i for i in self.other_messages
            if not (hasattr(i, 'channel') and i.channel == current_ind)
        ]

    def to_piece(self, *args, **kwargs):
        return chord_to_piece(self, *args, **kwargs)

    def apply_rhythm(self, current_rhythm, set_duration=True):
        temp = copy(self)
        length = len(temp)
        counter = -1
        has_beat = False
        current_start_time = 0
        for i, each in enumerate(current_rhythm):
            current_duration = each.get_duration()
            if type(each) is beat:
                has_beat = True
                counter += 1
                if counter >= length:
                    break
                temp.interval[counter] = current_duration
                if set_duration:
                    if current_duration != 0:
                        temp.notes[counter].duration = current_duration
            elif type(each) is rest_symbol:
                if not has_beat:
                    current_start_time += current_duration
                else:
                    temp.interval[counter] += current_duration
            elif type(each) is continue_symbol:
                if not has_beat:
                    current_start_time += current_duration
                else:
                    temp.interval[counter] += current_duration
                    temp.notes[counter].duration += current_duration
        temp.start_time = current_start_time
        return temp

    def from_rhythm(self, current_rhythm, set_duration=True):
        return get_chords_from_rhythm(chords=self,
                                         current_rhythm=current_rhythm,
                                         set_duration=set_duration)

    def fix_length(self, n, round_duration=False, round_cut_interval=False):
        current_bar = self.bars(mode=2, start_time=self.start_time)
        if current_bar < n:
            extra = n - current_bar
            result = self | extra
        elif current_bar > n:
            result = self.cut(0,
                              n,
                              start_time=self.start_time,
                              cut_extra_duration=True,
                              cut_extra_interval=True,
                              round_duration=round_duration,
                              round_cut_interval=round_cut_interval)
        else:
            result = copy(self)
        return result

    def is_empty(self):
        return not self.notes and not self.tempos and not self.pitch_bends and not self.other_messages


class scale:
    '''
    This class represents a scale.
    '''

    def __init__(self,
                 start=None,
                 mode=None,
                 interval=None,
                 notes=None,
                 standard_interval=True):
        self.interval = interval
        self.notes = None
        if notes is not None:
            notes = [to_note(i) if isinstance(i, str) else i for i in notes]
            self.notes = notes
            self.start = notes[0]
            self.mode = mode
        else:
            if isinstance(start, str):
                start = trans_note(start)
            self.start = start
            if mode is not None:
                self.mode = mode.lower()
            else:
                self.mode = mode
            self.notes = self.get_scale().notes

        if interval is None:
            self.interval = self.get_interval(
                standard_interval=standard_interval)
        if mode is None:
            current_mode = detect_scale_type(self.interval,
                                                    mode='interval')
            if current_mode is not None:
                self.mode = current_mode

    def set_mode_name(self, name):
        self.mode = name

    def change_interval(self, interval):
        self.interval = interval

    def __repr__(self):
        return f'[scale]\nscale name: {self.start} {self.mode} scale\nscale intervals: {self.get_interval()}\nscale notes: {self.get_scale().notes}'

    def __eq__(self, other):
        return type(other) is scale and self.notes == other.notes

    def get_scale_name(self, with_octave=True):
        return f'{self.start if with_octave else self.start.name} {self.mode} scale'

    def standard(self):
        if len(self) == 8:
            standard_notes = [i.name for i in copy(self.notes)[:-1]]
            compare_notes = [i.name for i in scale('C', 'major').notes[:-1]]
            inds = compare_notes.index(standard_notes[0][0])
            compare_notes = compare_notes[inds:] + compare_notes[:inds]
            standard_notes = [
                relative_note(standard_notes[i], compare_notes[i])
                for i in range(7)
            ]
            return standard_notes
        else:
            return self.names()

    def __contains__(self, note1):
        names = self.names()
        names = [
            standard_dict[i] if i in standard_dict else i
            for i in names
        ]
        if isinstance(note1, chord):
            chord_names = note1.names()
            chord_names = [
                standard_dict[i] if i in standard_dict else i
                for i in chord_names
            ]
            return all(i in names for i in chord_names)
        else:
            if isinstance(note1, note):
                note1 = note1.name
            else:
                note1 = trans_note(note1).name
            return (standard_dict[note1]
                    if note1 in standard_dict else note1) in names

    def __getitem__(self, ind):
        return self.notes[ind]

    def __iter__(self):
        for i in self.notes:
            yield i

    def __call__(self, n, duration=1 / 4, interval=0, num=3, step=2):
        if isinstance(n, int):
            return self.pick_chord_by_degree(n, duration, interval, num, step)
        elif isinstance(n, str):
            altered_notes = n.replace(' ', '').split(',')
            notes = copy(self.notes)
            for each in altered_notes:
                if each.startswith('#'):
                    current_ind = int(each.split('#')[1]) - 1
                    notes[current_ind] = notes[current_ind].sharp()
                elif each.startswith('b'):
                    current_ind = int(each.split('b')[1]) - 1
                    notes[current_ind] = notes[current_ind].flat()
            return scale(notes=notes)

    def get_interval(self, standard_interval=True):
        if self.mode is None:
            if self.interval is None:
                if self.notes is None:
                    raise ValueError(
                        'a mode or interval or notes list should be settled')
                else:
                    notes = self.notes
                    if not standard_interval:
                        root_degree = notes[0].degree
                        return [
                            notes[i].degree - notes[i - 1].degree
                            for i in range(1, len(notes))
                        ]
                    else:
                        start = notes[0]
                        return [
                            get_pitch_interval(notes[i - 1], notes[i])
                            for i in range(1, len(notes))
                        ]
            else:
                return self.interval
        else:
            if self.interval is not None:
                return self.interval
            mode = self.mode.lower()
            if mode in scaleTypes:
                return scaleTypes[mode]
            else:
                if self.notes is None:
                    raise ValueError(f'could not find scale {self.mode}')
                else:
                    notes = self.notes
                    if not standard_interval:
                        root_degree = notes[0].degree
                        return [
                            notes[i].degree - notes[i - 1].degree
                            for i in range(1, len(notes))
                        ]
                    else:
                        start = notes[0]
                        return [
                            get_pitch_interval(notes[i - 1], notes[i])
                            for i in range(1, len(notes))
                        ]

    def get_scale(self, intervals=1 / 4, durations=None):
        if self.mode is None:
            if self.interval is None:
                raise ValueError(
                    'at least one of mode or interval in the scale should be settled'
                )
            else:
                result = [self.start]
                start = copy(self.start)
                for t in self.interval:
                    start += t
                    result.append(start)
                if (result[-1].degree - result[0].degree) % 12 == 0:
                    result[-1].name = result[0].name
                return chord(result, duration=durations, interval=intervals)
        else:
            result = [self.start]
            start = copy(self.start)
            interval1 = self.get_interval()
            for t in interval1:
                start += t
                result.append(start)
            if (result[-1].degree - result[0].degree) % 12 == 0:
                result[-1].name = result[0].name
            return chord(result, duration=durations, interval=intervals)

    def __len__(self):
        return len(self.notes)

    def names(self, standardize_note=False):
        temp = [x.name for x in self.notes]
        result = []
        for i in temp:
            if i not in result:
                result.append(i)
        if standardize_note:
            result = [standardize_note(i) for i in result]
        return result

    def pick_chord_by_degree(self,
                             degree1,
                             duration=1 / 4,
                             interval=0,
                             num=3,
                             step=2,
                             standardize=False):
        result = []
        high = False
        if degree1 == 7:
            degree1 = 0
            high = True
        temp = copy(self)
        scale_notes = temp.notes[:-1]
        for i in range(degree1, degree1 + step * num, step):
            result.append(scale_notes[i % 7].name)
        result_chord = chord(result,
                             rootpitch=temp[0].num,
                             interval=interval,
                             duration=duration)
        if standardize:
            result_chord = result_chord.standardize()
        if high:
            result_chord = result_chord + octave
        return result_chord

    def pattern(self, indlist, duration=1 / 4, interval=0, num=3, step=2):
        if isinstance(indlist, str):
            indlist = [int(i) for i in indlist]
        elif isinstance(indlist, int):
            indlist = [int(i) for i in str(indlist)]
        return [
            self(n - 1, num=num, step=step).set(duration, interval)
            for n in indlist
        ]

    def __mod__(self, x):
        if isinstance(x, (int, str)):
            x = [x]
        return self.pattern(*x)

    def dom(self):
        return self[4]

    def dom_mode(self):
        if self.mode is not None:
            return scale(self[4], mode=self.mode)
        else:
            return scale(self[4], interval=self.get_interval())

    def fifth(self, step=1, inner=False):
        # move the scale on the circle of fifths by number of steps,
        # if the step is > 0, then move clockwise,
        # if the step is < 0, then move counterclockwise,
        # if inner is True: pick the inner scales from the circle of fifths,
        # i.e. those minor scales.
        return circle_of_fifths().rotate_get_scale(self[0].name,
                                                   step,
                                                   pitch=self[0].num,
                                                   inner=inner)

    def fourth(self, step=1, inner=False):
        # same as fifth but instead of circle of fourths
        # Maybe someone would notice that circle of fourths is just
        # the reverse of circle of fifths.
        return circle_of_fourths().rotate_get_scale(self[0].name,
                                                    step,
                                                    pitch=self[0].num,
                                                    inner=inner)

    def tonic(self):
        return self[0]

    def supertonic(self):
        return self[1]

    def mediant(self):
        return self[2]

    def subdominant(self):
        return self[3]

    def dominant(self):
        return self[4]

    def submediant(self):
        return self[5]

    def leading_tone(self):
        return self[0].up(major_seventh)

    def subtonic(self):
        return self[0].up(minor_seventh)

    def tonic_chord(self):
        return self(0)

    def subdom(self):
        return self[3]

    def subdom_chord(self):
        return self(3)

    def dom_chord(self):
        return self(4)

    def dom7_chord(self):
        return self(4) + self[3].up(12)

    def leading_chord(self):
        return chord([self[6].down(octave), self[1], self[3]])

    def leading7_chord(self):
        return chord(
            [self[6].down(octave), self[1], self[3], self[5]])

    def scale_from(self, degree=4, mode=None, interval=None):
        # default is pick the dominant mode of the scale
        if mode is None and interval is None:
            mode, interval = self.mode, self.interval
        return scale(self[degree], mode, interval)

    def secondary_dom(self, degree=4):
        newscale = self.scale_from(degree, 'major')
        return newscale.dom_chord()

    def secondary_dom7(self, degree=4):
        return self.scale_from(degree, 'major').dom7_chord()

    def secondary_leading(self, degree=4):
        return self.scale_from(degree, 'major').leading_chord()

    def secondary_leading7(self, degree=4):
        return self.scale_from(degree, 'major').leading7_chord()

    def pick_chord_by_index(self, indlist):
        return chord([self[i] for i in indlist])

    def detect(self):
        return detect_scale_type(self)

    def get_all_chord(self, duration=None, interval=0, num=3, step=2):
        return [
            self.pick_chord_by_degree(i,
                                      duration=duration,
                                      interval=interval,
                                      num=num,
                                      step=step)
            for i in range(len(self.get_interval()) + 1)
        ]

    def relative_key(self):
        if self.mode == 'major':
            return scale(self[5].reset_octave(self[0].num), 'minor')
        elif self.mode == 'minor':
            return scale(self[2].reset_octave(self[0].num), 'major')
        else:
            raise ValueError(
                'this function only applies to major and minor scales')

    def parallel_key(self):
        if self.mode == 'major':
            return scale(self[0], 'minor')
        elif self.mode == 'minor':
            return scale(self[0], 'major')
        else:
            raise ValueError(
                'this function only applies to major and minor scales')

    def get_note_from_degree(self, degree, pitch=None):
        if degree < 1:
            raise ValueError('scale degree starts from 1')
        extra_num, current_degree = divmod(degree - 1, 7)
        result = self[current_degree]
        if pitch is not None:
            result = result.reset_octave(pitch)
        result += extra_num * octave
        return result

    def get_chord(self, degree, chord_type=None, natural=False):
        current_accidental = None
        original_degree = degree
        if degree.startswith('#') or degree.startswith('b'):
            current_accidental = degree[0]
            degree = degree[1:]
        if not chord_type:
            current_keys = list(roman_numerals_dict.keys())
            current_keys.sort(key=lambda s: len(s[0]), reverse=True)
            found = False
            for each in current_keys:
                for i in each:
                    if degree.startswith(i):
                        found = True
                        chord_type = degree[len(i):]
                        degree = i
                        break
                if found:
                    break
            if not found:
                raise ValueError(
                    f'{original_degree} is not a valid roman numerals chord representation'
                )
        if degree not in roman_numerals_dict:
            raise ValueError(
                f'{original_degree} is not a valid roman numerals chord representation'
            )
        current_degree = roman_numerals_dict[degree] - 1
        current_note = self[current_degree].name
        if natural:
            temp = C(current_note + chord_type)
            if not isinstance(temp, chord):
                raise ValueError(f'{chord_type} is not a valid chord type')
            length = len(temp)
            return self.pick_chord_by_degree(current_degree, num=length)
        if degree.islower():
            try:
                result = C(current_note + 'm' + chord_type)
            except:
                result = C(current_note + chord_type)
        else:
            result = C(current_note + chord_type)
        if current_accidental is not None:
            if current_accidental == '#':
                result += 1
            elif current_accidental == 'b':
                result -= 1
        return result

    def up(self, unit=1, ind=None, ind2=None):
        if ind2 is not None:
            notes = copy(self.notes)
            return scale(notes=[
                notes[i].up(unit) if ind <= i < ind2 else notes[i]
                for i in range(len(notes))
            ])
        if ind is None:
            return scale(self[0].up(unit), self.mode, self.interval)
        else:
            notes = copy(self.notes)
            if isinstance(ind, int):
                notes[ind] = notes[ind].up(unit)
            else:
                notes = [
                    notes[i].up(unit) if i in ind else notes[i]
                    for i in range(len(notes))
                ]
            result = scale(notes=notes)
            return result

    def down(self, unit=1, ind=None, ind2=None):
        return self.up(-unit, ind, ind2)

    def sharp(self, unit=1):
        return scale(self[0].sharp(unit), self.mode, self.interval)

    def flat(self, unit=1):
        return scale(self[0].flat(unit), self.mode, self.interval)

    def __pos__(self):
        return self.up()

    def __neg__(self):
        return self.down()

    def __invert__(self):
        return scale(self[0], interval=list(reversed(self.interval)))

    def reverse(self):
        return ~self

    def move(self, x):
        notes = copy(self.get_scale())
        return scale(notes=notes.move(x))

    def inversion(self, ind, parallel=False, start=None):
        # return the inversion of a scale with the beginning note of a given index
        if ind < 1:
            raise ValueError('inversion of scale starts from 1')
        ind -= 1
        interval1 = self.get_interval()
        new_interval = interval1[ind:] + interval1[:ind]
        if parallel:
            start1 = self.start
        else:
            if start is not None:
                start1 = start
            else:
                start1 = self.get_scale().notes[ind]
        result = scale(start=start1, interval=new_interval)
        result.mode = result.detect()
        return result

    def play(self, intervals=1 / 4, durations=None, *args, **kwargs):
        play(self.get_scale(intervals, durations), *args, **kwargs)

    def __add__(self, obj):
        if isinstance(obj, (int, Interval)):
            return self.up(obj)
        elif isinstance(obj, tuple):
            return self.up(*obj)

    def __sub__(self, obj):
        if isinstance(obj, (int, Interval)):
            return self.down(obj)
        elif isinstance(obj, tuple):
            return self.down(*obj)

    def chord_progression(self,
                          chords,
                          durations=1 / 4,
                          intervals=0,
                          volumes=None,
                          chords_interval=None,
                          merge=True):
        current_keys = list(roman_numerals_dict.keys())
        current_keys.sort(key=lambda s: len(s[0]), reverse=True)
        for k in range(len(chords)):
            current_chord = chords[k]
            if isinstance(current_chord, (tuple, list)):
                current_degree_name = current_chord[0]
                current_accidental = None
                if current_degree_name.startswith(
                        '#') or current_degree_name.startswith('b'):
                    current_accidental = current_degree_name[0]
                    current_degree_name = current_degree_name[1:]
                if current_degree_name not in roman_numerals_dict:
                    raise ValueError(
                        f'{"".join(current_chord)} is not a valid roman numerals chord representation'
                    )
                current_degree = roman_numerals_dict[
                    current_degree_name] - 1
                current_note = self[current_degree]
                if current_accidental is not None:
                    if current_accidental == '#':
                        current_note += 1
                    elif current_accidental == 'b':
                        current_note -= 1
                if current_degree_name.islower():
                    try:
                        current_chord_name = current_note.name + 'm' + current_chord[
                            1]
                        temp = C(current_chord_name)
                    except:
                        current_chord_name = current_note.name + current_chord[
                            1]
                else:
                    current_chord_name = current_note.name + current_chord[1]
                chords[k] = current_chord_name
            else:
                found = False
                current_degree = None
                current_accidental = None
                original_current_chord = current_chord
                if current_chord.startswith('#') or current_chord.startswith(
                        'b'):
                    current_accidental = current_chord[0]
                    current_chord = current_chord[1:]
                for each in current_keys:
                    for i in each:
                        if current_chord.startswith(i):
                            found = True
                            current_degree_name = i
                            current_degree = roman_numerals_dict[i] - 1
                            current_note = self[current_degree]
                            if current_accidental is not None:
                                if current_accidental == '#':
                                    current_note += 1
                                elif current_accidental == 'b':
                                    current_note -= 1
                            if current_degree_name.islower():
                                try:
                                    current_chord_name = current_note.name + 'm' + current_chord[
                                        len(i):]
                                    temp = C(current_chord_name)
                                except:
                                    current_chord_name = current_note.name + current_chord[
                                        len(i):]
                            else:
                                current_chord_name = current_note.name + current_chord[
                                    len(i):]
                            chords[k] = current_chord_name
                            break
                    if found:
                        break
                if not found:
                    raise ValueError(
                        f'{original_current_chord} is not a valid roman numerals chord representation'
                    )
        return chord_progression(chords, durations, intervals, volumes,
                                    chords_interval, merge)

    def reset_octave(self, num):
        return scale(self.start.reset_octave(num), self.mode, self.interval)

    def reset_pitch(self, name):
        return scale(self.start.reset_pitch(name), self.mode, self.interval)

    def reset_mode(self, mode):
        return scale(self.start, mode=mode)

    def reset_interval(self, interval):
        return scale(self.start, interval=interval)

    def set(self, start=None, num=None, mode=None, interval=None):
        temp = copy(self)
        if start is None:
            start = testart
        else:
            if isinstance(start, str):
                start = trans_note(start)
        if num is not None:
            start.num = num
        if mode is None and interval is None:
            mode = temode
            interval = temp.interval
        return scale(start, mode, interval)

    def __truediv__(self, n):
        if isinstance(n, tuple):
            return self.inversion(*n)
        else:
            return self.inversion(n)

    def _parse_scale_text(self, text, rootpitch, pitch_mode=0):
        octaves = None
        if '.' in text:
            text, octaves = text.split('.', 1)
        if text.endswith('#'):
            current_degree = int(text[:-1])
            if pitch_mode == 0:
                result = self.get_note_from_degree(current_degree,
                                                   pitch=rootpitch) + 1
            elif pitch_mode == 1:
                extra_num, current_degree_in_scale = divmod(
                    current_degree - 1, 7)
                diff = self[current_degree_in_scale].degree - self[0].degree
                result = self.get_note_from_degree(
                    1,
                    pitch=rootpitch) + 1 + diff + extra_num * octave
        elif text.endswith('b'):
            current_degree = int(text[:-1])
            if pitch_mode == 0:
                result = self.get_note_from_degree(current_degree,
                                                   pitch=rootpitch) - 1
            elif pitch_mode == 1:
                extra_num, current_degree_in_scale = divmod(
                    current_degree - 1, 7)
                diff = self[current_degree_in_scale].degree - self[0].degree
                result = self.get_note_from_degree(
                    1,
                    pitch=rootpitch) - 1 + diff + extra_num * octave
        else:
            current_degree = int(text)
            if pitch_mode == 0:
                result = self.get_note_from_degree(current_degree,
                                                   pitch=rootpitch)
            elif pitch_mode == 1:
                extra_num, current_degree_in_scale = divmod(
                    current_degree - 1, 7)
                diff = self[current_degree_in_scale].degree - self[0].degree
                result = self.get_note_from_degree(
                    1, pitch=rootpitch) + diff + extra_num * octave
        if octaves:
            octaves = int(octaves) * octave
            result += octaves
        return result

    def get(self,
            current_ind,
            default_duration=1 / 8,
            default_interval=1 / 8,
            default_volume=100,
            pitch_mode=0):
        if isinstance(current_ind, list):
            current_ind = ','.join([str(i) for i in current_ind])
        current = current_ind.replace(' ', '').split(',')
        notes_result = []
        intervals = []
        start_time = 0
        rootpitch = self[0].num
        for each in current:
            if each == '':
                continue
            if each.startswith('o'):
                rootpitch = int(each.split('o', 1)[1])
            else:
                has_settings = False
                duration = default_duration
                interval = default_interval
                volume = default_volume
                if '[' in each and ']' in each:
                    has_settings = True
                    each, current_settings = each.split('[', 1)
                    current_settings = current_settings[:-1].split(';')
                    current_settings_len = len(current_settings)
                    if current_settings_len == 1:
                        duration = _process_note(current_settings[0])
                    else:
                        if current_settings_len == 2:
                            duration, interval = current_settings
                        else:
                            duration, interval, volume = current_settings
                            volume = parse_num(volume)
                        duration = _process_note(duration)
                        interval = _process_note(
                            interval) if interval != '.' else duration
                current_notes = each.split(';')
                current_length = len(current_notes)
                for i, each_note in enumerate(current_notes):
                    has_same_time = True
                    if i == current_length - 1:
                        has_same_time = False
                    notes_result, intervals, start_time = self._read_single_note(
                        each_note,
                        rootpitch,
                        duration,
                        interval,
                        volume,
                        notes_result,
                        intervals,
                        start_time,
                        has_settings=has_settings,
                        has_same_time=has_same_time,
                        pitch_mode=pitch_mode)
        current_chord = chord(notes_result,
                              interval=intervals,
                              start_time=start_time)
        return current_chord

    def _read_single_note(self,
                          each,
                          rootpitch,
                          duration,
                          interval,
                          volume,
                          notes_result,
                          intervals,
                          start_time,
                          has_settings=False,
                          has_same_time=False,
                          pitch_mode=0):
        dotted_num = 0
        if each.endswith('.'):
            for k in range(len(each) - 1, -1, -1):
                if each[k] != '.':
                    each = each[:k + 1]
                    break
                else:
                    dotted_num += 1
        if each == 'r':
            current_interval = duration if has_settings else (
                dotted(interval, dotted_num) if interval != 0 else 1 / 4)
            if not notes_result:
                start_time += current_interval
            elif intervals:
                intervals[-1] += current_interval
        elif each == '-':
            current_interval = duration if has_settings else (
                dotted(interval, dotted_num) if interval != 0 else 1 / 4)
            if notes_result:
                notes_result[-1].duration += current_interval
            if intervals:
                intervals[-1] += current_interval
        else:
            current_note = self._parse_scale_text(each,
                                                  rootpitch,
                                                  pitch_mode=pitch_mode)
            current_note.duration = duration
            current_note.volume = volume
            if has_same_time:
                current_interval = 0
                if not has_settings:
                    current_note.duration = dotted(current_note.duration,
                                                      dotted_num)
            else:
                if has_settings:
                    current_interval = interval
                else:
                    current_interval = dotted(interval, dotted_num)
                    current_note.duration = dotted(current_note.duration,
                                                      dotted_num)
            notes_result.append(current_note)
            intervals.append(current_interval)
        return notes_result, intervals, start_time

    def index(self, current_note):
        if isinstance(current_note, note):
            current_note = current_note.name
        else:
            current_note = N(standardize_note(current_note)).name
        current_note = standardize_note(current_note)
        current_names = [standardize_note(i) for i in self.names()]
        if current_note not in current_names:
            raise ValueError(
                f'{current_note} is not in {self.get_scale_name(with_octave=False)}'
            )
        result = current_names.index(current_note)
        return result

    def get_scale_degree(self, current_note):
        return self.index(current_note) + 1

    def get_note_with_interval(self, current_note, interval, standard=False):
        current_scale_degree = self.get_scale_degree(current_note)
        current_num = current_note.num if isinstance(
            current_note, note) else get_note_num(current_note)
        if not current_num:
            current_num = 4
        if interval < 0:
            start_note = self.get_note_from_degree(current_scale_degree,
                                                   pitch=current_num)
            current_degree = current_scale_degree
            for i in range(abs(interval) - 1):
                current_degree -= 1
                if current_degree < 1:
                    current_degree = 7
                start_note -= self.interval[current_degree - 1]
            result = start_note
        else:
            current_scale_degree += (interval - 1)
            result = self.get_note_from_degree(current_scale_degree,
                                               pitch=current_num)
        if standard:
            result_scale_degree = self.get_scale_degree(result.name) - 1
            current_name = self.standard()[result_scale_degree]
            if current_name not in standard:
                current_name = standardize_note(current_name)
            result.name = current_name
        return result

    def get_standard_notation(self, current_note):
        current_scale_degree = self.get_scale_degree(current_note)
        result = self.standard()[current_scale_degree - 1]
        return result


class circle_of_fifths:
    '''
    This class represents the circle of fifths.
    '''
    outer = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    inner = [
        'Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'Ebm', 'Bbm', 'Fm', 'Cm', 'Gm',
        'Dm'
    ]

    def __init__(self):
        pass

    def __getitem__(self, ind):
        if isinstance(ind, int):
            if not (0 <= ind < 12):
                ind = ind % 12
            return self.outer[ind]
        elif isinstance(ind, tuple):
            ind = ind[0]
            if not (0 <= ind < 12):
                ind = ind % 12
            return self.inner[ind]

    def get(self, ind, mode=0):
        if mode == 0:
            return self[ind]
        else:
            return self[
                ind,
            ]

    def rotate(self, start, step=1, direction='cw', inner=False):
        if direction == 'ccw':
            step = -step
        if isinstance(start, note):
            startind = self.outer.index(start.name)
        elif isinstance(start, str):
            startind = self.outer.index(start)
        else:
            startind = start
        return self[startind + step] if not inner else self[
            startind + step,
        ]

    def rotate_get_scale(self,
                         start,
                         step=1,
                         direction='cw',
                         pitch=4,
                         inner=False):
        if not inner:
            return scale(note(self.rotate(start, step, direction), pitch),
                         'major')
        else:
            return scale(
                note(self.rotate(start, step, direction, True)[:-1], pitch),
                'minor')

    def get_scale(self, ind, pitch, inner=False):
        return scale(note(self[ind], pitch), 'major') if not inner else scale(
            note(self[
                ind,
            ][:-1], pitch), 'minor')

    def __repr__(self):
        return f'[circle of fifths]\nouter circle: {self.outer}\ninner circle: {self.inner}\ndirection: clockwise'


class circle_of_fourths(circle_of_fifths):
    '''
    This class represents the circle of fourths.
    '''
    outer = list(reversed(circle_of_fifths.outer))
    outer.insert(0, outer.pop())
    inner = list(reversed(circle_of_fifths.inner))
    inner.insert(0, inner.pop())

    def __init__(self):
        pass

    def __repr__(self):
        return f'[circle of fourths]\nouter circle: {self.outer}\ninner circle: {self.inner}\ndirection: clockwise'


class piece:
    '''
    This class represents a piece which contains multiple tracks.
    '''

    def __init__(self,
                 tracks,
                 instruments=None,
                 bpm=120,
                 start_times=None,
                 track_names=None,
                 channels=None,
                 name=None,
                 pan=None,
                 volume=None,
                 other_messages=[],
                 daw_channels=None):
        self.tracks = tracks
        if instruments is None:
            self.instruments = [1 for i in range(len(self.tracks))]
        else:
            self.instruments = [
                INSTRUMENTS[i] if isinstance(i, str) else i
                for i in instruments
            ]
        self.bpm = bpm
        self.start_times = start_times
        if self.start_times is None:
            self.start_times = [0 for i in range(self.track_number)]
        self.track_names = track_names
        self.channels = channels
        self.name = name
        self.pan = pan
        self.volume = volume
        if not self.pan:
            self.pan = [[] for i in range(self.track_number)]
        if not self.volume:
            self.volume = [[] for i in range(self.track_number)]
        self.other_messages = other_messages
        self.daw_channels = daw_channels
        self.ticks_per_beat = None

    @property
    def track_number(self):
        return len(self.tracks)

    def __repr__(self):
        return self.show()

    def show(self, limit=10):
        result = (
            f'[piece] {self.name if self.name is not None else ""}\n'
        ) + f'BPM: {round(self.bpm, 3)}\n' + '\n'.join([
            f'track {i} | channel: {self.channels[i] if self.channels is not None else None} | track name: {self.track_names[i] if self.track_names is not None and self.track_names[i] is not None else None} | instrument: {reverse_instruments[self.instruments[i]]} | start time: {self.start_times[i]} | content: {self.tracks[i].show(limit=limit)}'
            for i in range(len(self.tracks))
        ])
        return result

    def __eq__(self, other):
        return type(other) is piece and self.__dict__ == other.__dict__

    def __iter__(self):
        for i in self.tracks:
            yield i

    def __getitem__(self, i):
        return track(
            content=self.tracks[i],
            instrument=self.instruments[i],
            start_time=self.start_times[i],
            channel=self.channels[i] if self.channels is not None else None,
            track_name=self.track_names[i]
            if self.track_names is not None else None,
            pan=self.pan[i],
            volume=self.volume[i],
            bpm=self.bpm,
            name=self.name,
            daw_channel=self.daw_channels[i]
            if self.daw_channels is not None else None)

    def __delitem__(self, i):
        del self.tracks[i]
        del self.instruments[i]
        del self.start_times[i]
        if self.track_names is not None:
            del self.track_names[i]
        if self.channels is not None:
            del self.channels[i]
        del self.pan[i]
        del self.volume[i]
        if self.daw_channels is not None:
            del self.daw_channels[i]

    def __setitem__(self, i, new_track):
        self.tracks[i] = new_track.content
        self.instruments[i] = new_track.instrument
        self.start_times[i] = new_track.start_time
        if self.track_names is not None and new_track.track_name is not None:
            self.track_names[i] = new_track.track_name
        if self.channels is not None and new_track.channel is not None:
            self.channels[i] = new_track.channel
        if new_track.pan is not None:
            self.pan[i] = new_track.pan
        if new_track.volume is not None:
            self.volume[i] = new_track.volume
        if self.daw_channels is not None and new_track.daw_channel is not None:
            self.daw_channels[i] = new_track.daw_channel

    def __len__(self):
        return len(self.tracks)

    def get_instrument_names(self):
        return [reverse_instruments[i] for i in self.instruments]

    def mute(self, i=None):
        if not hasattr(self, 'muted_msg'):
            self.muted_msg = [each.get_volume() for each in self.tracks]
        if i is None:
            for k in range(len(self.tracks)):
                self.tracks[k].set_volume(0)
        else:
            self.tracks[i].set_volume(0)

    def unmute(self, i=None):
        if not hasattr(self, 'muted_msg'):
            return
        if i is None:
            for k in range(len(self.tracks)):
                self.tracks[k].set_volume(self.muted_msg[k])
        else:
            self.tracks[i].set_volume(self.muted_msg[i])

    def update_msg(self):
        self.other_messages = concat(
            [i.other_messages for i in self.tracks], start=[])

    def append(self, new_track):
        if not isinstance(new_track, track):
            raise ValueError('must be a track type to be appended')
        new_track = copy(new_track)
        self.tracks.append(new_track.content)
        self.instruments.append(new_track.instrument)
        self.start_times.append(new_track.start_time)
        if self.channels is not None:
            if new_track.channel is not None:
                self.channels.append(new_track.channel)
            else:
                self.channels.append(
                    max(self.channels) + 1 if self.channels else 0)
        if self.track_names is not None:
            if new_track.track_name is not None:
                self.track_names.append(new_track.track_name)
            else:
                self.track_names.append(
                    new_track.name if new_track.
                    name is not None else f'track {self.track_number}')
        self.pan.append(new_track.pan if new_track.pan is not None else [])
        self.volume.append(
            new_track.volume if new_track.volume is not None else [])
        if self.daw_channels is not None:
            if new_track.daw_channel is not None:
                self.daw_channels.append(new_track.daw_channel)
            else:
                self.daw_channels.append(0)
        self.tracks[-1].reset_track(len(self.tracks) - 1)
        self.update_msg()

    def insert(self, ind, new_track):
        if not isinstance(new_track, track):
            raise ValueError('must be a track type to be inserted')
        new_track = copy(new_track)
        self.tracks.insert(ind, new_track.content)
        self.instruments.insert(ind, new_track.instrument)
        self.start_times.insert(ind, new_track.start_time)
        if self.channels is not None:
            if new_track.channel is not None:
                self.channels.insert(ind, new_track.channel)
            else:
                self.channels.insert(
                    ind,
                    max(self.channels) + 1 if self.channels else 0)
        if self.track_names is not None:
            if new_track.track_name is not None:
                self.track_names.insert(ind, new_track.track_name)
            else:
                self.track_names.insert(
                    ind, new_track.name if new_track.name is not None else
                    f'track {self.track_number}')
        self.pan.insert(ind,
                        new_track.pan if new_track.pan is not None else [])
        self.volume.insert(
            ind, new_track.volume if new_track.volume is not None else [])
        if self.daw_channels is not None:
            if new_track.daw_channel is not None:
                self.daw_channels.insert(ind, new_track.daw_channel)
            else:
                self.daw_channels.insert(ind, 0)
        for k in range(ind, len(self.tracks)):
            self.tracks[k].reset_track(k)
        self.update_msg()

    def up(self, n=1, mode=0):
        temp = copy(self)
        for i in range(tetrack_number):
            if mode == 0 or (mode == 1 and not (temp.channels is not None
                                                and temp.channels[i] == 9)):
                tetracks[i] += n
        return temp

    def down(self, n=1, mode=0):
        temp = copy(self)
        for i in range(tetrack_number):
            if mode == 0 or (mode == 1 and not (temp.channels is not None
                                                and temp.channels[i] == 9)):
                tetracks[i] -= n
        return temp

    def __mul__(self, n):
        if isinstance(n, tuple):
            return self | n
        else:
            temp = copy(self)
            for i in range(n - 1):
                temp |= self
            return temp

    def __or__(self, n):
        if isinstance(n, tuple):
            n, start_time = n
            if isinstance(n, int):
                temp = copy(self)
                for k in range(n - 1):
                    temp |= (self, start_time)
                return temp
            elif isinstance(n, piece):
                return self.merge_track(n,
                                        mode='after',
                                        extra_interval=start_time)
        elif isinstance(n, piece):
            return self + n
        elif isinstance(n, (int, float)):
            return self.rest(n)

    def __and__(self, n):
        if isinstance(n, tuple):
            n, start_time = n
            if isinstance(n, int):
                temp = copy(self)
                for k in range(n - 1):
                    temp &= (self, (k + 1) * start_time)
                return temp
        elif isinstance(n, int):
            return self & (n, 0)
        else:
            start_time = 0
        return self.merge_track(n, mode='head', start_time=start_time)

    def __add__(self, n):
        if isinstance(n, (int, Interval)):
            return self.up(n)
        elif isinstance(n, piece):
            return self.merge_track(n, mode='after')
        elif isinstance(n, tuple):
            return self.up(*n)

    def __sub__(self, n):
        if isinstance(n, (int, Interval)):
            return self.down(n)
        elif isinstance(n, tuple):
            return self.down(*n)

    def __neg__(self):
        return self.down()

    def __pos__(self):
        return self.up()

    def __call__(self, ind):
        return self.tracks[ind]

    def merge_track(self,
                    n,
                    mode='after',
                    start_time=0,
                    ind_mode=1,
                    include_last_interval=False,
                    ignore_last_duration=False,
                    extra_interval=0):
        temp = copy(self)
        temp2 = copy(n)
        max_track_number = max(len(self), len(n))
        temp_length = len(temp)
        if temp_length < max_track_number:
            tepan.extend([[]
                             for i in range(max_track_number - temp_length)])
            tevolume.extend([[]
                                for i in range(max_track_number - temp_length)
                                ])
        if temp.channels is not None:
            free_channel_numbers = [
                i for i in range(16) if i not in temp.channels
            ]
            counter = 0
        if mode == 'after':
            if ignore_last_duration:
                bars_mode = 0
            else:
                bars_mode = 1 if not include_last_interval else 2
            start_time = tebars(mode=bars_mode) + extra_interval
        for i in range(len(temp2)):
            current_instrument_number = temp2.instruments[i]
            if current_instrument_number in teinstruments:
                if ind_mode == 0:
                    current_ind = teinstruments.index(
                        current_instrument_number)
                elif ind_mode == 1:
                    current_ind = i
                    if current_ind > len(tetracks) - 1:
                        teappend(
                            track(content=chord([]), start_time=start_time))
                current_track = temp2.tracks[i]
                for each in current_track.tempos:
                    each.start_time += start_time
                for each in current_track.pitch_bends:
                    each.start_time += start_time
                current_start_time = temp2.start_times[
                    i] + start_time - temp.start_times[current_ind]
                tetracks[current_ind] = tetracks[current_ind].add(
                    current_track,
                    start=current_start_time,
                    mode='head',
                    adjust_msg=False)
                if current_start_time < 0:
                    temp.start_times[current_ind] += current_start_time
                for each in temp2.pan[i]:
                    each.start_time += start_time
                for each in temp2.volume[i]:
                    each.start_time += start_time
                tepan[current_ind].extend(temp2.pan[i])
                tevolume[current_ind].extend(temp2.volume[i])
            else:
                teinstruments.append(current_instrument_number)
                current_start_time = temp2.start_times[i]
                current_start_time += start_time
                current_track = temp2.tracks[i]
                for each in current_track.tempos:
                    each.start_time += start_time
                for each in current_track.pitch_bends:
                    each.start_time += start_time
                tetracks.append(current_track)
                temp.start_times.append(current_start_time)
                for each in temp2.pan[i]:
                    each.start_time += start_time
                for each in temp2.volume[i]:
                    each.start_time += start_time
                tepan.append(temp2.pan[i])
                tevolume.append(temp2.volume[i])
                if temp.channels is not None:
                    if temp2.channels is not None:
                        current_channel_number = temp2.channels[i]
                        if current_channel_number in temp.channels:
                            current_channel_number = free_channel_numbers[
                                counter]
                            counter += 1
                        else:
                            if current_channel_number in free_channel_numbers:
                                del free_channel_numbers[
                                    free_channel_numbers.index(
                                        current_channel_number)]
                    else:
                        current_channel_number = free_channel_numbers[counter]
                        counter += 1
                    temp.channels.append(current_channel_number)
                if tetrack_names is not None:
                    tetrack_names.append(temp2.track_names[i])
        return temp

    def repeat(self,
               n,
               start_time=0,
               include_last_interval=False,
               ignore_last_duration=False,
               ind_mode=1,
               mode='after'):
        temp = copy(self)
        if mode == 'after':
            for k in range(n - 1):
                temp = temerge_track(
                    self,
                    mode=mode,
                    extra_interval=start_time,
                    include_last_interval=include_last_interval,
                    ignore_last_duration=ignore_last_duration,
                    ind_mode=ind_mode)
        elif mode == 'head':
            for k in range(n - 1):
                temp = temerge_track(
                    self,
                    mode=mode,
                    start_time=(k + 1) * start_time,
                    include_last_interval=include_last_interval,
                    ignore_last_duration=ignore_last_duration,
                    ind_mode=ind_mode)
        return temp

    def align(self, extra=0):
        temp = copy(self)
        track_lens = [
            temp.start_times[i] + tetracks[i].bars()
            for i in range(len(tetracks))
        ]
        for each in tetracks:
            each.interval[-1] = each.notes[-1].duration
        max_len = max(track_lens) + extra
        for i in range(len(tetracks)):
            extra_lens = max_len - track_lens[i]
            if extra_lens > 0:
                tetracks[i] |= extra_lens
        return temp

    def rest(self, n):
        return self.align(n)

    def add_pitch_bend(self,
                       value,
                       start_time,
                       channel='all',
                       track=0,
                       mode='cents'):
        if channel == 'all':
            for i in range(len(self.tracks)):
                current_channel = self.channels[
                    i] if self.channels is not None else i
                current_pitch_bend = pitch_bend(value=value,
                                                start_time=start_time,
                                                mode=mode,
                                                channel=current_channel,
                                                track=track)
                self.tracks[i].pitch_bends.append(current_pitch_bend)
        else:
            current_channel = self.channels[
                channel] if self.channels is not None else channel
            self.tracks[channel].pitch_bends.append(
                pitch_bend(value=value,
                           start_time=start_time,
                           mode=mode,
                           channel=current_channel,
                           track=track))

    def add_tempo_change(self, bpm, start_time, track_ind=0):
        self.tracks[track_ind].tempos.append(
            tempo(bpm=bpm, start_time=start_time))

    def clear_pitch_bend(self, ind='all', value='all', cond=None):
        if ind == 'all':
            for each in self.tracks:
                each.clear_pitch_bend(value, cond)
        else:
            self.tracks[ind].clear_pitch_bend(value, cond)

    def clear_tempo(self, ind='all', cond=None):
        if ind == 'all':
            for each in self.tracks:
                each.clear_tempo(cond)
        else:
            self.tracks[ind].clear_tempo(cond)

    def normalize_tempo(self, bpm=None, reset_bpm=False):
        if bpm is None:
            bpm = self.bpm
        if bpm == self.bpm and all(i.bpm == bpm for each in self.tracks
                                   for i in each.tempos):
            self.clear_tempo()
            return
        temp = copy(self)
        original_bpm = None
        if bpm != self.bpm and all(not each.tempos for each in self.tracks):
            original_bpm = self.bpm
        _piece_process_normalize_tempo(temp,
                                       bpm,
                                       min(temp.start_times),
                                       original_bpm=original_bpm)
        self.start_times = temp.start_times
        self.other_messages = temp.other_messages
        self.pan = tepan
        self.volume = tevolume
        for i in range(len(self.tracks)):
            self.tracks[i] = tetracks[i]
        if reset_bpm:
            self.bpm = bpm

    def get_tempo_changes(self, ind=None):
        tempo_changes = [i.tempos for i in self.tracks
                         ] if ind is None else self.tracks[ind].tempos
        return tempo_changes

    def get_pitch_bend(self, ind=None):
        pitch_bend_changes = [
            i.pitch_bends for i in self.tracks
        ] if ind is None else self.tracks[ind].pitch_bends
        return pitch_bend_changes

    def get_msg(self, types, ind=None):
        if ind is None:
            return [i for i in self.other_messages if i.type == types]
        else:
            return [
                i for i in self.tracks[ind].other_messages if i.type == types
            ]

    def add_pan(self,
                value,
                ind,
                start_time=0,
                mode='percentage',
                channel=None,
                track=None):
        self.pan[ind].append(pan(value, start_time, mode, channel, track))

    def add_volume(self,
                   value,
                   ind,
                   start_time=0,
                   mode='percentage',
                   channel=None,
                   track=None):
        self.volume[ind].append(volume(value, start_time, mode, channel,
                                       track))

    def clear_pan(self, ind='all'):
        if ind == 'all':
            for each in self.pan:
                each.clear()
        else:
            self.pan[ind].clear()

    def clear_volume(self, ind='all'):
        if ind == 'all':
            for each in self.volume:
                each.clear()
        else:
            self.volume[ind].clear()

    def reassign_channels(self, start=0):
        new_channels_numbers = [start + i for i in range(len(self.tracks))]
        self.channels = new_channels_numbers

    def delete_track(self, current_ind, only_clear_msg=False):
        if not only_clear_msg:
            del self[current_ind]
        self.other_messages = [
            i for i in self.other_messages if i.track != current_ind
        ]
        for each in self.tracks:
            each.delete_track(current_ind)
        self.pan = [[i for i in each if i.track != current_ind]
                    for each in self.pan]
        for each in self.pan:
            for i in each:
                if i.track is not None and i.track > current_ind:
                    i.track -= 1
        self.volume = [[i for i in each if i.track != current_ind]
                       for each in self.volume]
        for each in self.volume:
            for i in each:
                if i.track is not None and i.track > current_ind:
                    i.track -= 1

    def delete_channel(self, current_ind):
        for each in self.tracks:
            each.delete_channel(current_ind)
        self.other_messages = [
            i for i in self.other_messages
            if not (hasattr(i, 'channel') and i.channel == current_ind)
        ]
        self.pan = [[i for i in each if i.channel != current_ind]
                    for each in self.pan]
        self.volume = [[i for i in each if i.channel != current_ind]
                       for each in self.volume]

    def get_off_drums(self):
        if self.channels is not None:
            while 9 in self.channels:
                current_ind = self.channels.index(9)
                self.delete_track(current_ind)
        self.delete_channel(9)

    def merge(self,
              add_labels=True,
              add_pan_volume=False,
              get_off_drums=False,
              track_names_add_channel=False):
        temp = copy(self)
        if add_labels:
            teadd_track_labels()
        if get_off_drums:
            teget_off_drums()
        if track_names_add_channel and temp.channels is not None:
            for i, each in enumerate(tetracks):
                for j in each.other_messages:
                    if j.type == 'track_name':
                        j.channel = temp.channels[i]
        all_tracks = tetracks
        length = len(all_tracks)
        track_map_dict = {}
        if temp.channels is not None:
            merge_channels = list(dict.fromkeys(temp.channels))
            merge_length = len(merge_channels)
            if merge_length < length:
                for i in range(merge_length, length):
                    track_map_dict[i] = temp.channels.index(temp.channels[i])
        start_time_ls = temp.start_times
        sort_tracks_inds = [[i, start_time_ls[i]] for i in range(length)]
        sort_tracks_inds.sort(key=lambda s: s[1])
        first_track_start_time = sort_tracks_inds[0][1]
        first_track_ind = sort_tracks_inds[0][0]
        first_track = all_tracks[first_track_ind]
        for i in sort_tracks_inds[1:]:
            current_track = all_tracks[i[0]]
            current_start_time = i[1]
            current_shift = current_start_time - first_track_start_time
            first_track = first_track.add(current_track,
                                          start=current_shift,
                                          mode='head',
                                          adjust_msg=False)
        first_track.other_messages = temp.other_messages
        if add_pan_volume:
            whole_pan = concat(tepan)
            whole_volume = concat(tevolume)
            pan_msg = [
                event('control_change',
                      channel=i.channel,
                      track=i.track,
                      start_time=i.start_time,
                      control=10,
                      value=i.value) for i in whole_pan
            ]
            volume_msg = [
                event('control_change',
                      channel=i.channel,
                      track=i.track,
                      start_time=i.start_time,
                      control=7,
                      value=i.value) for i in whole_volume
            ]
            first_track.other_messages += pan_msg
            first_track.other_messages += volume_msg
        first_track_start_time += first_track.start_time
        first_track.start_time = 0
        if track_map_dict:
            if add_labels:
                for i in first_track.notes:
                    if i.track_num in track_map_dict:
                        i.track_num = track_map_dict[i.track_num]
            for i in first_track.tempos:
                if i.track in track_map_dict:
                    current_track = track_map_dict[i.track]
                    i.track = current_track
                    if add_labels:
                        i.track_num = current_track
            for i in first_track.pitch_bends:
                if i.track in track_map_dict:
                    current_track = track_map_dict[i.track]
                    i.track = current_track
                    if add_labels:
                        i.track_num = current_track
            for i in first_track.other_messages:
                if i.track in track_map_dict:
                    i.track = track_map_dict[i.track]
        return first_track, tebpm, first_track_start_time

    def add_track_labels(self):
        all_tracks = self.tracks
        length = len(all_tracks)
        for k in range(length):
            current_track = all_tracks[k]
            for each in current_track.notes:
                each.track_num = k
            for each in current_track.tempos:
                each.track_num = k
            for each in current_track.pitch_bends:
                each.track_num = k

    def reconstruct(self,
                    track,
                    start_time=0,
                    offset=0,
                    correct=False,
                    include_empty_track=False,
                    get_channels=True):
        first_track, first_track_start_time = track, start_time
        length = len(self.tracks)
        start_times_inds = [[
            i for i in range(len(first_track))
            if first_track.notes[i].track_num == k
        ] for k in range(length)]
        if not include_empty_track:
            available_tracks_inds = [
                k for k in range(length) if start_times_inds[k]
            ]
        else:
            available_tracks_inds = [k for k in range(length)]
        available_tracks_messages = [
            self.tracks[i].other_messages for i in available_tracks_inds
        ]
        if not include_empty_track:
            start_times_inds = [i[0] for i in start_times_inds if i]
        else:
            empty_track_inds = [
                i for i in range(length) if not start_times_inds[i]
            ]
            start_times_inds = [i[0] if i else -1 for i in start_times_inds]
        new_start_times = [
            first_track_start_time +
            first_track[:k].bars(mode=0) if k != -1 else 0
            for k in start_times_inds
        ]
        if correct:
            new_start_times_offset = [
                self.start_times[i] - offset for i in available_tracks_inds
            ]
            start_time_offset = min([
                new_start_times_offset[k] - new_start_times[k]
                for k in range(len(new_start_times))
            ],
                                    key=lambda s: abs(s))
            new_start_times = [i + start_time_offset for i in new_start_times]
            if include_empty_track:
                new_start_times = [
                    new_start_times[i] if i not in empty_track_inds else 0
                    for i in range(length)
                ]
        new_start_times = [i if i >= 0 else 0 for i in new_start_times]
        new_track_notes = [[] for k in range(length)]
        new_track_inds = [[] for k in range(length)]
        new_track_tempos = [[] for k in range(length)]
        new_track_pitch_bends = [[] for k in range(length)]
        whole_length = len(first_track)
        for j in range(whole_length):
            current_note = first_track.notes[j]
            new_track_notes[current_note.track_num].append(current_note)
            new_track_inds[current_note.track_num].append(j)
        for i in first_track.tempos:
            new_track_tempos[i.track_num].append(i)
        for i in first_track.pitch_bends:
            new_track_pitch_bends[i.track_num].append(i)
        whole_interval = first_track.interval
        new_track_intervals = [[
            sum(whole_interval[inds[i]:inds[i + 1]])
            for i in range(len(inds) - 1)
        ] for inds in new_track_inds]
        for i in available_tracks_inds:
            if new_track_inds[i]:
                new_track_intervals[i].append(
                    sum(whole_interval[new_track_inds[i][-1]:]))
        new_tracks = []
        for i in range(len(available_tracks_inds)):
            current_track_ind = available_tracks_inds[i]
            current_track = chord(
                new_track_notes[current_track_ind],
                interval=new_track_intervals[current_track_ind],
                tempos=new_track_tempos[current_track_ind],
                pitch_bends=new_track_pitch_bends[current_track_ind],
                other_messages=available_tracks_messages[i])
            current_track.track_ind = current_track_ind
            current_track.interval = [
                int(i) if isinstance(i, float) and i.is_integer() else i
                for i in current_track.interval
            ]
            new_tracks.append(current_track)
        self.tracks = new_tracks
        self.start_times = [
            int(i) if isinstance(i, float) and i.is_integer() else i
            for i in new_start_times
        ]
        self.instruments = [self.instruments[k] for k in available_tracks_inds]
        if self.track_names is not None:
            self.track_names = [
                self.track_names[k] for k in available_tracks_inds
            ]
        if self.channels is not None:
            self.channels = [self.channels[k] for k in available_tracks_inds]
        else:
            if get_channels:
                from collections import Counter
                current_channels = [
                    Counter([i.channel for i in each
                             if i.channel is not None]).most_common(1)
                    for each in self.tracks
                ]
                if all(i for i in current_channels):
                    self.channels = [i[0][0] for i in current_channels]
        self.pan = [self.pan[k] for k in available_tracks_inds]
        self.volume = [self.volume[k] for k in available_tracks_inds]
        self.reset_track(list(range(self.track_number)))

    def eval_time(self,
                  bpm=None,
                  ind1=None,
                  ind2=None,
                  mode='seconds',
                  normalize_tempo=True,
                  audio_mode=0):
        merged_result, temp_bpm, start_time = self.merge()
        if bpm is not None:
            temp_bpm = bpm
        if normalize_tempo:
            merged_result.normalize_tempo(temp_bpm)
        return merged_result.eval_time(temp_bpm,
                                       ind1,
                                       ind2,
                                       mode,
                                       start_time=start_time,
                                       audio_mode=audio_mode)

    def cut(self,
            ind1=0,
            ind2=None,
            correct=False,
            cut_extra_duration=False,
            cut_extra_interval=False,
            round_duration=False,
            round_cut_interval=False):
        merged_result, temp_bpm, start_time = self.merge()
        if ind1 < 0:
            ind1 = 0
        result = merged_result.cut(ind1,
                                   ind2,
                                   start_time,
                                   cut_extra_duration=cut_extra_duration,
                                   cut_extra_interval=cut_extra_interval,
                                   round_duration=round_duration,
                                   round_cut_interval=round_cut_interval)
        offset = ind1
        temp = copy(self)
        start_time -= ind1
        if start_time < 0:
            start_time = 0
        tereconstruct(track=result,
                         start_time=start_time,
                         offset=offset,
                         correct=correct)
        if ind2 is None:
            ind2 = tebars()
        for each in tepan:
            for i in each:
                i.start_time -= ind1
                if i.start_time < 0:
                    i.start_time = 0
        for each in tevolume:
            for i in each:
                i.start_time -= ind1
                if i.start_time < 0:
                    i.start_time = 0
        tepan = [[i for i in each if i.start_time < ind2]
                    for each in tepan]
        tevolume = [[i for i in each if i.start_time < ind2]
                       for each in tevolume]
        tempo_changes = concat(teget_tempo_changes(), start=[])
        temp.clear_tempo()
        track_inds = [each.track_ind for each in tetracks]
        temp.other_messages = [
            i for i in temp.other_messages if ind1 <= i.start_time < ind2
        ]
        temp.other_messages = [
            i for i in temp.other_messages if i.track in track_inds
        ]
        for each in temp.other_messages:
            each.track = track_inds.index(each.track)
        tetracks[0].tempos.extend(tempo_changes)
        tereset_track([*range(len(tetracks))])
        return temp

    def cut_time(self,
                 time1=0,
                 time2=None,
                 bpm=None,
                 start_time=0,
                 cut_extra_duration=False,
                 cut_extra_interval=False,
                 round_duration=False,
                 round_cut_interval=False):
        temp = copy(self)
        tenormalize_tempo()
        if bpm is not None:
            temp_bpm = bpm
        else:
            temp_bpm = tebpm
        bar_left = time1 / ((60 / temp_bpm) * 4)
        bar_right = time2 / (
            (60 / temp_bpm) * 4) if time2 is not None else tebars()
        result = tecut(bar_left,
                          bar_right,
                          cut_extra_duration=cut_extra_duration,
                          cut_extra_interval=cut_extra_interval,
                          round_duration=round_duration,
                          round_cut_interval=round_cut_interval)
        return result

    def get_bar(self, n):
        start_time = min(self.start_times)
        return self.cut(n + start_time, n + start_time)

    def firstnbars(self, n):
        start_time = min(self.start_times)
        return self.cut(start_time, n + start_time)

    def bars(self, mode=1, audio_mode=0, bpm=None):
        return max([
            self.tracks[i].bars(start_time=self.start_times[i],
                                mode=mode,
                                audio_mode=audio_mode,
                                bpm=bpm) for i in range(len(self.tracks))
        ])

    def total(self):
        return sum([len(i) for i in self.tracks])

    def count(self, note1, mode='name'):
        return sum([each.count(note1, mode) for each in self.tracks])

    def most_appear(self, choices=None, mode='name', as_standard=False):
        return self.quick_merge().most_appear(choices, mode, as_standard)

    def quick_merge(self):
        result = chord([])
        for each in self.tracks:
            result.notes += each.notes
            result.interval += each.interval
            result.other_messages += each.other_messages
            result.tempos += each.tempos
            result.pitch_bends += each.pitch_bends
        return result

    def standard_notation(self):
        temp = copy(self)
        tetracks = [each.standard_notation() for each in tetracks]
        return temp

    def count_appear(self, choices=None, as_standard=True, sort=False):
        return self.quick_merge().count_appear(choices, as_standard, sort)

    def apply_start_time_to_changes(self,
                                    start_time,
                                    msg=False,
                                    pan_volume=False):
        if isinstance(start_time, (int, float)):
            start_time = [start_time for i in range(len(self.tracks))]
        tracks = self.tracks
        for i in range(len(tracks)):
            current_start_time = start_time[i]
            current_track = tracks[i]
            for each in current_track.tempos:
                each.start_time += current_start_time
                if each.start_time < 0:
                    each.start_time = 0
            for each in current_track.pitch_bends:
                each.start_time += current_start_time
                if each.start_time < 0:
                    each.start_time = 0
            if msg:
                for each in current_track.other_messages:
                    each.start_time += current_start_time
                    if each.start_time < 0:
                        each.start_time = 0
            if pan_volume:
                current_pan = self.pan[i]
                current_volume = self.volume[i]
                for each in current_pan:
                    each.start_time += current_start_time
                    if each.start_time < 0:
                        each.start_time = 0
                for each in current_volume:
                    each.start_time += current_start_time
                    if each.start_time < 0:
                        each.start_time = 0

    def reverse(self):
        temp = copy(self)
        tetracks = [
            tetracks[i].reverse(start_time=temp.start_times[i])
            for i in range(len(tetracks))
        ]
        length = tebars()
        start_times = temp.start_times
        tracks = tetracks
        track_num = len(tetracks)
        first_start_time = min(self.start_times)
        temp.start_times = [
            length - (start_times[i] + tracks[i].bars() - first_start_time)
            for i in range(track_num)
        ]

        teapply_start_time_to_changes(temp.start_times)
        for each in tepan:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        for each in tevolume:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        return temp

    def reverse_chord(self):
        temp = copy(self)
        tetracks = [
            tetracks[i].reverse_chord(start_time=temp.start_times[i])
            for i in range(len(tetracks))
        ]
        length = tebars()
        start_times = temp.start_times
        tracks = tetracks
        track_num = len(tetracks)
        first_start_time = min(self.start_times)
        temp.start_times = [
            length - (start_times[i] + tracks[i].bars() - first_start_time)
            for i in range(track_num)
        ]
        teapply_start_time_to_changes(temp.start_times)
        for each in tepan:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        for each in tevolume:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        return temp

    def __invert__(self):
        return self.reverse()

    def clear_program_change(self, apply_tracks=True):
        if apply_tracks:
            for each in self.tracks:
                each.clear_program_change()
        self.other_messages = [
            i for i in self.other_messages if i.type != 'program_change'
        ]

    def clear_other_messages(self, types=None, apply_tracks=True, ind=None):
        if ind is None:
            if types is None:
                self.other_messages.clear()
            else:
                self.other_messages = [
                    i for i in self.other_messages if i.type != types
                ]
            if apply_tracks:
                for each in self.tracks:
                    each.clear_other_messages(types)
        else:
            if types is None:
                self.other_messages = [
                    i for i in self.other_messages if i.track != ind
                ]
            else:
                self.other_messages = [
                    i for i in self.other_messages
                    if not (i.track == ind and i.type == types)
                ]
            if apply_tracks:
                self.tracks[ind].clear_other_messages(types)

    def change_instruments(self, instruments, ind=None):
        if ind is None:
            if all(isinstance(i, int) for i in instruments):
                self.instruments = copy(instruments)
            elif all(isinstance(i, str) for i in instruments):
                self.instruments = [
                    INSTRUMENTS[i] for i in instruments
                ]
            elif any(
                    isinstance(i, list) and all(isinstance(j, int) for j in i)
                    for i in instruments):
                self.instruments = copy(instruments)
        else:
            if isinstance(instruments, int):
                self.instruments[ind] = instruments
            elif isinstance(instruments, str):
                self.instruments[ind] = INSTRUMENTS[instruments]
            elif isinstance(instruments, list) and all(
                    isinstance(j, int) for j in instruments):
                self.instruments[ind] = copy(instruments)

    def move(self, time=0, ind='all'):
        temp = copy(self)
        if ind == 'all':
            temp.start_times = [i + time for i in temp.start_times]
            temp.start_times = [0 if i < 0 else i for i in temp.start_times]
            teapply_start_time_to_changes(
                [time for i in range(len(temp.start_times))],
                msg=True,
                pan_volume=True)
        else:
            temp.start_times[ind] += time
            tetracks[ind].apply_start_time_to_changes(time, msg=True)
            for each in tepan[ind]:
                each.start_time += time
                if each.start_time < 0:
                    each.start_time = 0
            for each in tevolume[ind]:
                each.start_time += time
                if each.start_time < 0:
                    each.start_time = 0
        return temp

    def modulation(self, old_scale, new_scale, mode=1, inds='all'):
        temp = copy(self)
        if inds == 'all':
            inds = list(range(len(temp)))
        for i in inds:
            if not (mode == 1 and temp.channels is not None
                    and temp.channels[i] == 9):
                tetracks[i] = tetracks[i].modulation(
                    old_scale, new_scale)
        return temp

    def apply(self, func, inds='all', mode=0, new=True):
        if new:
            temp = copy(self)
            if isinstance(inds, int):
                inds = [inds]
            elif inds == 'all':
                inds = list(range(len(temp)))
            if mode == 0:
                for i in inds:
                    tetracks[i] = func(tetracks[i])
            elif mode == 1:
                for i in inds:
                    func(tetracks[i])
            return temp
        else:
            if isinstance(inds, int):
                inds = [inds]
            elif inds == 'all':
                inds = list(range(len(self)))
            if mode == 0:
                for i in inds:
                    self.tracks[i] = func(self.tracks[i])
            elif mode == 1:
                for i in inds:
                    func(self.tracks[i])

    def reset_channel(self,
                      channels,
                      reset_msg=True,
                      reset_pitch_bend=True,
                      reset_pan_volume=True,
                      reset_note=True):
        if isinstance(channels, (int, float)):
            channels = [channels for i in range(len(self.tracks))]
        self.channels = channels
        for i in range(len(self.tracks)):
            current_channel = channels[i]
            current_track = self.tracks[i]
            if reset_msg:
                current_other_messages = current_track.other_messages
                for each in current_other_messages:
                    if hasattr(each, 'channel'):
                        each.channel = current_channel
            if reset_pitch_bend:
                for each in current_track.pitch_bends:
                    each.channel = current_channel
            if reset_note:
                for each in current_track.notes:
                    each.channel = current_channel
            if reset_pan_volume:
                current_pan = self.pan[i]
                current_volume = self.volume[i]
                for each in current_pan:
                    each.channel = current_channel
                for each in current_volume:
                    each.channel = current_channel

    def reset_track(self,
                    tracks,
                    reset_msg=True,
                    reset_pitch_bend=True,
                    reset_pan_volume=True):
        if isinstance(tracks, (int, float)):
            tracks = [tracks for i in range(len(self.tracks))]
        for i in range(len(self.tracks)):
            current_track_num = tracks[i]
            current_track = self.tracks[i]
            if reset_msg:
                current_other_messages = current_track.other_messages
                for each in current_other_messages:
                    each.track = current_track_num
            if reset_pitch_bend:
                for each in current_track.pitch_bends:
                    each.track = current_track_num
            if reset_pan_volume:
                current_pan = self.pan[i]
                current_volume = self.volume[i]
                for each in current_pan:
                    each.track = current_track_num
                for each in current_volume:
                    each.track = current_track_num


class tempo:
    '''
    This is a class to change tempo for the notes after it when it is read,
    it can be inserted into a chord, and if the chord is in a piece,
    then it also works for the piece.
    '''

    def __init__(self, bpm, start_time=0, channel=None, track=None):
        self.bpm = bpm
        self.start_time = start_time
        self.channel = channel
        self.track = track

    def __repr__(self):
        attributes = ['bpm', 'start_time', 'channel', 'track']
        result = f'tempo({", ".join([f"{i}={j}" for i, j in self.__dict__.items() if i in attributes])})'
        return result

    def set_volume(self, vol):
        vol = int(vol)
        self.volume = vol

    def set_channel(self, channel):
        self.channel = channel

    def with_channel(self, channel):
        temp = copy(self)
        temp.channel = channel
        return temp


class pitch_bend:
    '''
    This class represents a pitch bend event in midi.
    '''

    def __init__(self,
                 value,
                 start_time=0,
                 mode='cents',
                 channel=None,
                 track=None):
        '''
        general midi pitch bend values could be taken from -8192 to 8192,
        and the default pitch bend range is -2 semitones to 2 semitones,
        which is -200 cents to 200 cents, which means 1 cent equals to
        8192/200 = 40.96, about 41 values, and 1 semitone equals to
        8192/2 = 4096 values.
        if mode == 'cents', convert value as cents to midi pitch bend values,
        if mode == 'semitones', convert value as semitones to midi pitch bend values,
        if mode == other values, use value as midi pitch bend values
        '''
        self.value = value
        self.start_time = start_time
        self.channel = channel
        self.track = track
        self.mode = mode
        if self.mode == 'cents':
            self.value = int(self.value * 40.96)
        elif self.mode == 'semitones':
            self.value = int(self.value * 4096)

    def __repr__(self):
        attributes = ['value', 'start_time', 'channel', 'track']
        current_cents = self.value / 40.96
        if isinstance(current_cents, float) and current_cents.is_integer():
            current_cents = int(current_cents)
        current_dict = {
            i: j
            for i, j in self.__dict__.items() if i in attributes
        }
        current_dict['cents'] = current_cents
        result = f'pitch_bend({", ".join([f"{i}={j}" for i, j in current_dict.items()])})'
        return result

    def set_volume(self, vol):
        vol = int(vol)
        self.volume = vol

    def set_channel(self, channel):
        self.channel = channel

    def with_channel(self, channel):
        temp = copy(self)
        temp.channel = channel
        return temp


class track:
    '''
    This class represents a single track, which content is a chord instance, and has other attributes that define a track.
    '''

    def __init__(self,
                 content,
                 instrument=1,
                 start_time=0,
                 channel=None,
                 track_name=None,
                 pan=None,
                 volume=None,
                 bpm=120,
                 name=None,
                 daw_channel=None):
        self.content = content
        self.instrument = INSTRUMENTS[instrument] if isinstance(
            instrument, str) else instrument
        self.bpm = bpm
        self.start_time = start_time
        self.track_name = track_name
        self.channel = channel
        self.name = name
        self.pan = pan
        self.volume = volume
        self.daw_channel = daw_channel
        if self.pan:
            if not isinstance(self.pan, list):
                self.pan = [self.pan]
        else:
            self.pan = []
        if self.volume:
            if not isinstance(self.volume, list):
                self.volume = [self.volume]
        else:
            self.volume = []

    def __repr__(self):
        return self.show()

    def show(self, limit=10):
        return (f'[track] {self.name if self.name is not None else ""}\n') + (
            f'BPM: {round(self.bpm, 3)}\n' if self.bpm is not None else ""
        ) + f'channel: {self.channel} | track name: {self.track_name} | instrument: {reverse_instruments[self.instrument]} | start time: {self.start_time} | content: {self.content.show(limit=limit)}'

    def get_instrument_name(self):
        return reverse_instruments[self.instrument]

    def add_pan(self,
                value,
                start_time=0,
                mode='percentage',
                channel=None,
                track=None):
        self.pan.append(pan(value, start_time, mode, channel, track))

    def add_volume(self,
                   value,
                   start_time=0,
                   mode='percentage',
                   channel=None,
                   track=None):
        self.volume.append(volume(value, start_time, mode, channel, track))

    def get_interval(self):
        return self.content.interval

    def get_duration(self):
        return self.content.get_duration()

    def get_volume(self):
        return self.content.get_volume()

    def reverse(self, *args, **kwargs):
        temp = copy(self)
        tecontent = tecontent.reverse(*args, **kwargs)
        length = tebars()
        for each in tepan:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        for each in tevolume:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        return temp

    def reverse_chord(self, *args, **kwargs):
        temp = copy(self)
        tecontent = tecontent.reverse_chord(*args, **kwargs)
        length = tebars()
        for each in tepan:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        for each in tevolume:
            for i in each:
                i.start_time = length - i.start_time
                if i.start_time < 0:
                    i.start_time = 0
        return temp

    def __invert__(self):
        return self.reverse()

    def up(self, n=1):
        temp = copy(self)
        tecontent += n
        return temp

    def down(self, n=1):
        temp = copy(self)
        tecontent -= n
        return temp

    def __mul__(self, n):
        temp = copy(self)
        tecontent *= n
        return temp

    def __pos__(self):
        return self.up()

    def __neg__(self):
        return self.down()

    def __add__(self, i):
        if isinstance(i, (int, Interval)):
            return self.up(i)
        else:
            temp = copy(self)
            tecontent += i
            return temp

    def __sub__(self, i):
        if isinstance(i, (int, Interval)):
            return self.down(i)

    def __or__(self, i):
        temp = copy(self)
        tecontent |= i
        return temp

    def __and__(self, i):
        temp = copy(self)
        tecontent &= i
        return temp

    def set(self, duration=None, interval=None, volume=None, ind='all'):
        temp = copy(self)
        tecontent = tecontent.set(duration, interval, volume, ind)
        return temp

    def __getitem__(self, i):
        return self.content[i]

    def __setitem__(self, i, value):
        self.content[i] = value

    def __delitem__(self, i):
        del self.content[i]

    def __len__(self):
        return len(self.content)

    def delete_track(self, current_ind):
        self.content.delete_track(current_ind)
        if self.pan:
            self.pan = [i for i in self.pan if i.track != current_ind]
            for i in self.pan:
                if i.track is not None and i.track > current_ind:
                    i.track -= 1
        if self.volume:
            self.volume = [i for i in self.volume if i.track != current_ind]
            for i in self.volume:
                if i.track is not None and i.track > current_ind:
                    i.track -= 1

    def delete_channel(self, current_ind):
        self.content.delete_channel(current_ind)
        self.pan = [i for i in self.pan if i.channel != current_ind]
        self.volume = [i for i in self.volume if i.channel != current_ind]


class pan:
    '''
    This is a class to set the pan position for a midi channel,
    it only works in piece class or track class, and must be set as one of the elements
    of the pan list of a piece.
    '''

    def __init__(self,
                 value,
                 start_time=0,
                 mode='percentage',
                 channel=None,
                 track=None):
        # when mode == 'percentage', percentage ranges from 0% to 100%,
        # value takes an integer or float number from 0 to 100 (inclusive),
        # 0% means pan left most, 100% means pan right most, 50% means pan middle
        # when mode == 'value', value takes an integer from 0 to 127 (inclusive),
        # and corresponds to the pan positions same as percentage mode
        self.mode = mode
        if self.mode == 'percentage':
            self.value = int(127 * value / 100)
            self.value_percentage = value
        elif self.mode == 'value':
            self.value = value
            self.value_percentage = (self.value / 127) * 100
        self.start_time = start_time
        self.channel = channel
        self.track = track

    def __repr__(self):
        current_dict = {
            i: j
            for i, j in self.__dict__.items()
            if i in ['value_percentage', 'start_time', 'channel', 'track']
        }
        current_dict['percentage'] = round(
            current_dict.pop('value_percentage'), 3)
        attributes = ['percentage', 'start_time', 'channel', 'track']
        result = f'pan({", ".join([f"{i}={current_dict[i]}" for i in attributes])})'
        return result

    def get_pan_value(self):
        return -((50 - self.value_percentage) /
                 50) if self.value_percentage <= 50 else (
                     self.value_percentage - 50) / 50


class volume:
    '''
    This is a class to set the volume for a midi channel,
    it only works in piece class or track class, and must be set as one of the elements
    of the volume list of a piece.
    '''

    def __init__(self,
                 value,
                 start_time=0,
                 mode='percentage',
                 channel=None,
                 track=None):
        # when mode == 'percentage', percentage ranges from 0% to 100%,
        # value takes an integer or float number from 0 to 100 (inclusive),
        # when mode == 'value', value takes an integer from 0 to 127 (inclusive)
        self.mode = mode
        if self.mode == 'percentage':
            self.value = int(127 * value / 100)
            self.value_percentage = value
        elif self.mode == 'value':
            self.value = value
            self.value_percentage = (self.value / 127) * 100
        self.start_time = start_time
        self.channel = channel
        self.track = track

    def __repr__(self):
        current_dict = {
            i: j
            for i, j in self.__dict__.items()
            if i in ['value_percentage', 'start_time', 'channel', 'track']
        }
        current_dict['percentage'] = round(
            current_dict.pop('value_percentage'), 3)
        attributes = ['percentage', 'start_time', 'channel', 'track']
        result = f'volume({", ".join([f"{i}={current_dict[i]}" for i in attributes])})'
        return result


class drum:
    '''
    This class represents a drum beat.
    '''

    def __init__(self,
                 pattern='',
                 mapping=drum_mapping,
                 name=None,
                 notes=None,
                 i=1,
                 start_time=None,
                 default_duration=1 / 8,
                 default_interval=1 / 8,
                 default_volume=100,
                 translate_mode=0):
        self.pattern = pattern
        self.mapping = mapping
        self.name = name
        self.default_duration = default_duration
        self.default_interval = default_interval
        self.default_volume = default_volume
        self.translate_mode = translate_mode
        self.last_non_num_note = None
        self.notes = self.translate(
            self.pattern,
            self.mapping,
            default_duration=self.default_duration,
            default_interval=self.default_interval,
            default_volume=self.default_volume,
            translate_mode=self.translate_mode) if not notes else notes
        if start_time is not None:
            self.notes.start_time = start_time
        self.instrument = i if isinstance(
            i, int) else (drum_set_dict_reverse[i]
                          if i in drum_set_dict_reverse else 1)

    def __repr__(self):
        return f"[drum] {self.name if self.name is not None else ''}\n{self.notes}"

    def translate(self,
                  pattern,
                  mapping,
                  default_duration=1 / 8,
                  default_interval=1 / 8,
                  default_volume=100,
                  translate_mode=0):
        current_rest_symbol = '0'
        current_continue_symbol = '-'
        self.last_non_num_note = None
        if -1 in mapping.values():
            current_rest_symbol = [i for i, j in mapping.items() if j == -1][0]
        if -2 in mapping.values():
            current_continue_symbol = [
                i for i, j in mapping.items() if j == -2
            ][0]
        start_time = 0
        current_has_keyword = False
        whole_parts = []
        whole_keywords = []
        current_keyword = []
        current_part = []
        global_keywords = []
        pattern = pattern.replace(' ', '').replace('\n', '')
        whole_units = [each.split(',') for each in pattern.split('|')]
        repeat_times = 1
        whole_set = False
        for units in whole_units:
            current_has_keyword = False
            for each in units:
                if ':' in each and each[:each.
                                        index(':')] in drum_keywords:
                    current_keyword.append(each)
                    if not current_has_keyword:
                        current_has_keyword = True
                        whole_parts.append(current_part)
                        current_part = []
                elif each.startswith('!'):
                    global_keywords.append(each)
                else:
                    current_part.append(each)
                    if current_has_keyword:
                        current_has_keyword = False
                        whole_keywords.append(current_keyword)
                        current_keyword = []
            if not current_has_keyword:
                whole_parts.append(current_part)
                current_part = []
                whole_keywords.append([])
            else:
                whole_keywords.append(current_keyword)
                current_keyword = []
        notes = []
        durations = []
        intervals = []
        volumes = []
        name_dict = {}

        global_default_duration, global_default_interval, global_default_volume, global_repeat_times, global_all_same_duration, global_all_same_interval, global_all_same_volume, global_fix_length, global_fix_beats = self._translate_global_keyword_parser(
            global_keywords)

        for i in range(len(whole_parts)):
            current_part = whole_parts[i]
            current_keyword = whole_keywords[i]
            current_notes = []
            current_durations = []
            current_intervals = []
            current_volumes = []
            current_custom_durations = []
            current_same_times = []
            if current_part:
                current_part_default_duration, current_part_default_interval, current_part_default_volume, current_part_repeat_times, current_part_all_same_duration, current_part_all_same_interval, current_part_all_same_volume, current_part_fix_length, current_part_name, current_part_fix_beats = self._translate_keyword_parser(
                    current_keyword,
                    default_duration if global_default_duration is None else
                    global_default_duration,
                    default_interval if global_default_interval is None else
                    global_default_interval, default_volume if
                    global_default_volume is None else global_default_volume,
                    global_all_same_duration, global_all_same_interval,
                    global_all_same_volume, global_fix_length,
                    global_fix_beats)
                current_part_fix_length_unit = None
                if current_part_fix_length is not None:
                    if current_part_fix_beats is not None:
                        current_part_fix_length_unit = current_part_fix_length / current_part_fix_beats
                    else:
                        current_part_fix_length_unit = current_part_fix_length / self._get_length(
                            current_part)
                for each in current_part:
                    if each.startswith('i:'):
                        current_extra_interval = _process_note(each[2:])
                        if current_intervals:
                            current_intervals[-1][-1] += current_extra_interval
                        else:
                            if intervals:
                                intervals[-1] += current_extra_interval
                            else:
                                start_time = current_extra_interval
                        continue
                    elif each.startswith('u:'):
                        content = each[2:]
                        if content in name_dict:
                            current_append_notes, current_append_durations, current_append_intervals, current_append_volumes = name_dict[
                                content]
                        else:
                            continue
                    elif '[' in each and ']' in each:
                        current_append_notes, current_append_durations, current_append_intervals, current_append_volumes, current_custom_duration, current_same_time = self._translate_setting_parser(
                            each, mapping, current_part_default_duration,
                            current_part_default_interval,
                            current_part_default_volume, current_rest_symbol,
                            current_continue_symbol,
                            current_part_fix_length_unit, translate_mode)
                    else:
                        current_append_notes, current_append_durations, current_append_intervals, current_append_volumes = self._translate_normal_notes_parser(
                            each, mapping, current_part_default_duration,
                            current_part_default_interval,
                            current_part_default_volume, current_rest_symbol,
                            current_continue_symbol,
                            current_part_fix_length_unit, translate_mode)
                        current_custom_duration = False
                        current_same_time = True
                    current_notes.append(current_append_notes)
                    current_durations.append(current_append_durations)
                    current_intervals.append(current_append_intervals)
                    current_volumes.append(current_append_volumes)
                    current_custom_durations.append(current_custom_duration)
                    current_same_times.append(current_same_time)
                if current_part_all_same_duration is not None:
                    current_durations = [[
                        current_part_all_same_duration for k in each_part
                    ] for each_part in current_durations]
                if current_part_all_same_interval is not None:
                    current_intervals = [
                        [current_part_all_same_interval for k in each_part]
                        if not (len(each_part) > 1 and current_same_times[j])
                        else each_part[:-1] + [current_part_all_same_interval]
                        for j, each_part in enumerate(current_intervals)
                    ]
                if current_part_all_same_volume is not None:
                    current_volumes = [[
                        current_part_all_same_volume for k in each_part
                    ] for each_part in current_volumes]
                current_notes, current_durations, current_intervals, current_volumes = self._split_symbol(
                    [
                        current_notes, current_durations, current_intervals,
                        current_volumes
                    ])

                symbol_inds = [
                    j for j, each_note in enumerate(current_notes)
                    if each_note and isinstance(each_note[0], (
                        rest_symbol, continue_symbol))
                ]
                current_part_start_time = 0
                if symbol_inds:
                    last_symbol_ind = None
                    last_symbol_start_ind = None
                    available_symbol_ind = len(symbol_inds)
                    if symbol_inds[0] == 0:
                        for k in range(1, len(symbol_inds)):
                            if symbol_inds[k] - symbol_inds[k - 1] != 1:
                                available_symbol_ind = k
                                break
                        for ind in symbol_inds[:available_symbol_ind]:
                            current_symbol = current_notes[ind][0]
                            if isinstance(current_symbol, rest_symbol):
                                current_symbol_interval = current_intervals[
                                    ind][0]
                                current_part_start_time += current_symbol_interval
                                if i == 0:
                                    start_time += current_symbol_interval
                                else:
                                    if intervals:
                                        intervals[
                                            -1] += current_symbol_interval
                                    else:
                                        start_time += current_symbol_interval
                    else:
                        available_symbol_ind = 0
                    for ind in symbol_inds[available_symbol_ind:]:
                        if last_symbol_ind is None:
                            last_symbol_ind = ind
                            last_symbol_start_ind = ind - 1
                        else:
                            if any(not isinstance(j[0], (rest_symbol,
                                                         continue_symbol))
                                   for j in current_notes[last_symbol_ind +
                                                          1:ind]):
                                last_symbol_ind = ind
                                last_symbol_start_ind = ind - 1
                        current_symbol = current_notes[ind][0]
                        current_symbol_interval = current_intervals[ind][0]
                        current_symbol_duration = current_durations[ind][0]
                        if isinstance(current_symbol, rest_symbol):
                            last_symbol_interval = current_intervals[
                                last_symbol_start_ind]
                            last_symbol_interval[-1] += current_symbol_interval
                        elif isinstance(current_symbol, continue_symbol):
                            last_symbol_interval = current_intervals[
                                last_symbol_start_ind]
                            last_symbol_duration = current_durations[
                                last_symbol_start_ind]
                            last_symbol_interval[-1] += current_symbol_interval
                            if current_symbol.mode is None:
                                if all(k == 0
                                       for k in last_symbol_interval[:-1]):
                                    for j in range(len(last_symbol_duration)):
                                        last_symbol_duration[
                                            j] += current_symbol_duration
                                else:
                                    last_symbol_duration[
                                        -1] += current_symbol_duration
                            elif current_symbol.mode == 0:
                                last_symbol_duration[
                                    -1] += current_symbol_duration
                            elif current_symbol.mode == 1:
                                for j in range(len(last_symbol_duration)):
                                    last_symbol_duration[
                                        j] += current_symbol_duration
                    current_length = len(current_notes)
                    current_notes = [
                        current_notes[j] for j in range(current_length)
                        if j not in symbol_inds
                    ]
                    current_durations = [
                        current_durations[j] for j in range(current_length)
                        if j not in symbol_inds
                    ]
                    current_intervals = [
                        current_intervals[j] for j in range(current_length)
                        if j not in symbol_inds
                    ]
                    current_volumes = [
                        current_volumes[j] for j in range(current_length)
                        if j not in symbol_inds
                    ]
                current_notes = [j for k in current_notes for j in k]
                current_durations = [j for k in current_durations for j in k]
                current_intervals = [j for k in current_intervals for j in k]
                current_volumes = [j for k in current_volumes for j in k]
                if current_part_repeat_times > 1:
                    current_notes = copy_list(current_notes,
                                              current_part_repeat_times)
                    current_durations = copy_list(current_durations,
                                                  current_part_repeat_times)
                    current_intervals = copy_list(
                        current_intervals,
                        current_part_repeat_times,
                        start_time=current_part_start_time)
                    current_volumes = copy_list(current_volumes,
                                                current_part_repeat_times)
                if current_part_name:
                    name_dict[current_part_name] = [
                        current_notes, current_durations, current_intervals,
                        current_volumes
                    ]
            notes.extend(current_notes)
            durations.extend(current_durations)
            intervals.extend(current_intervals)
            volumes.extend(current_volumes)
        if global_repeat_times > 1:
            notes = copy_list(notes, global_repeat_times)
            durations = copy_list(durations, global_repeat_times)
            intervals = copy_list(intervals,
                                  global_repeat_times,
                                  start_time=start_time)
            volumes = copy_list(volumes, global_repeat_times)
        result = chord(notes) % (durations, intervals, volumes)
        result.start_time = start_time
        return result

    def _split_symbol(self, current_list):
        current_notes = current_list[0]
        return_list = [[] for i in range(len(current_list))]
        for k, each_note in enumerate(current_notes):
            if len(each_note) > 1 and any(
                    isinstance(j, (rest_symbol, continue_symbol))
                    for j in each_note):
                current_return_list = [[] for j in range(len(return_list))]
                current_ind = [
                    k1 for k1, k2 in enumerate(each_note)
                    if isinstance(k2, (rest_symbol, continue_symbol))
                ]
                start_part = [
                    each[k][:current_ind[0]] for each in current_list
                ]
                if start_part[0]:
                    for i, each in enumerate(current_return_list):
                        each.append(start_part[i])
                for j in range(len(current_ind) - 1):
                    current_symbol_part = [[each[k][current_ind[j]]]
                                           for each in current_list]
                    for i, each in enumerate(current_return_list):
                        each.append(current_symbol_part[i])
                    middle_part = [
                        each[k][current_ind[j] + 1:current_ind[j + 1]]
                        for each in current_list
                    ]
                    if middle_part[0]:
                        for i, each in enumerate(current_return_list):
                            each.append(middle_part[i])
                current_symbol_part = [[each[k][current_ind[-1]]]
                                       for each in current_list]
                for i, each in enumerate(current_return_list):
                    each.append(current_symbol_part[i])
                end_part = [
                    each[k][current_ind[-1] + 1:] for each in current_list
                ]
                if end_part[0]:
                    for i, each in enumerate(current_return_list):
                        each.append(end_part[i])
                for i, each in enumerate(return_list):
                    each.extend(current_return_list[i])
            else:
                for i, each in enumerate(return_list):
                    each.append(current_list[i][k])
        return return_list

    def _translate_setting_parser(self, each, mapping, default_duration,
                                  default_interval, default_volume,
                                  current_rest_symbol, current_continue_symbol,
                                  current_part_fix_length_unit,
                                  translate_mode):
        left_bracket_inds = [k for k in range(len(each)) if each[k] == '[']
        right_bracket_inds = [k for k in range(len(each)) if each[k] == ']']
        current_brackets = [
            each[left_bracket_inds[k] + 1:right_bracket_inds[k]]
            for k in range(len(left_bracket_inds))
        ]
        current_append_notes = each[:left_bracket_inds[0]]
        relative_pitch_num = 0
        if '(' in current_append_notes and ')' in current_append_notes:
            current_append_notes, relative_pitch_settings = current_append_notes.split(
                '(', 1)
            relative_pitch_settings = relative_pitch_settings[:-1]
            relative_pitch_num = _parse_change_num(relative_pitch_settings)[0]
        if ';' in current_append_notes:
            current_append_notes = current_append_notes.split(';')
        else:
            current_append_notes = [current_append_notes]
        current_same_time = True
        current_chord_same_time = True
        current_repeat_times = 1
        current_after_repeat_times = 1
        current_fix_length = None
        current_fix_beats = None
        current_inner_fix_beats = 1
        current_append_durations = [
            self._apply_dotted_notes(default_duration, self._get_dotted(i))
            for i in current_append_notes
        ]
        current_append_intervals = [
            self._apply_dotted_notes(default_interval, self._get_dotted(i))
            for i in current_append_notes
        ]
        current_append_volumes = [default_volume for i in current_append_notes]
        if translate_mode == 0:
            current_append_notes = [
                self._convert_to_note(each_note, mapping) if all(
                    not each_note.startswith(j)
                    for j in [current_rest_symbol, current_continue_symbol])
                else self._convert_to_symbol(each_note, current_rest_symbol,
                                             current_continue_symbol)
                for each_note in current_append_notes
            ]
        else:
            new_current_append_notes = []
            for each_note in current_append_notes:
                if ':' not in each_note:
                    if all(not each_note.startswith(j) for j in
                           [current_rest_symbol, current_continue_symbol]):
                        current_each_note = self._convert_to_note(each_note,
                                                                  mode=1)
                        new_current_append_notes.append(current_each_note)
                    else:
                        new_current_append_notes.append(
                            self._convert_to_symbol(each_note,
                                                    current_rest_symbol,
                                                    current_continue_symbol))
                else:
                    current_note, current_chord_type = each_note.split(":")
                    if current_note not in [
                            current_rest_symbol, current_continue_symbol
                    ]:
                        current_note = self._convert_to_note(current_note,
                                                             mode=1)
                        current_each_note = C(
                            f'{current_note.name}{current_chord_type}',
                            current_note.num)
                        for i in current_each_note:
                            i.dotted_num = current_note.dotted_num
                        new_current_append_notes.append(current_each_note)
                        self.last_non_num_note = current_each_note.notes[-1]
            current_append_notes = new_current_append_notes
        if relative_pitch_num != 0:
            dotted_num_list = [i.dotted_num for i in current_append_notes]
            current_append_notes = [
                each_note + relative_pitch_num
                for each_note in current_append_notes
            ]
            for i, each_note in enumerate(current_append_notes):
                each_note.dotted_num = dotted_num_list[i]
            self.last_non_num_note = current_append_notes[-1]
        custom_durations = False
        for j in current_brackets:
            current_bracket_settings = [k.split(':') for k in j.split(';')]
            if all(len(k) == 1 for k in current_bracket_settings):
                current_settings = _process_settings(
                    [k[0] for k in current_bracket_settings])
                current_append_durations, current_append_intervals, current_append_volumes = current_settings
                if current_append_durations is None:
                    current_append_durations = default_duration
                if not isinstance(current_append_durations, list):
                    current_append_durations = [
                        current_append_durations for k in current_append_notes
                    ]
                    custom_durations = True
                if current_append_intervals is None:
                    current_append_intervals = default_interval
                if not isinstance(current_append_intervals, list):
                    current_append_intervals = [
                        current_append_intervals for k in current_append_notes
                    ]
                if current_append_volumes is None:
                    current_append_volumes = default_volume
                if not isinstance(current_append_volumes, list):
                    current_append_volumes = [
                        current_append_volumes for k in current_append_notes
                    ]
            else:
                for each_setting in current_bracket_settings:
                    if len(each_setting) != 2:
                        continue
                    current_setting_keyword, current_content = each_setting
                    if current_setting_keyword == 's':
                        if current_content == 'F':
                            current_same_time = False
                        elif current_content == 'T':
                            current_same_time = True
                    if current_setting_keyword == 'cs':
                        if current_content == 'F':
                            current_chord_same_time = False
                        elif current_content == 'T':
                            current_chord_same_time = True
                    elif current_setting_keyword == 'r':
                        current_repeat_times = int(current_content)
                    elif current_setting_keyword == 'R':
                        current_after_repeat_times = int(current_content)
                    elif current_setting_keyword == 't':
                        current_fix_length = _process_note(current_content)
                    elif current_setting_keyword == 'b':
                        current_fix_beats = _process_note(current_content)
                    elif current_setting_keyword == 'B':
                        current_inner_fix_beats = _process_note(
                            current_content)
                    elif current_setting_keyword == 'i':
                        if current_content == '.':
                            current_append_intervals = _process_note(
                                current_content,
                                mode=1,
                                value2=current_append_durations)
                        else:
                            current_append_intervals = _process_note(
                                current_content)
                        if current_append_intervals is None:
                            current_append_intervals = default_interval
                        if not isinstance(current_append_intervals, list):
                            current_append_intervals = [
                                current_append_intervals
                                for k in current_append_notes
                            ]
                    elif current_setting_keyword == 'l':
                        current_append_durations = _process_note(
                            current_content)
                        if current_append_durations is None:
                            current_append_durations = default_duration
                        if not isinstance(current_append_durations, list):
                            current_append_durations = [
                                current_append_durations
                                for k in current_append_notes
                            ]
                            custom_durations = True
                    elif current_setting_keyword == 'v':
                        current_append_volumes = _process_note(current_content,
                                                               mode=2)
                        if current_append_volumes is None:
                            current_append_volumes = default_volume
                        if not isinstance(current_append_volumes, list):
                            current_append_volumes = [
                                current_append_volumes
                                for k in current_append_notes
                            ]
                    elif current_setting_keyword == 'cm':
                        if len(current_append_notes) == 1 and isinstance(
                                current_append_notes[0], continue_symbol):
                            current_append_notes[0].mode = int(current_content)
        current_fix_length_unit = None
        if current_fix_length is not None:
            if current_same_time:
                current_fix_length_unit = current_fix_length / (
                    current_repeat_times * current_inner_fix_beats)
            else:
                current_fix_length_unit = current_fix_length / (
                    self._get_length(current_append_notes) *
                    current_repeat_times * current_inner_fix_beats)
            if current_fix_beats is not None:
                current_fix_length_unit *= current_fix_beats
        elif current_part_fix_length_unit is not None:
            if current_same_time:
                current_fix_length_unit = current_part_fix_length_unit / current_repeat_times
            else:
                current_fix_length_unit = current_part_fix_length_unit / (
                    self._get_length(current_append_notes) *
                    current_repeat_times)
            if current_fix_beats is not None:
                current_fix_length_unit *= current_fix_beats
        if current_same_time:
            current_append_intervals = [
                0 for k in range(len(current_append_notes) - 1)
            ] + [current_append_intervals[-1]]
        if current_fix_length_unit is not None:
            if current_same_time:
                current_append_intervals = [
                    0 for k in range(len(current_append_notes) - 1)
                ] + [
                    self._apply_dotted_notes(
                        current_fix_length_unit,
                        current_append_notes[-1].dotted_num)
                ]
            else:
                current_append_intervals = [
                    self._apply_dotted_notes(current_fix_length_unit,
                                             k.dotted_num)
                    for k in current_append_notes
                ]
            if not custom_durations:
                current_append_durations = [
                    self._apply_dotted_notes(current_fix_length_unit,
                                             k.dotted_num)
                    for k in current_append_notes
                ]
        if current_repeat_times > 1:
            current_append_notes = copy_list(current_append_notes,
                                             current_repeat_times)
            current_append_durations = copy_list(current_append_durations,
                                                 current_repeat_times)
            current_append_intervals = copy_list(current_append_intervals,
                                                 current_repeat_times)
            current_append_volumes = copy_list(current_append_volumes,
                                               current_repeat_times)

        if current_after_repeat_times > 1:
            current_append_notes = copy_list(current_append_notes,
                                             current_after_repeat_times)
            current_append_durations = copy_list(current_append_durations,
                                                 current_after_repeat_times)
            current_append_intervals = copy_list(current_append_intervals,
                                                 current_after_repeat_times)
            current_append_volumes = copy_list(current_append_volumes,
                                               current_after_repeat_times)

        if translate_mode == 1:
            new_current_append_durations = []
            new_current_append_intervals = []
            new_current_append_volumes = []
            new_current_append_notes = []
            for i, each in enumerate(current_append_notes):
                if isinstance(each, chord):
                    if not current_chord_same_time:
                        current_duration = [
                            current_append_durations[i] / len(each)
                            for k in each.notes
                        ]
                        current_interval = [
                            current_append_intervals[i] / len(each)
                            for k in each.notes
                        ]
                    else:
                        current_duration = [
                            current_append_intervals[i] for k in each.notes
                        ]
                        current_interval = [0 for j in range(len(each) - 1)
                                            ] + [current_append_intervals[i]]
                    new_current_append_durations.extend(current_duration)
                    new_current_append_intervals.extend(current_interval)
                    new_current_append_volumes.extend(
                        [current_append_volumes[i] for k in each.notes])
                    new_current_append_notes.extend(each.notes)
                else:
                    new_current_append_durations.append(
                        current_append_durations[i])
                    new_current_append_intervals.append(
                        current_append_intervals[i])
                    new_current_append_volumes.append(
                        current_append_volumes[i])
                    new_current_append_notes.append(each)
            current_append_durations = new_current_append_durations
            current_append_intervals = new_current_append_intervals
            current_append_volumes = new_current_append_volumes
            current_append_notes = new_current_append_notes
        return current_append_notes, current_append_durations, current_append_intervals, current_append_volumes, custom_durations, current_same_time

    def _translate_normal_notes_parser(self, each, mapping, default_duration,
                                       default_interval, default_volume,
                                       current_rest_symbol,
                                       current_continue_symbol,
                                       current_part_fix_length_unit,
                                       translate_mode):
        current_append_notes = each
        relative_pitch_num = 0
        if '(' in current_append_notes and ')' in current_append_notes:
            current_append_notes, relative_pitch_settings = current_append_notes.split(
                '(', 1)
            relative_pitch_settings = relative_pitch_settings[:-1]
            relative_pitch_num = _parse_change_num(relative_pitch_settings)[0]
        if ';' in current_append_notes:
            current_append_notes = current_append_notes.split(';')
        else:
            current_append_notes = [current_append_notes]
        current_append_notes = [i for i in current_append_notes if i]
        if translate_mode == 0:
            current_append_notes = [
                self._convert_to_note(each_note, mapping) if all(
                    not each_note.startswith(j)
                    for j in [current_rest_symbol, current_continue_symbol])
                else self._convert_to_symbol(each_note, current_rest_symbol,
                                             current_continue_symbol)
                for each_note in current_append_notes
            ]
        else:
            new_current_append_notes = []
            for each_note in current_append_notes:
                if ':' not in each_note:
                    if all(not each_note.startswith(j) for j in
                           [current_rest_symbol, current_continue_symbol]):
                        current_each_note = self._convert_to_note(each_note,
                                                                  mode=1)
                        new_current_append_notes.append(current_each_note)
                    else:
                        new_current_append_notes.append(
                            self._convert_to_symbol(each_note,
                                                    current_rest_symbol,
                                                    current_continue_symbol))
                else:
                    current_note, current_chord_type = each_note.split(":")
                    if current_note not in [
                            current_rest_symbol, current_continue_symbol
                    ]:
                        current_note = self._convert_to_note(current_note,
                                                             mode=1)
                        current_each_note = C(
                            f'{current_note.name}{current_chord_type}',
                            current_note.num)
                        for i in current_each_note:
                            i.dotted_num = current_note.dotted_num
                        new_current_append_notes.extend(
                            current_each_note.notes)
                        self.last_non_num_note = current_each_note.notes[-1]
            current_append_notes = new_current_append_notes

        if relative_pitch_num != 0:
            dotted_num_list = [i.dotted_num for i in current_append_notes]
            current_append_notes = [
                each_note + relative_pitch_num
                for each_note in current_append_notes
            ]
            for i, each_note in enumerate(current_append_notes):
                each_note.dotted_num = dotted_num_list[i]
            self.last_non_num_note = current_append_notes[-1]
        current_append_durations = [
            self._apply_dotted_notes(default_duration, k.dotted_num)
            if not current_part_fix_length_unit else self._apply_dotted_notes(
                current_part_fix_length_unit, k.dotted_num)
            for k in current_append_notes
        ]
        current_append_intervals = [
            self._apply_dotted_notes(default_interval, k.dotted_num)
            if not current_part_fix_length_unit else self._apply_dotted_notes(
                current_part_fix_length_unit, k.dotted_num)
            for k in current_append_notes
        ]
        current_append_volumes = [default_volume for k in current_append_notes]
        if len(current_append_notes) > 1:
            current_append_intervals = [
                0 for i in range(len(current_append_intervals) - 1)
            ] + [current_append_intervals[-1]]
        return current_append_notes, current_append_durations, current_append_intervals, current_append_volumes

    def _translate_keyword_parser(self, current_keyword, default_duration,
                                  default_interval, default_volume,
                                  default_all_same_duration,
                                  default_all_same_interval,
                                  default_all_same_volume, default_fix_length,
                                  default_fix_beats):
        current_part_default_duration = default_duration
        current_part_default_interval = default_interval
        current_part_default_volume = default_volume
        current_part_repeat_times = 1
        current_part_all_same_duration = default_all_same_duration
        current_part_all_same_interval = default_all_same_interval
        current_part_all_same_volume = default_all_same_volume
        current_part_fix_length = default_fix_length
        current_part_fix_beats = default_fix_beats
        current_part_name = None
        for each in current_keyword:
            keyword, content = each.split(':')
            if keyword == 't':
                current_part_fix_length = _process_note(content)
            elif keyword == 'b':
                current_part_fix_beats = _process_note(content)
            elif keyword == 'r':
                current_part_repeat_times = int(content)
            elif keyword == 'n':
                current_part_name = content
            elif keyword == 'd':
                current_part_default_duration, current_part_default_interval, current_part_default_volume = _process_settings(
                    content.split(';'))
                if current_part_default_duration is None:
                    current_part_default_duration = self.default_duration
                if current_part_default_interval is None:
                    current_part_default_interval = self.default_interval
                if current_part_default_volume is None:
                    current_part_default_volume = self.default_volume
            elif keyword == 'a':
                current_part_all_same_duration, current_part_all_same_interval, current_part_all_same_volume = _process_settings(
                    content.split(';'))
            elif keyword == 'dl':
                current_part_default_duration = _process_note(content)
            elif keyword == 'di':
                current_part_default_interval = _process_note(content)
            elif keyword == 'dv':
                current_part_default_volume = _process_note(content, mode=2)
            elif keyword == 'al':
                current_part_all_same_duration = _process_note(content)
            elif keyword == 'ai':
                current_part_all_same_interval = _process_note(content)
            elif keyword == 'av':
                current_part_all_same_volume = _process_note(content, mode=2)
        return current_part_default_duration, current_part_default_interval, current_part_default_volume, current_part_repeat_times, current_part_all_same_duration, current_part_all_same_interval, current_part_all_same_volume, current_part_fix_length, current_part_name, current_part_fix_beats

    def _translate_global_keyword_parser(self, global_keywords):
        global_default_duration = None
        global_default_interval = None
        global_default_volume = None
        global_repeat_times = 1
        global_all_same_duration = None
        global_all_same_interval = None
        global_all_same_volume = None
        global_fix_length = None
        global_fix_beats = None
        for each in global_keywords:
            keyword, content = each[1:].split(':')
            if keyword == 't':
                global_fix_length = _process_note(content)
            elif keyword == 'b':
                global_fix_beats = _process_note(content)
            elif keyword == 'r':
                global_repeat_times = int(content)
            elif keyword == 'd':
                global_default_duration, global_default_interval, global_default_volume = _process_settings(
                    content.split(';'))
            elif keyword == 'a':
                global_all_same_duration, global_all_same_interval, global_all_same_volume = _process_settings(
                    content.split(';'))
            elif keyword == 'dl':
                global_default_duration = _process_note(content)
            elif keyword == 'di':
                global_default_interval = _process_note(content)
            elif keyword == 'dv':
                global_default_volume = _process_note(content, mode=2)
            elif keyword == 'al':
                global_all_same_duration = _process_note(content)
            elif keyword == 'ai':
                global_all_same_interval = _process_note(content)
            elif keyword == 'av':
                global_all_same_volume = _process_note(content, mode=2)
        return global_default_duration, global_default_interval, global_default_volume, global_repeat_times, global_all_same_duration, global_all_same_interval, global_all_same_volume, global_fix_length, global_fix_beats

    def _convert_to_symbol(self, text, current_rest_symbol,
                           current_continue_symbol):
        dotted_num = 0
        if '.' in text:
            text, dotted = text.split('.', 1)
            dotted_num = len(dotted) + 1
        if text == current_rest_symbol:
            result = rest_symbol()
        elif text == current_continue_symbol:
            result = continue_symbol()
        result.dotted_num = dotted_num
        result.mode = None
        return result

    def _convert_to_note(self, text, mapping=None, mode=0):
        dotted_num = 0
        if text.startswith('+') or text.startswith('-'):
            current_num, current_changed, dotted_num = _parse_change_num(text)
            if self.last_non_num_note is not None:
                result = self.last_non_num_note + current_num
                result.dotted_num = dotted_num
            else:
                raise ValueError(
                    'requires at least a previous non-number note')
            if current_changed:
                self.last_non_num_note = result
        else:
            if '.' in text:
                text, dotted = text.split('.', 1)
                dotted_num = len(dotted) + 1
            if mode == 0:
                result = degree_to_note(mapping[text])
            else:
                result = N(text)
            result.dotted_num = dotted_num
            self.last_non_num_note = result

        return result

    def _get_dotted(self, text):
        result = 0
        if isinstance(text, str):
            if ':' in text:
                text = text.split(':', 1)[0]
            if '.' in text:
                ind = text.index('.')
                ind2 = len(text)
                if '[' in text:
                    ind2 = text.index('[')
                if ind2 > ind:
                    if all(i == '.' for i in text[ind:ind2]):
                        text, dotted = text[:ind2].split('.', 1)
                        dotted += '.'
                        result = len(dotted)
                    else:
                        raise Exception(
                            'for drum notes group, dotted notes syntax should be placed after the last note'
                        )
        else:
            result = text.dotted_num
        return result

    def _get_length(self, notes):
        return sum([
            dotted(1, self._get_dotted(i)) for i in notes
            if not (isinstance(i, str) and i.startswith('i:'))
        ])

    def _apply_dotted_notes(self, current_part_fix_length_unit, dotted_num):
        return dotted(current_part_fix_length_unit, dotted_num)

    def __mul__(self, n):
        temp = copy(self)
        temp.notes *= n
        return temp

    def __add__(self, other):
        temp = copy(self)
        if isinstance(other, tuple) and isinstance(other[0], drum):
            other = (other[0].notes, ) + other[1:]
        temp.notes += (other.notes if isinstance(other, drum) else other)
        return temp

    def __and__(self, other):
        temp = copy(self)
        if isinstance(other, tuple) and isinstance(other[0], drum):
            other = (other[0].notes, ) + other[1:]
        temp.notes &= (other.notes if isinstance(other, drum) else other)
        return temp

    def __or__(self, other):
        temp = copy(self)
        if isinstance(other, tuple) and isinstance(other[0], drum):
            other = (other[0].notes, ) + other[1:]
        temp.notes |= (other.notes if isinstance(other, drum) else other)
        return temp

    def set(self, durations=None, intervals=None, volumes=None):
        return self % (durations, intervals, volumes)

    def with_start(self, start_time):
        temp = copy(self)
        temp.notes.start_time = start_time
        return temp


class event:

    def __init__(self, type, track=0, start_time=0, is_meta=False, **kwargs):
        self.type = type
        self.track = track
        self.start_time = start_time
        self.is_meta = is_meta
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'event({", ".join([f"{i}={j}" for i, j in self.__dict__.items()])})'


class beat:
    '''
    This class represents a single beat.
    '''

    def __init__(self, duration=1 / 4, dotted=None):
        self.rhythm_name = 'beat'
        self.duration = duration
        self.dotted = dotted

    def __repr__(self):
        current_duration = Fraction(self.duration).limit_denominator()
        dotted_part = "." * (self.dotted if self.dotted is not None else 0)
        return f'{self.rhythm_name}({current_duration}{dotted_part})'

    def get_duration(self):
        if self.dotted is not None and self.dotted != 0:
            duration = self.duration * sum([(1 / 2)**i
                                            for i in range(self.dotted + 1)])
        else:
            duration = self.duration
        return duration


class rest_symbol(beat):
    '''
    This class represents a single rest.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhythm_name = 'rest'


class continue_symbol(beat):
    '''
    This class represents a single continuation of previous note.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhythm_name = 'continue'


class rest(rest_symbol):
    pass


class rhythm(list):
    '''
    This class represents a rhythm, which consists of beats, rest symbols and continue symbols.
    '''

    def __init__(self,
                 beat_list,
                 total_bar_length=None,
                 beats=None,
                 time_signature=None,
                 separator=' ',
                 unit=None):
        is_str = False
        settings_list = []
        if isinstance(beat_list, str):
            is_str = True
            beat_list, settings_list = self._convert_to_rhythm(
                beat_list, separator)
        if time_signature is None:
            self.time_signature = [4, 4]
        else:
            self.time_signature = time_signature
        current_duration = None
        if total_bar_length is not None:
            if beat_list:
                current_time_signature_ratio = self.time_signature[
                    0] / self.time_signature[1]
                current_duration = total_bar_length * current_time_signature_ratio / (
                    self.get_length(beat_list) if beats is None else beats)
        elif unit is not None:
            current_duration = unit
        if not is_str:
            if current_duration is not None:
                for each in beat_list:
                    each.duration = current_duration
        else:
            new_beat_list = []
            for i, each in enumerate(beat_list):
                current_beat = [each]
                current_repeat_times, current_beats_num, current_after_repeat_times = settings_list[
                    i]
                current_beat_duration = current_duration if current_duration is not None else each.duration
                current_new_duration = current_beat_duration * current_beats_num / current_repeat_times
                if current_repeat_times > 1:
                    current_beat = [
                        copy(each) for j in range(current_repeat_times)
                    ]
                for k in current_beat:
                    k.duration = current_new_duration
                if current_after_repeat_times > 1:
                    current_beat = copy_list(current_beat,
                                             current_after_repeat_times)
                new_beat_list.extend(current_beat)
            beat_list = new_beat_list

        super().__init__(beat_list)

    @property
    def total_bar_length(self):
        return self.get_total_duration(apply_time_signature=True)

    def __repr__(self):
        current_total_bar_length = Fraction(
            self.total_bar_length).limit_denominator()
        current_rhythm = ', '.join([str(i) for i in self])
        return f'[rhythm]\nrhythm: {current_rhythm}\ntotal bar length: {current_total_bar_length}\ntime signature: {self.time_signature[0]} / {self.time_signature[1]}'

    def _convert_to_rhythm(self, current_rhythm, separator=' '):
        settings_list = []
        current_beat_list = current_rhythm.split(separator)
        current_beat_list = [i.strip() for i in current_beat_list if i]
        for i, each in enumerate(current_beat_list):
            current_settings = None
            if '[' in each and ']' in each:
                each, current_settings = each.split('[')
            if ':' in each:
                current, duration = each.split(':')
                duration = _process_note(duration)
            else:
                current, duration = each, 1 / 4
            current_beat = None
            if current.startswith('b') and all(j == '.' for j in current[1:]):
                dotted_num = len(current[1:])
                current_beat = beat(
                    duration=duration,
                    dotted=dotted_num if dotted_num != 0 else None)
            elif current.startswith('-') and all(j == '.'
                                                 for j in current[1:]):
                dotted_num = len(current[1:])
                current_beat = continue_symbol(
                    duration=duration,
                    dotted=dotted_num if dotted_num != 0 else None)
            elif current.startswith('0') and all(j == '.'
                                                 for j in current[1:]):
                dotted_num = len(current[1:])
                current_beat = rest_symbol(
                    duration=duration,
                    dotted=dotted_num if dotted_num != 0 else None)
            if current_beat is not None:
                current_repeat_times = 1
                current_after_repeat_times = 1
                current_beats = 1
                if current_settings is not None:
                    current_settings = [
                        j.strip().split(':')
                        for j in current_settings[:-1].split(';')
                    ]
                    for k in current_settings:
                        current_keyword, current_content = k
                        if current_keyword == 'r':
                            current_repeat_times = int(current_content)
                        elif current_keyword == 'R':
                            current_after_repeat_times = int(current_content)
                        if current_keyword == 'b':
                            current_beats = _process_note(current_content)
                current_beat_list[i] = current_beat
                settings_list.append([
                    current_repeat_times, current_beats,
                    current_after_repeat_times
                ])
        return current_beat_list, settings_list

    def __add__(self, *args, **kwargs):
        return rhythm(super().__add__(*args, **kwargs))

    def __mul__(self, *args, **kwargs):
        return rhythm(super().__mul__(*args, **kwargs))

    def get_length(self, beat_list=None):
        return sum([
            1 if i.dotted is None else dotted(1, i.dotted)
            for i in (beat_list if beat_list is not None else self)
        ])

    def get_beat_num(self, beat_list=None):
        return len([
            i for i in (beat_list if beat_list is not None else self)
            if type(i) is beat
        ])

    def get_total_duration(self, beat_list=None, apply_time_signature=False):
        result = sum([
            i.get_duration()
            for i in (beat_list if beat_list is not None else self)
        ])
        if apply_time_signature:
            result /= (self.time_signature[0] / self.time_signature[1])
        return result

    def play(self, *args, notes='C4', **kwargs):
        result = chord([copy(notes) for i in range(self.get_beat_num())
                        ]).apply_rhythm(self)
        result.play(*args, **kwargs)

    def convert_time_signature(self, time_signature, mode=0):
        temp = copy(self)
        ratio = (time_signature[0] / time_signature[1]) / (
            self.time_signature[0] / self.time_signature[1])
        tetime_signature = time_signature
        if mode == 1:
            for each in temp:
                each.duration *= ratio
        return temp


@dataclass
class chord_type:
    '''
    This class represents a chord type, which defines how a chord is derived precisely.
    '''
    root: str = None
    chord_type: str = None
    chord_speciality: str = 'root position'
    inversion: int = None
    omit: list = None
    altered: list = None
    non_chord_bass_note: str = None
    voicing: list = None
    type: str = 'chord'
    note_name: str = None
    interval_name: str = None
    polychords: list = None
    order: list = None

    def get_root_position(self):
        if self.root is not None and self.chord_type is not None:
            return f'{self.root}{self.chord_type}'
        else:
            return None

    def to_chord(self,
                 root_position=False,
                 apply_voicing=True,
                 apply_omit=True,
                 apply_altered=True,
                 apply_non_chord_bass_note=True,
                 apply_inversion=True,
                 custom_mapping=None,
                 custom_order=None,
                 root_octave=None):
        if self.type == 'note':
            return chord([self.note_name])
        elif self.type == 'interval':
            current_root = N(self.root)
            if root_octave is not None:
                current_root.num = root_octave
            return chord([
                current_root,
                current_root.up(NAME_OF_INTERVAL[self.interval_name])
            ])
        elif self.type == 'chord':
            if self.chord_speciality == 'polychord':
                current_chords = [
                    each.to_chord(
                        root_position=root_position,
                        apply_voicing=apply_voicing,
                        apply_omit=apply_omit,
                        apply_altered=apply_altered,
                        apply_non_chord_bass_note=apply_non_chord_bass_note,
                        apply_inversion=apply_inversion,
                        custom_mapping=custom_mapping,
                        custom_order=custom_order)
                    for each in self.polychords[::-1]
                ]
                current = functools.reduce(chord.on, current_chords)
            else:
                if self.root is None or self.chord_type is None:
                    return None
                current = C(self.get_root_position(),
                               custom_mapping=custom_mapping)
                if not root_position:
                    if custom_order is not None:
                        current_order = custom_order
                    else:
                        if self.order is not None:
                            current_order = self.order
                        else:
                            current_order = [0, 1, 2, 3, 4]
                    current_apply = [
                        apply_omit, apply_altered, apply_inversion,
                        apply_voicing, apply_non_chord_bass_note
                    ]
                    for each in current_order:
                        if current_apply[each]:
                            current = self._apply_order(current, each)
            if root_octave is not None:
                current = current.reset_octave(root_octave)
            return current

    def _apply_order(self, current, order):
        '''
        order is an integer that represents a type of chord alternation
        0: omit some notes
        1: alter some notes
        2: inversion
        3: chord voicing
        4: add a non-chord bass note
        '''
        if order == 0:
            if self.omit:
                current = current.omit([
                    precise_degree_match.get(i.split('/')[0], i)
                    for i in self.omit
                ],
                                       mode=1)
        elif order == 1:
            if self.altered:
                current = current(','.join(self.altered))
        elif order == 2:
            if self.inversion:
                current = current.inversion(self.inversion)
        elif order == 3:
            if self.voicing:
                current = current @ self.voicing
        elif order == 4:
            if self.non_chord_bass_note:
                current = current.on(self.non_chord_bass_note)
        return current

    def _add_order(self, order):
        if self.order is not None:
            if order in self.order:
                self.order.remove(order)
            self.order.append(order)

    def to_text(self,
                show_degree=True,
                show_voicing=True,
                custom_mapping=None):
        if self.type == 'note':
            return f'note {self.note_name}'
        elif self.type == 'interval':
            return f'{self.root} with {self.interval_name}'
        elif self.type == 'chord':
            if self.chord_speciality == 'polychord':
                return '/'.join([
                    f'[{i.to_text(show_degree=show_degree, show_voicing=show_voicing, custom_mapping=custom_mapping)}]'
                    for i in self.polychords[::-1]
                ])
            else:
                if self.root is None or self.chord_type is None:
                    return None
                current_chord = C(self.get_root_position(),
                                     custom_mapping=custom_mapping)
                if self.altered:
                    if show_degree:
                        altered_msg = ', '.join(self.altered)
                    else:
                        current_alter = []
                        for i in self.altered:
                            if i[1:].isdigit():
                                current_degree = current_chord.interval_note(
                                    i[1:])
                                if current_degree is not None:
                                    current_alter.append(i[0] +
                                                         current_degree.name)
                            else:
                                current_alter.append(i)
                        altered_msg = ', '.join(current_alter)
                else:
                    altered_msg = ''
                if self.omit:
                    if show_degree:
                        omit_msg = f'omit {", ".join([i if not ("/" not in i and (i.startswith("b") or i.startswith("#"))) else i[1:] for i in self.omit])}'
                    else:
                        current_omit = []
                        for i in self.omit:
                            i = i.split('/')[0]
                            if i in precise_degree_match:
                                current_degree = current_chord.interval_note(
                                    i, mode=1)
                                if current_degree is not None:
                                    current_omit.append(current_degree.name)
                            else:
                                if i.startswith('b') or i.startswith('#'):
                                    i = i[1:]
                                current_omit.append(i)
                        omit_msg = f'omit {", ".join(current_omit)}'
                else:
                    omit_msg = ''
                voicing_msg = f'sort as {self.voicing}' if self.voicing else ''
                non_chord_bass_note_msg = f'/{self.non_chord_bass_note}' if self.non_chord_bass_note else ''
                if self.inversion:
                    current_new_chord = self.to_chord(
                        custom_mapping=custom_mapping, apply_inversion=False)
                    inversion_msg = f'/{current_new_chord[self.inversion].name}'
                else:
                    inversion_msg = ''
                result = f'{self.root}{self.chord_type}'
                other_msg = [
                    omit_msg, altered_msg, inversion_msg, voicing_msg,
                    non_chord_bass_note_msg
                ]
                if not self.order:
                    current_order = [0, 1, 2, 3, 4]
                else:
                    current_order = self.order
                if not show_voicing and 3 in current_order:
                    current_order.remove(3)
                other_msg = [other_msg[i] for i in current_order]
                other_msg = [i for i in other_msg if i]
                if other_msg:
                    if other_msg[0] != inversion_msg:
                        result += ' '
                    result += ' '.join(other_msg)
                return result

    def clear(self):
        self.root = None
        self.chord_type = None
        self.chord_speciality = 'root position'
        self.inversion = None
        self.omit = None
        self.altered = None
        self.non_chord_bass_note = None
        self.voicing = None
        self.type = 'chord'
        self.note_name = None
        self.interval_name = None
        self.order = []

    def apply_sort_msg(self, msg, change_order=False):
        if isinstance(msg, int):
            self.chord_speciality = 'inverted chord'
            self.inversion = msg
            if change_order and self.order is not None:
                self._add_order(2)
        else:
            self.chord_speciality = 'chord voicings'
            self.voicing = msg
            if change_order and self.order is not None:
                self._add_order(3)

    def simplify(self):
        if self.inversion is not None and self.voicing is not None:
            current_chord1 = self.to_chord()
            current_chord2 = self.to_chord(apply_inversion=False,
                                           apply_voicing=False)
            current_inversion_way = inversion_way(
                current_chord1, current_chord2)
            if isinstance(current_inversion_way, int):
                self.inversion = current_inversion_way
                self.voicing = None
                self.chord_speciality = 'inverted chord'
                self._add_order(2)
            elif isinstance(current_inversion_way, list):
                self.inversion = None
                self.voicing = current_inversion_way
                self.chord_speciality = 'chord voicings'
                self._add_order(3)

    def show(self, **to_text_args):
        current_vars = vars(self)
        if self.type == 'note':
            current = '\n'.join([
                f'{i.replace("_", " ")}: {current_vars[i]}'
                for i in ['type', 'note_name']
            ])
        elif self.type == 'interval':
            current = '\n'.join([
                f'{i.replace("_", " ")}: {current_vars[i]}'
                for i in ['type', 'root', 'interval_name']
            ])
        elif self.type == 'chord':
            current = [f'type: {self.type}'] + [
                f'{i.replace("_", " ")}: {j}'
                for i, j in current_vars.items() if i not in [
                    'type', 'note_name', 'interval_name', 'highest_ratio',
                    'order'
                ]
            ]
            if self.chord_speciality == 'polychord':
                for i, each in enumerate(current):
                    if each.startswith('polychords:'):
                        current[
                            i] = f'polychords: {[i.to_text(**to_text_args) for i in self.polychords]}'
                        break
            current = '\n'.join(current)
        return current

    def get_complexity(self):
        score = 0
        if self.type == 'chord':
            if self.chord_speciality == 'polychord':
                score += 100
            else:
                if self.inversion is not None:
                    score += 10
                if self.omit is not None:
                    score += 10 * len(self.omit)
                if self.altered is not None:
                    score += 30 * len(self.altered)
                if self.non_chord_bass_note is not None:
                    score += 30
                if self.voicing is not None:
                    score += 10
        return score


def _read_notes(note_ls,
                rootpitch=4,
                default_duration=1 / 4,
                default_interval=0,
                default_volume=100):
    intervals = []
    notes_result = []
    start_time = 0
    last_non_num_note = None
    for each in note_ls:
        if each == '':
            continue
        if isinstance(each, note):
            notes_result.append(each)
            intervals.append(default_interval)
            last_non_num_note = notes_result[-1]
        elif isinstance(each, rest):
            if not notes_result:
                start_time += each.get_duration()
            elif intervals:
                intervals[-1] += each.get_duration()
        elif isinstance(each, str):
            has_settings = False
            duration = default_duration
            interval = default_interval
            volume = default_volume
            relative_pitch_num = 0
            if '[' in each and ']' in each:
                has_settings = True
                each, current_settings = each.split('[', 1)
                current_settings = current_settings[:-1].split(';')
                current_settings_len = len(current_settings)
                if current_settings_len == 1:
                    duration = _process_note(current_settings[0])
                else:
                    if current_settings_len == 2:
                        duration, interval = current_settings
                    else:
                        duration, interval, volume = current_settings
                        volume = parse_num(volume)
                    duration = _process_note(duration)
                    interval = _process_note(
                        interval) if interval != '.' else duration
            if '(' in each and ')' in each:
                each, relative_pitch_settings = each.split('(', 1)
                relative_pitch_settings = relative_pitch_settings[:-1]
                relative_pitch_num = _parse_change_num(
                    relative_pitch_settings)[0]
            current_notes = each.split(';')
            current_length = len(current_notes)
            for i, each_note in enumerate(current_notes):
                has_same_time = True
                if i == current_length - 1:
                    has_same_time = False
                last_non_num_note, notes_result, intervals, start_time = _read_single_note(
                    each_note,
                    rootpitch,
                    duration,
                    interval,
                    volume,
                    last_non_num_note,
                    notes_result,
                    intervals,
                    start_time,
                    has_settings=has_settings,
                    has_same_time=has_same_time,
                    relative_pitch_num=relative_pitch_num)
        else:
            notes_result.append(each)
            intervals.append(default_interval)
    if len(intervals) != len(notes_result):
        intervals = []
    return notes_result, intervals, start_time


def _read_single_note(each,
                      rootpitch,
                      duration,
                      interval,
                      volume,
                      last_non_num_note,
                      notes_result,
                      intervals,
                      start_time,
                      has_settings=False,
                      has_same_time=False,
                      relative_pitch_num=0):
    dotted_num = 0
    if '.' in each:
        each, dottedn = each.split('.', 1)
        dotted_num = len(dottedn) + 1
    if each == 'r':
        current_interval = duration if has_settings else (
            dotted(interval, dotted_num) if interval != 0 else 1 / 4)
        if not notes_result:
            start_time += current_interval
        elif intervals:
            intervals[-1] += current_interval
    elif each == '-':
        current_interval = duration if has_settings else (
            dotted(interval, dotted_num) if interval != 0 else 1 / 4)
        if notes_result:
            notes_result[-1].duration += current_interval
        if intervals:
            intervals[-1] += current_interval
    elif each != '-' and (each.startswith('+') or each.startswith('-')):
        current_num, current_changed, current_dotted_num = _parse_change_num(
            each)
        if last_non_num_note is None:
            raise ValueError('requires at least a previous non-number note')
        current_note = last_non_num_note + current_num + relative_pitch_num
        current_note.duration = duration
        current_note.volume = volume
        current_interval = interval
        if has_same_time:
            current_interval = 0
            if not has_settings:
                current_note.duration = dotted(current_note.duration,
                                                  dotted_num)
        else:
            if has_settings:
                current_interval = interval
            else:
                current_interval = dotted(current_interval, dotted_num)
                current_note.duration = dotted(current_note.duration,
                                                  dotted_num)
        if current_changed:
            last_non_num_note = current_note
        notes_result.append(current_note)
        intervals.append(current_interval)
    else:
        current_note = to_note(
            each, duration=duration, volume=volume,
            pitch=rootpitch) + relative_pitch_num
        if has_same_time:
            current_interval = 0
            if not has_settings:
                current_note.duration = dotted(current_note.duration,
                                                  dotted_num)
        else:
            if has_settings:
                current_interval = interval
            else:
                current_interval = dotted(interval, dotted_num)
                current_note.duration = dotted(current_note.duration,
                                                  dotted_num)
        notes_result.append(current_note)
        intervals.append(current_interval)
        last_non_num_note = current_note
    return last_non_num_note, notes_result, intervals, start_time


def _parse_change_num(each):
    current_changed = False
    dotted_num = 0
    if '.' in each:
        each, dotted = each.split('.', 1)
        dotted_num = len(dotted) + 1
    if each.startswith('++'):
        current_changed = True
        current_content = each.split('++', 1)[1]
        if 'o' in current_content:
            current_octave, current_extra = current_content.split('o', 1)
            current_octave = int(current_octave)
            if current_extra:
                current_extra = int(current_extra)
            else:
                current_extra = 0
            current_num = current_octave * octave + current_extra
        else:
            current_num = int(current_content)
    elif each.startswith('--'):
        current_changed = True
        current_content = each.split('--', 1)[1]
        if 'o' in current_content:
            current_octave, current_extra = current_content.split('o', 1)
            current_octave = int(current_octave)
            if current_extra:
                current_extra = int(current_extra)
            else:
                current_extra = 0
            current_num = -(current_octave * octave + current_extra)
        else:
            current_num = -int(current_content)
    elif each.startswith('+'):
        current_content = each.split('+', 1)[1]
        if 'o' in current_content:
            current_octave, current_extra = current_content.split('o', 1)
            current_octave = int(current_octave)
            if current_extra:
                current_extra = int(current_extra)
            else:
                current_extra = 0
            current_num = current_octave * octave + current_extra
        else:
            current_num = int(current_content)
    elif each.startswith('-'):
        current_content = each.split('-', 1)[1]
        if 'o' in current_content:
            current_octave, current_extra = current_content.split('o', 1)
            current_octave = int(current_octave)
            if current_extra:
                current_extra = int(current_extra)
            else:
                current_extra = 0
            current_num = -(current_octave * octave + current_extra)
        else:
            current_num = -int(current_content)
    else:
        raise ValueError('Invalid relative pitch syntax')
    return current_num, current_changed, dotted_num


def _process_note(value, mode=0, value2=None):
    if mode == 1 and value == '.':
        return value2
    if ';' in value:
        result = [_process_note(i) for i in value.split(';')]
        if mode == 2:
            result = [
                int(i) if isinstance(i, float) and i.is_integer() else i
                for i in result
            ]
        return result
    elif value == 'n':
        return None
    else:
        length = len(value)
        if value[0] != '.':
            num_ind = length - 1
            for k in range(num_ind, -1, -1):
                if value[k] != '.':
                    num_ind = k
                    break
            dotted_notes = value[num_ind + 1:].count('.')
            value = value[:num_ind + 1]
            value = parse_num(value) * sum(
                [(1 / 2)**i for i in range(dotted_notes + 1)])
        elif length > 1:
            num_ind = 0
            for k, each in enumerate(value):
                if each != '.':
                    num_ind = k
                    break
            if value[-1] != '.':
                value = 1 / parse_num(value[num_ind:])
            else:
                dotted_notes_start_ind = length - 1
                for k in range(dotted_notes_start_ind, -1, -1):
                    if value[k] != '.':
                        dotted_notes_start_ind = k + 1
                        break
                dotted_notes = length - dotted_notes_start_ind
                value = (1 / parse_num(
                    value[num_ind:dotted_notes_start_ind])) * sum(
                        [(1 / 2)**i for i in range(dotted_notes + 1)])
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return value


def _process_settings(settings):
    settings += ['n' for i in range(3 - len(settings))]
    duration, interval, volume = settings
    duration = _process_note(duration)
    interval = _process_note(interval, mode=1, value2=duration)
    volume = _process_note(volume, mode=2)
    return [duration, interval, volume]


def _process_normalize_tempo(obj, tempo_changes_ranges, bpm, mode=0):
    whole_notes = obj.notes
    intervals = obj.interval
    count_length = 0
    for k in range(len(obj)):
        current_note = whole_notes[k]
        current_interval = intervals[k]
        if mode == 0:
            current_note_left, current_note_right = count_length, count_length + current_note.duration
            new_note_duration = 0
        current_interval_left, current_interval_right = count_length, count_length + current_interval
        new_interval_duration = 0
        for each in tempo_changes_ranges:
            each_left, each_right, each_tempo = each
            if mode == 0:
                if not (current_note_left >= each_right
                        or current_note_right <= each_left):
                    valid_length = min(current_note_right, each_right) - max(
                        current_note_left, each_left)
                    current_ratio = each_tempo / bpm
                    valid_length /= current_ratio
                    new_note_duration += valid_length

            if not (current_interval_left >= each_right
                    or current_interval_right <= each_left):
                valid_length = min(current_interval_right, each_right) - max(
                    current_interval_left, each_left)
                current_ratio = each_tempo / bpm
                valid_length /= current_ratio
                new_interval_duration += valid_length
        if mode == 0:
            current_note.duration = new_note_duration
        obj.interval[k] = new_interval_duration
        count_length += current_interval


def _piece_process_normalize_tempo(self,
                                   bpm,
                                   first_track_start_time,
                                   original_bpm=None):
    other_messages = self.other_messages
    temp = copy(self)
    start_time_ls = temp.start_times
    all_tracks = tetracks
    length = len(all_tracks)
    for k in range(length):
        current_track = all_tracks[k]
        for each in current_track.notes:
            each.track_num = k
        for each in current_track.tempos:
            each.track_num = k
        for each in current_track.pitch_bends:
            each.track_num = k

    first_track_ind = start_time_ls.index(first_track_start_time)
    start_time_ls.insert(0, start_time_ls.pop(first_track_ind))

    all_tracks.insert(0, all_tracks.pop(first_track_ind))
    first_track = all_tracks[0]

    for i in range(1, length):
        current_track = all_tracks[i]
        current_start_time = start_time_ls[i]
        current_shift = current_start_time - first_track_start_time
        first_track = first_track.add(current_track,
                                      start=current_shift,
                                      mode='head',
                                      adjust_msg=False)
    first_track.other_messages = other_messages
    if self.pan:
        for k in range(len(self.pan)):
            current_pan = self.pan[k]
            for each in current_pan:
                each.track = k
    if self.volume:
        for k in range(len(self.volume)):
            current_volume = self.volume[k]
            for each in current_volume:
                each.track = k
    whole_pan = concat(self.pan) if self.pan else None
    whole_volume = concat(self.volume) if self.volume else None
    current_start_time = first_track_start_time + first_track.start_time
    normalize_values = first_track.normalize_tempo(
        bpm,
        start_time=current_start_time,
        pan_msg=whole_pan,
        volume_msg=whole_volume,
        original_bpm=original_bpm)
    if normalize_values:
        normalize_result, first_track_start_time = normalize_values
    else:
        normalize_result = None
        first_track_start_time = current_start_time
    if normalize_result:
        new_other_messages = normalize_result[0]
        self.other_messages = new_other_messages
        if whole_pan or whole_volume:
            whole_pan, whole_volume = normalize_result[1], normalize_result[2]
            self.pan = [[i for i in whole_pan if i.track == j]
                        for j in range(len(self.tracks))]
            self.volume = [[i for i in whole_volume if i.track == j]
                           for j in range(len(self.tracks))]
    else:
        new_other_messages = self.other_messages
    start_times_inds = [[
        i for i in range(len(first_track))
        if first_track.notes[i].track_num == k
    ] for k in range(length)]
    start_times_inds = [each[0] if each else -1 for each in start_times_inds]
    new_start_times = [
        first_track_start_time + first_track[:k].bars(mode=0) if k != -1 else 0
        for k in start_times_inds
    ]
    new_track_notes = [[] for k in range(length)]
    new_track_inds = [[] for k in range(length)]
    new_track_intervals = [[] for k in range(length)]
    new_track_tempos = [[] for k in range(length)]
    new_track_pitch_bends = [[] for k in range(length)]
    whole_length = len(first_track)
    for j in range(whole_length):
        current_note = first_track.notes[j]
        new_track_notes[current_note.track_num].append(current_note)
        new_track_inds[current_note.track_num].append(j)
    for i in first_track.tempos:
        new_track_tempos[i.track_num].append(i)
    for i in first_track.pitch_bends:
        new_track_pitch_bends[i.track_num].append(i)
    whole_interval = first_track.interval
    new_track_intervals = [[
        sum(whole_interval[inds[i]:inds[i + 1]]) for i in range(len(inds) - 1)
    ] for inds in new_track_inds]
    for i in range(length):
        if new_track_inds[i]:
            new_track_intervals[i].append(
                sum(whole_interval[new_track_inds[i][-1]:]))
    new_tracks = []
    for k in range(length):
        current_track_ind = k
        current_track = chord(
            new_track_notes[current_track_ind],
            interval=new_track_intervals[current_track_ind],
            tempos=new_track_tempos[current_track_ind],
            pitch_bends=new_track_pitch_bends[current_track_ind],
            other_messages=[
                each for each in new_other_messages if each.track == k
            ])
        new_tracks.append(current_track)
    self.tracks = new_tracks
    self.start_times = new_start_times


def copy_list(current_list, n, start_time=0):
    result = []
    unit = copy(current_list)
    for i in range(n):
        current = copy(unit)
        if start_time != 0 and i != n - 1:
            current[-1] += start_time
        result.extend(current)
    return result


process_note = _process_note

#=================================================================================================
# musicpy.py
#=================================================================================================

class MetaSpec_key_signature(mido.midifiles.meta.MetaSpec_key_signature):

    def decode(self, message, data):
        try:
            super().decode(message, data)
        except mido.midifiles.meta.KeySignatureError:
            message.key = None

    def check(self, name, value):
        if value is not None:
            super().check(name, value)


mido.midifiles.meta.add_meta_spec(MetaSpec_key_signature)


def method_wrapper(cls):

    def method_decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        if not isinstance(cls, list):
            types = [cls]
        else:
            types = cls
        for each in types:
            setattr(each, func.__name__, wrapper)
        return func

    return method_decorator


def to_note(notename, duration=1 / 4, volume=100, pitch=4, channel=None):
    num_text = ''.join([x for x in notename if x.isdigit()])
    if not num_text.isdigit():
        num = pitch
    else:
        num = int(num_text)
    name = ''.join([x for x in notename if not x.isdigit()])
    return note(name, num, duration, volume, channel)


def degree_to_note(degree, duration=1 / 4, volume=100, channel=None):
    name = standard_reverse[degree % 12]
    num = (degree // 12) - 1
    return note(name, num, duration, volume, channel)


def degrees_to_chord(ls, *args, **kwargs):
    return chord([degree_to_note(i) for i in ls], *args, **kwargs)


def note_to_degree(obj):
    if not isinstance(obj, note):
        obj = to_note(obj)
    return standard[obj.name] + 12 * (obj.num + 1)


def trans_note(notename, duration=1 / 4, volume=100, pitch=4, channel=None):
    num = ''.join([x for x in notename if x.isdigit()])
    if not num:
        num = pitch
    else:
        num = int(num)
    name = ''.join([x for x in notename if not x.isdigit()])
    return note(name, num, duration, volume, channel)


def to_tuple(obj):
    if isinstance(obj, str):
        return (obj, )
    try:
        return tuple(obj)
    except:
        return (obj, )


def get_freq(y, standard=440):
    if isinstance(y, str):
        y = to_note(y)
    semitones = y.degree - 69
    return standard * 2**(semitones / 12)


def freq_to_note(freq, to_str=False, standard=440):
    quotient = freq / standard
    semitones = round(math.log(quotient, 2) * 12)
    result = N('A4') + semitones
    if to_str:
        return str(result)
    return result


def secondary_dom(root, current_scale='major'):
    if isinstance(root, str):
        root = to_note(root)
    newscale = scale(root, current_scale)
    return newscale.dom_chord()


def secondary_dom7(root, current_scale='major'):
    if isinstance(root, str):
        root = to_note(root)
    newscale = scale(root, current_scale)
    return newscale.dom7_chord()


def get_chord_by_interval(start,
                          interval1,
                          duration=1 / 4,
                          interval=0,
                          cumulative=True,
                          start_time=0):

    if isinstance(start, str):
        start = to_note(start)
    result = [start]
    if cumulative:
        # in this case all the notes has distance only with the start note
        startind = start.degree
        result += [
            degree_to_note(startind + interval1[i])
            for i in range(len(interval1))
        ]
    else:
        # in this case current note and next note has distance corresponding to the given interval
        startind = start.degree
        for i in range(len(interval1)):
            startind += interval1[i]
            result.append(degree_to_note(startind))
    return chord(result, duration, interval, start_time=start_time)


def inversion(current_chord, num=1):
    return current_chord.inversion(num)


def get_chord(start,
              current_chord_type=None,
              duration=1 / 4,
              intervals=None,
              interval=None,
              cumulative=True,
              pitch=4,
              start_time=0,
              custom_mapping=None,
              pitch_interval=True):
    if not isinstance(start, note):
        start = to_note(start, pitch=pitch)
    if interval is not None:
        return get_chord_by_interval(start,
                                     interval,
                                     duration,
                                     intervals,
                                     cumulative,
                                     start_time=start_time)
    pre_chord_type = current_chord_type
    current_chord_type = current_chord_type.lower().replace(' ', '')
    chordlist = [start]
    current_chord_types = chordTypes if custom_mapping is None else custom_mapping
    if pre_chord_type in current_chord_types:
        interval_pre_chord_type = current_chord_types[pre_chord_type]
        interval = interval_pre_chord_type
    else:
        if current_chord_type in current_chord_types:
            interval_current_chord_type = current_chord_types[
                current_chord_type]
            interval = interval_current_chord_type
        else:
            raise ValueError(
                f'could not detect the chord type {current_chord_type}')
    for i in range(len(interval)):
        if pitch_interval:
            current_note = start + interval[i]
        else:
            current_pitch_interval = interval[i]
            if isinstance(current_pitch_interval, Interval):
                current_pitch_interval = current_pitch_interval.value
            current_note = start + current_pitch_interval
        chordlist.append(current_note)
    return chord(chordlist, duration, intervals, start_time=start_time)


def concat(chordlist, mode='+', extra=None, start=None):
    if not chordlist:
        return chordlist
    temp = copy(chordlist[0]) if start is None else start
    start_ind = 1 if start is None else 0
    chordlist = chordlist[start_ind:]
    if mode == '+':
        if not extra:
            for t in chordlist:
                temp += t
        else:
            for t in chordlist:
                temp += (t, extra)
    elif mode == '|':
        if not extra:
            for t in chordlist:
                temp |= t
        else:
            for t in chordlist:
                temp |= (t, extra)
    elif mode == '&':
        if not extra:
            for t in chordlist:
                temp &= t
        else:
            extra_unit = extra
            for t in chordlist:
                temp &= (t, extra)
                extra += extra_unit
    return temp


def multi_voice(*current_chord, method=chord, start_times=None):
    current_chord = [
        method(i) if isinstance(i, str) else i for i in current_chord
    ]
    if start_times is not None:
        current_chord = [current_chord[0]
                         ] + [i.with_start(0) for i in current_chord[1:]]
        result = copy(current_chord[0])
        for i in range(1, len(current_chord)):
            result &= (current_chord[i], start_times[i - 1])
    else:
        result = concat(current_chord, mode='&')
    return result


def read(name,
         is_file=False,
         get_off_drums=False,
         clear_empty_notes=False,
         clear_other_channel_msg=False,
         split_channels=None):
    if is_file:
        name.seek(0)
        try:
            current_midi = mido.MidiFile(file=name, clip=True)
            whole_bpm = find_first_tempo(name, is_file=is_file)
            name.close()
        except Exception as OSError:
            name.seek(0)
            current_midi = mido.MidiFile(file=riff_to_midi(name), clip=True)
            whole_bpm = find_first_tempo(name, is_file=is_file)
            name.close()
        name = getattr(name, 'name', '')
    else:
        try:
            current_midi = mido.MidiFile(name, clip=True)
        except Exception as OSError:
            current_midi = mido.MidiFile(file=riff_to_midi(name), clip=True)
        whole_bpm = find_first_tempo(name, is_file=is_file)
    whole_tracks = current_midi.tracks
    if not whole_tracks:
        raise ValueError(
            'No tracks found in the MIDI file, please check if the input MIDI file is empty'
        )
    current_type = current_midi.type
    current_ticks_per_beat = current_midi.ticks_per_beat
    interval_unit = current_ticks_per_beat * 4
    if split_channels is None:
        if current_type == 0:
            split_channels = True
        elif current_type == 1:
            split_channels = False
        elif current_type == 2:
            split_channels = False
    changes = chord([])
    if not split_channels:
        changes_track_ind = [
            i for i, each in enumerate(whole_tracks)
            if all((i.is_meta or i.type == 'sysex') for i in each)
        ]
        changes_track = [whole_tracks[i] for i in changes_track_ind]
        if changes_track:
            changes = [
                _midi_to_chord(each,
                               interval_unit,
                               add_track_num=split_channels,
                               clear_empty_notes=clear_empty_notes)[0]
                for each in changes_track
            ]
            changes = concat(changes)
        available_tracks = [
            whole_tracks[i] for i in range(len(whole_tracks))
            if i not in changes_track_ind
        ]
        all_tracks = [
            _midi_to_chord(available_tracks[j],
                           interval_unit,
                           whole_bpm,
                           add_track_num=split_channels,
                           clear_empty_notes=clear_empty_notes,
                           track_ind=j) for j in range(len(available_tracks))
        ]
        start_times_list = [j[2] for j in all_tracks]
        if available_tracks:
            channels_list = [[
                i.channel for i in each if hasattr(i, 'channel')
            ] for each in available_tracks]
            channels_list = [each[0] if each else -1 for each in channels_list]
            unassigned_channels = channels_list.count(-1)
            if unassigned_channels > 0:
                free_channel_numbers = [
                    i for i in range(16) if i not in channels_list
                ]
                free_channel_numbers_length = len(free_channel_numbers)
                unassigned_channels_number = []
                for k in range(unassigned_channels):
                    if k < free_channel_numbers_length:
                        unassigned_channels_number.append(
                            free_channel_numbers[k])
                    else:
                        unassigned_channels_number.append(
                            16 + k - free_channel_numbers_length)
                channels_list = [
                    each if each != -1 else unassigned_channels_number.pop(0)
                    for each in channels_list
                ]
        else:
            channels_list = None

        instruments = []
        program_change_events = concat(
            [[i for i in each if i.type == 'program_change']
             for each in available_tracks])
        program_change_events.sort(key=lambda s: s.time)
        for i, each in enumerate(available_tracks):
            current_channel = channels_list[i]
            current_program_change = [
                j for j in program_change_events
                if j.channel == current_channel
            ]
            if current_program_change:
                current_program = current_program_change[0].program + 1
                instruments.append(current_program)
            else:
                instruments.append(1)
        chords_list = [each[0] for each in all_tracks]
        pan_list = [k.pan_list for k in chords_list]
        volume_list = [k.volume_list for k in chords_list]
        tracks_names_list = [[k.name for k in each if k.type == 'track_name']
                             for each in available_tracks]
        if all(j for j in tracks_names_list):
            tracks_names_list = [j[0] for j in tracks_names_list]
        else:
            tracks_names_list = None
        result_piece = piece(tracks=chords_list,
                             instruments=instruments,
                             bpm=whole_bpm,
                             start_times=start_times_list,
                             track_names=tracks_names_list,
                             channels=channels_list,
                             name=os.path.splitext(os.path.basename(name))[0],
                             pan=pan_list,
                             volume=volume_list)
        result_piece.other_messages = concat(
            [each_track.other_messages for each_track in result_piece.tracks],
            start=[])
    else:
        available_tracks = whole_tracks
        channels_numbers = concat(
            [[i.channel for i in j if hasattr(i, 'channel')]
             for j in available_tracks])
        if not channels_numbers:
            raise ValueError(
                'Split channels requires the MIDI file contains channel messages for tracks'
            )
        channels_list = list(set(channels_numbers))
        channels_list.sort()
        channels_num = len(channels_list)
        track_channels = channels_list
        all_tracks = [
            _midi_to_chord(each,
                           interval_unit,
                           whole_bpm,
                           add_track_num=split_channels,
                           clear_empty_notes=clear_empty_notes,
                           track_ind=j,
                           track_channels=track_channels)
            for j, each in enumerate(available_tracks)
        ]
        if len(available_tracks) > 1:
            available_tracks = concat(available_tracks)
            start_time_ls = [j[2] for j in all_tracks]
            first_track_ind = start_time_ls.index(min(start_time_ls))
            all_tracks.insert(0, all_tracks.pop(first_track_ind))
            first_track = all_tracks[0]
            all_track_notes, tempos, first_track_start_time = first_track
            for i in all_tracks[1:]:
                current_track = i[0]
                current_start_time = i[2]
                current_shift = current_start_time - first_track_start_time
                all_track_notes = all_track_notes.add(current_track,
                                                      start=current_shift,
                                                      mode='head',
                                                      adjust_msg=False)
            all_track_notes.other_messages = concat(
                [each[0].other_messages for each in all_tracks])
            all_track_notes.pan_list = concat(
                [k[0].pan_list for k in all_tracks])
            all_track_notes.volume_list = concat(
                [k[0].volume_list for k in all_tracks])
            all_tracks = [all_track_notes, tempos, first_track_start_time]
        else:
            available_tracks = available_tracks[0]
            all_tracks = all_tracks[0]
        pan_list = all_tracks[0].pan_list
        volume_list = all_tracks[0].volume_list
        instruments = []
        program_change_events = [
            i for i in available_tracks if i.type == 'program_change'
        ]
        for i, each in enumerate(channels_list):
            current_program = None
            for j in program_change_events:
                if j.channel == each:
                    current_program = j.program
                    instruments.append(current_program + 1)
                    break
            if current_program is None:
                instruments.append(1)
        tracks_names_list = [
            i.name for i in available_tracks if i.type == 'track_name'
        ]
        rename_track_names = False
        if (not tracks_names_list) or (len(tracks_names_list) != channels_num):
            tracks_names_list = [f'Channel {i+1}' for i in channels_list]
            rename_track_names = True
        result_merge_track = all_tracks[0]
        result_piece = piece(tracks=[chord([]) for i in range(channels_num)],
                             instruments=instruments,
                             bpm=whole_bpm,
                             track_names=tracks_names_list,
                             channels=channels_list,
                             name=os.path.splitext(os.path.basename(name))[0],
                             pan=[[] for i in range(channels_num)],
                             volume=[[] for i in range(channels_num)])
        result_piece.reconstruct(track=result_merge_track,
                                 start_time=result_merge_track.start_time,
                                 include_empty_track=True)
        if len(result_piece.channels) != channels_num:
            pan_list = [
                i for i in pan_list if i.channel in result_piece.channels
            ]
            volume_list = [
                i for i in volume_list if i.channel in result_piece.channels
            ]
            for each in pan_list:
                each.track = result_piece.channels.index(each.channel)
            for each in volume_list:
                each.track = result_piece.channels.index(each.channel)
            for k in range(len(result_piece.tracks)):
                for each in result_piece.tracks[k].pitch_bends:
                    each.track = k
            result_merge_track.other_messages = [
                i for i in result_merge_track.other_messages
                if not (hasattr(i, 'channel')
                        and i.channel not in result_piece.channels)
            ]
            for each in result_merge_track.other_messages:
                if hasattr(each, 'channel'):
                    each.track = result_piece.channels.index(each.channel)
        result_piece.other_messages = result_merge_track.other_messages
        for k in range(len(result_piece)):
            current_other_messages = [
                i for i in result_piece.other_messages if i.track == k
            ]
            result_piece.tracks[k].other_messages = current_other_messages
            current_pan = [i for i in pan_list if i.track == k]
            result_piece.pan[k] = current_pan
            current_volume = [i for i in volume_list if i.track == k]
            result_piece.volume[k] = current_volume
        if not rename_track_names:
            current_track_names = result_piece.get_msg('track_name')
            for i in range(len(current_track_names)):
                result_piece.tracks[i].other_messages.append(
                    current_track_names[i])
    if current_type == 1 and not changes.is_empty():
        if result_piece.tracks:
            first_track = result_piece.tracks[0]
            first_track.notes.extend(changes.notes)
            first_track.interval.extend(changes.interval)
            first_track.tempos.extend(changes.tempos)
            first_track.pitch_bends.extend(changes.pitch_bends)
            first_track.other_messages[0:0] = changes.other_messages
        result_piece.other_messages[0:0] = changes.other_messages

    if clear_other_channel_msg:
        result_piece.other_messages = [
            i for i in result_piece.other_messages
            if not (hasattr(i, 'channel')
                    and i.channel not in result_piece.channels)
        ]
    if get_off_drums:
        result_piece.get_off_drums()
    for i in result_piece.tracks:
        if hasattr(i, 'pan_list'):
            del i.pan_list
        if hasattr(i, 'volume_list'):
            del i.volume_list
    result_piece.ticks_per_beat = current_ticks_per_beat
    return result_piece


def _midi_to_chord(current_track,
                   interval_unit,
                   bpm=None,
                   add_track_num=False,
                   clear_empty_notes=False,
                   track_ind=0,
                   track_channels=None):
    intervals = []
    notelist = []
    tempo_list = []
    pitch_bend_list = []
    notes_len = len(current_track)
    find_first_note = False
    start_time = 0
    current_time = 0
    pan_list = []
    volume_list = []
    other_messages = []

    for i in range(notes_len):
        current_msg = current_track[i]
        current_time += current_msg.time
        if current_msg.type == 'note_on' and current_msg.velocity != 0:
            current_msg_velocity = current_msg.velocity
            current_msg_note = current_msg.note
            current_msg_channel = current_msg.channel
            if not find_first_note:
                find_first_note = True
                start_time = sum(current_track[j].time
                                 for j in range(i + 1)) / interval_unit
                if start_time.is_integer():
                    start_time = int(start_time)
            current_interval = 0
            current_duration = 0
            current_note_interval = 0
            current_note_duration = 0
            find_interval = False
            find_duration = False
            for k in range(i + 1, notes_len):
                new_msg = current_track[k]
                new_msg_type = new_msg.type
                current_interval += new_msg.time
                current_duration += new_msg.time
                if not find_interval:
                    if new_msg_type == 'note_on' and new_msg.velocity != 0:
                        find_interval = True
                        current_interval /= interval_unit
                        if current_interval.is_integer():
                            current_interval = int(current_interval)
                        current_note_interval = current_interval
                if not find_duration:
                    if (
                            new_msg_type == 'note_off' or
                        (new_msg_type == 'note_on' and new_msg.velocity == 0)
                    ) and new_msg.note == current_msg_note and new_msg.channel == current_msg_channel:
                        find_duration = True
                        current_duration /= interval_unit
                        if current_duration.is_integer():
                            current_duration = int(current_duration)
                        current_note_duration = current_duration
                if find_interval and find_duration:
                    break
            if not find_interval:
                current_note_interval = current_note_duration
            current_append_note = degree_to_note(
                current_msg_note,
                duration=current_note_duration,
                volume=current_msg_velocity)
            current_append_note.channel = current_msg_channel
            intervals.append(current_note_interval)
            if add_track_num:
                if track_channels:
                    current_append_note.track_num = track_channels.index(
                        current_msg_channel)
                else:
                    current_append_note.track_num = track_ind
            notelist.append(current_append_note)
        elif current_msg.type == 'set_tempo':
            current_tempo = tempo(mido.tempo2bpm(current_msg.tempo),
                                  current_time / interval_unit,
                                  track=track_ind)
            if add_track_num:
                current_tempo.track_num = track_ind
            tempo_list.append(current_tempo)
        elif current_msg.type == 'pitchwheel':
            current_msg_channel = current_msg.channel
            if track_channels:
                current_track_ind = track_channels.index(current_msg_channel)
            else:
                current_track_ind = track_ind
            current_pitch_bend = pitch_bend(value=current_msg.pitch,
                                            start_time=current_time /
                                            interval_unit,
                                            mode='values',
                                            channel=current_msg_channel,
                                            track=current_track_ind)
            if add_track_num:
                current_pitch_bend.track_num = current_track_ind
            pitch_bend_list.append(current_pitch_bend)
        elif current_msg.type == 'control_change':
            current_msg_channel = current_msg.channel
            if track_channels:
                current_track_ind = track_channels.index(current_msg_channel)
            else:
                current_track_ind = track_ind
            if current_msg.control == 10:
                current_pan_msg = pan(current_msg.value,
                                      current_time / interval_unit,
                                      'value',
                                      channel=current_msg_channel,
                                      track=current_track_ind)
                pan_list.append(current_pan_msg)
            elif current_msg.control == 7:
                current_volume_msg = volume(current_msg.value,
                                            current_time / interval_unit,
                                            'value',
                                            channel=current_msg_channel,
                                            track=current_track_ind)
                volume_list.append(current_volume_msg)
            else:
                _read_other_messages(current_msg, other_messages,
                                     current_time / interval_unit,
                                     current_track_ind)
        else:
            if track_channels and hasattr(current_msg, 'channel'):
                current_msg_channel = current_msg.channel
                current_track_ind = track_channels.index(current_msg_channel)
            else:
                current_track_ind = track_ind
            _read_other_messages(current_msg, other_messages,
                                 current_time / interval_unit,
                                 current_track_ind)
    result = chord(notelist, interval=intervals)
    if clear_empty_notes:
        inds = [i for i, each in enumerate(result.notes) if each.duration > 0]
        result.notes = [result.notes[i] for i in inds]
        result.interval = [result.interval[i] for i in inds]
    result.tempos = tempo_list
    result.pitch_bends = pitch_bend_list
    result.pan_list = pan_list
    result.volume_list = volume_list
    result.other_messages = other_messages
    if bpm is not None:
        return [result, bpm, start_time]
    else:
        return [result, start_time]


def _read_other_messages(message, other_messages, time, track_ind):
    if message.type not in ['note_on', 'note_off']:
        current_attributes = {
            i: j
            for i, j in vars(message).items() if i != 'time'
        }
        current_message = event(track=track_ind,
                                start_time=time,
                                **current_attributes)
        current_message.is_meta = message.is_meta
        other_messages.append(current_message)


def write(current_chord,
          bpm=120,
          channel=0,
          start_time=None,
          name='temp.mid',
          instrument=None,
          i=None,
          save_as_file=True,
          msg=None,
          nomsg=False,
          ticks_per_beat=None,
          ignore_instrument=False,
          ignore_bpm=False,
          ignore_track_names=False,
          **midi_args):
    if i is not None:
        instrument = i
    is_track_type = False
    is_piece_like_type = True
    if isinstance(current_chord, note):
        current_chord = chord([current_chord])
    elif isinstance(current_chord, list):
        current_chord = concat(current_chord, '|')
    if isinstance(current_chord, chord):
        is_track_type = True
        is_piece_like_type = False
        if instrument is None:
            instrument = 1
        current_chord = P(
            tracks=[current_chord],
            instruments=[instrument],
            bpm=bpm,
            channels=[channel],
            start_times=[
                current_chord.start_time if start_time is None else start_time
            ],
            other_messages=current_chord.other_messages)
    elif isinstance(current_chord, track):
        is_track_type = True
        if hasattr(current_chord, 'other_messages'):
            msg = current_chord.other_messages
        else:
            msg = current_chord.content.other_messages
        current_chord = build(current_chord, bpm=current_chord.bpm)
    elif isinstance(current_chord, drum):
        is_track_type = True
        is_piece_like_type = False
        if hasattr(current_chord, 'other_messages'):
            msg = current_chord.other_messages
        current_chord = P(tracks=[current_chord.notes],
                          instruments=[current_chord.instrument],
                          bpm=bpm,
                          start_times=[
                              current_chord.notes.start_time
                              if start_time is None else start_time
                          ],
                          channels=[9])
    track_number, start_times, instruments, bpm, tracks_contents, track_names, channels, pan_msg, volume_msg = \
    current_chord.track_number, current_chord.start_times, current_chord.instruments, current_chord.bpm, current_chord.tracks, current_chord.track_names, current_chord.channels, current_chord.pan, current_chord.volume
    instruments = [
        i if isinstance(i, int) else INSTRUMENTS[i]
        for i in instruments
    ]
    if ticks_per_beat is None:
        if current_chord.ticks_per_beat is not None:
            ticks_per_beat = current_chord.ticks_per_beat
        else:
            ticks_per_beat = 960
    current_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat, **midi_args)
    current_midi.tracks.extend([mido.MidiTrack() for i in range(track_number)])
    if not ignore_bpm:
        current_midi.tracks[0].append(
            mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
    interval_unit = ticks_per_beat * 4
    for i in range(track_number):
        if channels:
            current_channel = channels[i]
        else:
            current_channel = i
        if not ignore_instrument:
            current_program_change_event = mido.Message(
                'program_change',
                channel=current_channel,
                time=0,
                program=instruments[i] - 1)
            current_midi.tracks[i].append(current_program_change_event)
        if not ignore_track_names and track_names:
            current_midi.tracks[i].append(
                mido.MetaMessage('track_name', time=0, name=track_names[i]))

        current_pan_msg = pan_msg[i]
        if current_pan_msg:
            for each in current_pan_msg:
                current_pan_track = i if each.track is None else each.track
                current_pan_channel = current_channel if each.channel is None else each.channel
                current_midi.tracks[
                    current_pan_track if not is_track_type else 0].append(
                        mido.Message('control_change',
                                     channel=current_pan_channel,
                                     time=int(each.start_time * interval_unit),
                                     control=10,
                                     value=each.value))
        current_volume_msg = volume_msg[i]
        if current_volume_msg:
            for each in current_volume_msg:
                current_volume_channel = current_channel if each.channel is None else each.channel
                current_volume_track = i if each.track is None else each.track
                current_midi.tracks[
                    current_volume_track if not is_track_type else 0].append(
                        mido.Message('control_change',
                                     channel=current_volume_channel,
                                     time=int(each.start_time * interval_unit),
                                     control=7,
                                     value=each.value))

        content = tracks_contents[i]
        content_notes = content.notes
        content_intervals = content.interval
        current_start_time = start_times[i]
        if is_piece_like_type:
            current_start_time += content.start_time
        current_start_time = current_start_time * interval_unit
        for j in range(len(content)):
            current_note = content_notes[j]
            current_note_on_message = mido.Message(
                'note_on',
                time=int(current_start_time),
                channel=current_channel
                if current_note.channel is None else current_note.channel,
                note=current_note.degree,
                velocity=current_note.volume)
            current_note_off_message = mido.Message(
                'note_off',
                time=int(current_start_time +
                         current_note.duration * interval_unit),
                channel=current_channel
                if current_note.channel is None else current_note.channel,
                note=current_note.degree,
                velocity=current_note.volume)
            current_midi.tracks[i].append(current_note_on_message)
            current_midi.tracks[i].append(current_note_off_message)
            current_start_time += content_intervals[j] * interval_unit
        for each in content.tempos:
            if each.start_time < 0:
                tempo_change_time = 0
            else:
                tempo_change_time = each.start_time * interval_unit
            current_midi.tracks[0].append(
                mido.MetaMessage('set_tempo',
                                 time=int(tempo_change_time),
                                 tempo=mido.bpm2tempo(each.bpm)))
        for each in content.pitch_bends:
            if each.start_time < 0:
                pitch_bend_time = 0
            else:
                pitch_bend_time = each.start_time * interval_unit
            pitch_bend_track = i if each.track is None else each.track
            pitch_bend_channel = current_channel if each.channel is None else each.channel
            current_midi.tracks[
                pitch_bend_track if not is_track_type else 0].append(
                    mido.Message('pitchwheel',
                                 time=int(pitch_bend_time),
                                 channel=pitch_bend_channel,
                                 pitch=each.value))

    if not nomsg:
        if current_chord.other_messages:
            _add_other_messages(
                current_midi=current_midi,
                other_messages=current_chord.other_messages,
                write_type='piece' if not is_track_type else 'track',
                interval_unit=interval_unit)
        elif msg:
            _add_other_messages(
                current_midi=current_midi,
                other_messages=msg,
                write_type='piece' if not is_track_type else 'track',
                interval_unit=interval_unit)
    for i, each in enumerate(current_midi.tracks):
        reset_control_change_list = [120, 121, 123]
        each.sort(key=lambda s: (s.time, not (s.is_cc() and s.control in
                                              reset_control_change_list)))
        current_relative_time = [each[0].time] + [
            each[j].time - each[j - 1].time for j in range(1, len(each))
        ]
        for k, each_msg in enumerate(each):
            each_msg.time = current_relative_time[k]
    if save_as_file:
        current_midi.save(name)
    else:
        current_io = BytesIO()
        current_midi.save(file=current_io)
        return current_io


def _add_other_messages(current_midi,
                        other_messages,
                        write_type='piece',
                        interval_unit=None):
    for each in other_messages:
        try:
            current_time = int(each.start_time * interval_unit)
            current_attributes = {
                i: j
                for i, j in vars(each).items()
                if i not in ['start_time', 'track', 'is_meta']
            }
            if not each.is_meta:
                current_message = mido.Message(time=current_time,
                                               **current_attributes)
            else:
                current_message = mido.MetaMessage(time=current_time,
                                                   **current_attributes)
        except:
            continue
        current_track = each.track if write_type == 'piece' else 0
        if current_track < len(current_midi.tracks):
            current_midi.tracks[current_track].append(current_message)


def find_first_tempo(file, is_file=False):
    if is_file:
        file.seek(0)
        try:
            current_midi = mido.MidiFile(file=file, clip=True)
            file.close()
        except Exception as OSError:
            file.seek(0)
            current_midi = mido.MidiFile(file=riff_to_midi(file), clip=True)
            file.close()
    else:
        try:
            current_midi = mido.MidiFile(file, clip=True)
        except Exception as OSError:
            current_midi = mido.MidiFile(file=riff_to_midi(file), clip=True)
    for track in current_midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
    return 120


def get_ticks_per_beat(file, is_file=False):
    if is_file:
        file.seek(0)
        try:
            current_midi = mido.MidiFile(file=file, clip=True)
            file.close()
        except Exception as OSError:
            file.seek(0)
            current_midi = mido.MidiFile(file=riff_to_midi(file), clip=True)
            file.close()
    else:
        try:
            current_midi = mido.MidiFile(file, clip=True)
        except Exception as OSError:
            current_midi = mido.MidiFile(file=riff_to_midi(file), clip=True)
    return current_midi.ticks_per_beat


def chord_to_piece(current_chord, bpm=120, start_time=0, has_track_num=False):
    channels_numbers = [i.channel for i in current_chord] + [
        i.channel
        for i in current_chord.other_messages if hasattr(i, 'channel')
    ]
    channels_list = list(set(channels_numbers))
    channels_list = [i for i in channels_list if i is not None]
    channels_list.sort()
    if not channels_list:
        channels_list = [0]
    channels_num = len(channels_list)
    current_start_time = start_time + current_chord.start_time
    pan_list = [
        pan(i.value,
            mode='value',
            start_time=i.start_time,
            channel=i.channel,
            track=i.track) for i in current_chord.other_messages
        if i.type == 'control_change' and i.control == 10
    ]
    volume_list = [
        volume(i.value,
               mode='value',
               start_time=i.start_time,
               channel=i.channel,
               track=i.track) for i in current_chord.other_messages
        if i.type == 'control_change' and i.control == 7
    ]
    track_names_msg = [[
        i for i in current_chord.other_messages
        if i.type == 'track_name' and i.track == j
    ] for j in range(channels_num)]
    track_names_msg = [i[0] for i in track_names_msg if i]
    if not track_names_msg:
        track_names_list = []
    else:
        if not has_track_num and all(
                hasattr(i, 'channel') for i in track_names_msg):
            track_names_channels = [i.channel for i in track_names_msg]
            current_track_names = [
                track_names_msg[track_names_channels.index(i)]
                for i in channels_list
            ]
        else:
            current_track_names = track_names_msg
        track_names_list = [i.name for i in current_track_names]
    if (not track_names_list) or (len(track_names_list) != channels_num):
        track_names_list = [f'Channel {i+1}' for i in range(channels_num)]
    if not has_track_num:
        for each in current_chord.notes:
            each.track_num = channels_list.index(
                each.channel) if each.channel is not None else 0
        for each in current_chord.pitch_bends:
            if each.track != each.track_num:
                each.track = each.track_num
        for each in current_chord.other_messages:
            if hasattr(each, 'channel') and each.channel is not None:
                each.track = channels_list.index(each.channel)
        for each in pan_list:
            if each.channel is not None:
                each.track = channels_list.index(each.channel)
        for each in volume_list:
            if each.channel is not None:
                each.track = channels_list.index(each.channel)
    result_piece = piece(tracks=[chord([]) for i in range(channels_num)],
                         bpm=bpm,
                         pan=[[] for i in range(channels_num)],
                         volume=[[] for i in range(channels_num)])
    result_piece.reconstruct(current_chord,
                             current_start_time,
                             include_empty_track=True)
    if result_piece.channels is not None:
        if len(result_piece.channels) != channels_num:
            pan_list = [
                i for i in pan_list if i.channel in result_piece.channels
            ]
            volume_list = [
                i for i in volume_list if i.channel in result_piece.channels
            ]
            for each in pan_list:
                each.track = result_piece.channels.index(each.channel)
            for each in volume_list:
                each.track = result_piece.channels.index(each.channel)
            for k in range(len(result_piece.tracks)):
                for each in result_piece.tracks[k].pitch_bends:
                    each.track = k
            current_chord.other_messages = [
                i for i in current_chord.other_messages
                if not (hasattr(i, 'channel')
                        and i.channel not in result_piece.channels)
            ]
            for each in current_chord.other_messages:
                if hasattr(each, 'channel'):
                    each.track = result_piece.channels.index(each.channel)
    result_piece.other_messages = current_chord.other_messages
    for k in range(len(result_piece)):
        current_other_messages = [
            i for i in result_piece.other_messages if i.track == k
        ]
        result_piece.tracks[k].other_messages = current_other_messages
        current_pan = [i for i in pan_list if i.track == k]
        result_piece.pan[k] = current_pan
        current_volume = [i for i in volume_list if i.track == k]
        result_piece.volume[k] = current_volume
    result_piece.track_names = track_names_list
    result_piece.other_messages = concat(
        [i.other_messages for i in result_piece.tracks], start=[])
    instruments = []
    program_change_events = [
        i for i in result_piece.other_messages if i.type == 'program_change'
    ]
    for i, each in enumerate(result_piece.channels):
        current_program = None
        for j in program_change_events:
            if j.channel == each:
                current_program = j.program
                instruments.append(current_program + 1)
                break
        if current_program is None:
            instruments.append(1)
    result_piece.change_instruments(instruments)
    return result_piece


def modulation(current_chord, old_scale, new_scale, **args):
    '''
    change notes (including both of melody and chords) in the given piece
    of music from a given scale to another given scale, and return
    the new changing piece of music.
    '''
    return current_chord.modulation(old_scale, new_scale, **args)


def trans(obj,
          pitch=4,
          duration=1 / 4,
          interval=None,
          custom_mapping=None,
          pitch_interval=True):
    obj = obj.replace(' ', '')
    if is_valid_note(obj):
        return get_chord(obj,
                         'M',
                         pitch=pitch,
                         duration=duration,
                         intervals=interval,
                         custom_mapping=custom_mapping,
                         pitch_interval=pitch_interval)
    elif ':' in obj:
        current = obj.split(':')
        current[0] = to_note(current[0])
        return trans(f'{current[0].name}{current[1]}', current[0].num,
                     duration, interval)
    elif obj.count('/') > 1:
        current_parts = obj.split('/')
        current_parts = [int(i) if i.isdigit() else i for i in current_parts]
        result = trans(current_parts[0], pitch, duration, interval)
        for each in current_parts[1:]:
            if not isinstance(each, int):
                each = trans(each, pitch, duration, interval)
            result /= each
        return result
    elif '/' not in obj:
        check_structure = obj.split(',')
        check_structure_len = len(check_structure)
        if check_structure_len > 1:
            return trans(check_structure[0], pitch)(','.join(
                check_structure[1:])) % (duration, interval)
        current_chord_types = chordTypes if custom_mapping is None else custom_mapping
        N = len(obj)
        if N == 2:
            first = obj[0]
            types = obj[1]
            if is_valid_note(first) and types in current_chord_types:
                return get_chord(first,
                                 types,
                                 pitch=pitch,
                                 duration=duration,
                                 intervals=interval,
                                 custom_mapping=custom_mapping,
                                 pitch_interval=pitch_interval)
        elif N > 2:
            current_root_name = ''
            for i, each in enumerate(obj):
                if is_valid_note(current_root_name + each):
                    current_root_name += each
                else:
                    type1 = obj[i:]
                    break
            if current_root_name and type1 in current_chord_types:
                return get_chord(current_root_name,
                                 type1,
                                 pitch=pitch,
                                 duration=duration,
                                 intervals=interval,
                                 custom_mapping=custom_mapping,
                                 pitch_interval=pitch_interval)
    else:
        parts = obj.split('/')
        part1, part2 = parts[0], '/'.join(parts[1:])
        first_chord = trans(part1, pitch)
        if isinstance(first_chord, chord):
            if part2.isdigit() or (part2[0] == '-' and part2[1:].isdigit()):
                return (first_chord / int(part2)) % (duration, interval)
            elif part2[-1] == '!' and part2[:-1].isdigit():
                return (first_chord @ int(part2[:-1])) % (duration, interval)
            elif is_valid_note(part2):
                first_chord_temp.names = first_chord.names()
                if part2 in first_chord_temp.names and part2 != first_chord_temp.names[
                        0]:
                    return (first_chord.inversion(
                        first_chord_temp.names.index(part2))) % (duration,
                                                                interval)
                return chord([part2] + first_chord_temp.names,
                             rootpitch=pitch,
                             duration=duration,
                             interval=interval)
            else:
                second_chord = trans(part2, pitch)
                if isinstance(second_chord, chord):
                    return chord(second_chord.names() + first_chord.names(),
                                 rootpitch=pitch,
                                 duration=duration,
                                 interval=interval)
    raise ValueError(
        f'{obj} is not a valid chord representation or chord types not in database'
    )


def to_scale(obj, pitch=None):
    tonic, scale_name = obj.strip(' ').split(' ', 1)
    tonic = N(tonic)
    if pitch is not None:
        tonic.num = pitch
    scale_name = scale_name.strip(' ')
    return scale(tonic, scale_name)


def intervalof(current_chord, cumulative=True, translate=False):
    if isinstance(current_chord, scale):
        current_chord = current_chord.get_scale()
    if not isinstance(current_chord, chord):
        current_chord = chord(current_chord)
    return current_chord.intervalof(cumulative, translate)


def sums(*chordls):
    if len(chordls) == 1:
        chordls = chordls[0]
        start = chordls[0]
        for i in chordls[1:]:
            start += i
        return start
    else:
        return sums(list(chordls))


def build(*tracks_list, **kwargs):
    if len(tracks_list) == 1 and isinstance(tracks_list[0], list):
        current_tracks_list = tracks_list[0]
        if current_tracks_list and isinstance(current_tracks_list[0],
                                              (list, track)):
            return build(*tracks_list[0], **kwargs)
    remain_list = [1, 0, None, None, [], [], None]
    tracks = []
    instruments = []
    start_times = []
    channels = []
    track_names = []
    pan_msg = []
    volume_msg = []
    daw_channels = []
    other_messages = []
    result = P(tracks=tracks,
               instruments=instruments,
               start_times=start_times,
               track_names=track_names,
               channels=channels,
               pan=pan_msg,
               volume=volume_msg,
               daw_channels=daw_channels,
               other_messages=other_messages)
    for each in tracks_list:
        if isinstance(each, track):
            result.append(each)
        else:
            new_each = each + remain_list[len(each) - 1:]
            each = track(content=new_each[0],
                         instrument=new_each[1],
                         start_time=new_each[2],
                         channel=new_each[3],
                         track_name=new_each[4],
                         pan=new_each[5],
                         volume=new_each[6],
                         daw_channel=new_each[7])
            result.append(each)

    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


def translate(pattern,
              default_duration=1 / 8,
              default_interval=0,
              default_volume=100,
              start_time=None,
              mapping=drum_mapping):
    result = drum(pattern,
                  mapping=mapping,
                  default_duration=default_duration,
                  start_time=start_time,
                  default_interval=default_interval,
                  default_volume=default_volume,
                  translate_mode=1).notes
    return result


def chord_progression(chords,
                      durations=1 / 4,
                      intervals=0,
                      volumes=None,
                      chords_interval=None,
                      merge=True,
                      scale=None,
                      separator=','):
    if scale:
        return scale.chord_progression(chords, durations, intervals, volumes,
                                       chords_interval, merge)
    if isinstance(chords, str):
        if ' ' not in separator:
            chords = chords.replace(' ', '')
        chords = chords.split(separator)
    chords = [(i, ) if isinstance(i, str) else i for i in chords]
    chords_len = len(chords)
    if not isinstance(durations, list):
        durations = [durations for i in range(chords_len)]
    if not isinstance(intervals, list):
        intervals = [intervals for i in range(chords_len)]
    if volumes and not isinstance(volumes, list):
        volumes = [volumes for i in range(chords_len)]
    if chords_interval and not isinstance(chords_interval, list):
        chords_interval = [chords_interval for i in range(chords_len)]
    chords = [C(*i) if isinstance(i, tuple) else i for i in chords]
    for i in range(chords_len):
        chords[i] %= (durations[i], intervals[i],
                      volumes[i] if volumes else volumes)
    if merge:
        result = chords[0]
        current_interval = 0
        for i in range(1, chords_len):
            if chords_interval:
                current_interval += chords_interval[i - 1]
                result = result & (chords[i], current_interval)
            else:
                result |= chords[i]
        return result
    else:
        return chords


def arpeggio(chord_type,
             start=3,
             stop=7,
             durations=1 / 4,
             intervals=1 / 32,
             first_half=True,
             second_half=False):
    if isinstance(chord_type, str):
        rule = lambda chord_type, start, stop: concat([
            C(chord_type, i) % (durations, intervals)
            for i in range(start, stop)
        ], '|')
    else:
        rule = lambda chord_type, start, stop: concat([
            chord_type.reset_octave(i) % (durations, intervals)
            for i in range(start, stop)
        ], '|')
    result = chord([])
    first_half_part = rule(chord_type, start, stop)
    second_half_part = ~first_half_part[:-1]
    if first_half:
        result += first_half_part
    if second_half:
        result += second_half_part
    return result


def distribute(current_chord,
               length=1 / 4,
               start=0,
               stop=None,
               method=translate,
               mode=0):
    if isinstance(current_chord, str):
        current_chord = method(current_chord)
    elif isinstance(current_chord, list):
        current_chord = chord(current_chord)
    if stop is None:
        stop = len(current_chord)
    temp = copy(current_chord)
    part = temp.notes[start:stop]
    intervals = temp.interval[start:stop]
    durations = [i.duration for i in part]
    whole_duration = sum(durations)
    whole_interval = sum(intervals)
    durations = [length * (i / whole_duration) for i in durations]
    if whole_interval != 0:
        intervals = [length * (i / whole_interval) for i in intervals]
    else:
        intervals = [0 for i in intervals]
    if mode == 1:
        intervals = durations
    new_duration = temp.get_duration()
    new_duration[start:stop] = durations
    new_interval = temp.interval
    new_interval[start:stop] = intervals
    temp %= (new_duration, new_interval)
    return temp


def get_chords_from_rhythm(chords, current_rhythm, set_duration=True):
    if isinstance(chords, note):
        chords = chord(
            [copy(chords) for i in range(current_rhythm.get_beat_num())])
        return chords.apply_rhythm(current_rhythm, set_duration=set_duration)
    if isinstance(chords, chord):
        chords = [copy(chords) for i in range(current_rhythm.get_beat_num())]
    else:
        chords = copy(chords)
    length = len(chords)
    counter = -1
    has_beat = False
    current_start_time = 0
    chord_intervals = [0 for i in range(len(chords))]
    for i, each in enumerate(current_rhythm):
        current_duration = each.get_duration()
        if type(each) is beat:
            counter += 1
            if counter >= length:
                break
            current_chord = chords[counter]
            if set_duration:
                if current_duration != 0:
                    for k in current_chord:
                        k.duration = current_duration
            chord_intervals[counter] += current_duration
            has_beat = True
        elif type(each) is rest_symbol:
            if not has_beat:
                current_start_time += current_duration
            else:
                chord_intervals[counter] += current_duration
        elif type(each) is continue_symbol:
            if not has_beat:
                current_start_time += current_duration
            else:
                current_chord = chords[counter]
                for k in current_chord:
                    k.duration += current_duration
                chord_intervals[counter] += current_duration
    result = chords[0]
    current_interval = 0
    for i, each in enumerate(chords[1:]):
        current_interval += chord_intervals[i]
        result = result & (each, current_interval)
    extra_interval = chord_intervals[len(chords) - 1]
    result.interval[-1] = extra_interval
    result.start_time = current_start_time
    return result


@method_wrapper(chord)
def analyze_rhythm(current_chord,
                   include_continue=True,
                   total_length=None,
                   remove_empty_beats=False,
                   unit=None,
                   find_unit_ignore_duration=False,
                   merge_continue=True):
    if all(i <= 0 for i in current_chord.interval):
        return rhythm([beat(0) for i in range(len(current_chord))])
    if unit is None:
        current_interval = copy(current_chord.interval)
        if not find_unit_ignore_duration:
            current_interval += [
                current_chord.interval[i] - current_chord[i].duration
                for i in range(len(current_chord))
            ]
        current_interval = [i for i in current_interval if i > 0]
        unit = min(current_interval)
    beat_list = []
    if current_chord.start_time > 0:
        beat_list.extend([
            rest_symbol(unit)
            for i in range(int(current_chord.start_time // unit))
        ])
    for i, each in enumerate(current_chord.interval):
        if each == 0:
            beat_list.append(beat(0))
        else:
            current_beat = beat(unit)
            remain_interval = each - unit
            rest_num, extra_beat = divmod(remain_interval, unit)
            if extra_beat > 0:
                current_dotted_num = int(
                    math.log(1 / (1 -
                                  (((extra_beat + unit) / unit) / 2)), 2)) - 1
                current_beat.dotted = current_dotted_num
            beat_list.append(current_beat)
            if not include_continue:
                beat_list.extend(
                    [rest_symbol(unit) for k in range(int(rest_num))])
            else:
                current_duration = current_chord.notes[i].duration
                if current_duration >= each:
                    if not merge_continue:
                        beat_list.extend([
                            continue_symbol(unit) for k in range(int(rest_num))
                        ])
                    else:
                        beat_list[-1].duration += unit * int(rest_num)
                else:
                    current_rest_duration = each - current_duration
                    rest_num = current_rest_duration // unit
                    current_continue_duration = current_duration - unit
                    continue_num = current_continue_duration // unit
                    if not merge_continue:
                        beat_list.extend([
                            continue_symbol(unit)
                            for k in range(int(continue_num))
                        ])
                    else:
                        beat_list[-1].duration += unit * int(continue_num)
                    beat_list.extend(
                        [rest_symbol(unit) for k in range(int(rest_num))])
    result = rhythm(beat_list)
    if total_length is not None:
        current_time_signature = Fraction(result.get_total_duration() /
                                          total_length).limit_denominator()
        if current_time_signature == 1:
            current_time_signature = [4, 4]
        else:
            current_time_signature = [
                current_time_signature.numerator,
                current_time_signature.denominator
            ]
        result.time_signature = current_time_signature
    if remove_empty_beats:
        result = rhythm([i for i in result if i.duration != 0],
                        time_signature=result.time_signature)
    return result

def dotted(duration, num=1):
    if num == 0:
        return duration
    else:
        result = duration * sum([(1 / 2)**i for i in range(num + 1)])
        return result


def parse_dotted(text, get_fraction=False):
    length = len(text)
    dotted_num = 0
    ind = 0
    for i in range(length - 1, -1, -1):
        if text[i] != '.':
            ind = i
            break
        else:
            dotted_num += 1
    duration = parse_num(text[:ind + 1], get_fraction=get_fraction)
    current_duration = beat(duration, dotted_num).get_duration()
    return current_duration


def parse_num(duration, get_fraction=False):
    if '/' in duration:
        numerator, denominator = duration.split('/')
        numerator = int(numerator) if numerator.isdigit() else float(numerator)
        denominator = int(denominator) if denominator.isdigit() else float(
            denominator)
        if get_fraction:
            if not (isinstance(numerator, int)
                    and isinstance(denominator, int)):
                duration = Fraction(numerator /
                                    denominator).limit_denominator()
            else:
                duration = Fraction(numerator, denominator)
        else:
            duration = numerator / denominator
    else:
        duration = int(duration) if duration.isdigit() else float(duration)
        if get_fraction:
            duration = Fraction(duration).limit_denominator()
    return duration


def relative_note(a, b):
    '''
    return the notation of note a from note b with accidentals
    (how note b adds accidentals to match the same pitch as note a),
    works for the accidentals including sharp, flat, natural,
    double sharp, double flat
    (a, b are strings that represents a note, could be with accidentals)
    '''
    len_a, len_b = len(a), len(b)
    a_name, b_name, accidental_a, accidental_b = a[0], b[0], a[1:], b[1:]
    if len_a == 1 and len_b > 1 and a_name == b_name:
        return a + '♮'
    if a in standard:
        a = note(a, 5)
    else:
        a = note(a_name, 5)
        a_distinct = list(set(accidental_a))
        if len(a_distinct) == 1 and a_distinct[0] == 'b':
            a = a.flat(len(accidental_a))
        elif len(a_distinct) == 1 and a_distinct[0] == '#':
            a = a.sharp(len(accidental_a))
        elif accidental_a == 'x':
            a = a.sharp(2)
        elif accidental_a == '♮':
            pass
        else:
            raise ValueError(f'unrecognizable accidentals {accidental_a}')
    if b in standard:
        b = note(b, 5)
    else:
        b = note(b_name, 5)
        b_distinct = list(set(accidental_b))
        if len(b_distinct) == 1 and b_distinct[0] == 'b':
            b = b.flat(len(accidental_b))
        elif len(b_distinct) == 1 and b_distinct[0] == '#':
            b = b.sharp(len(accidental_b))
        elif accidental_b == 'x':
            b = b.sharp(2)
        elif accidental_b == '♮':
            pass
        else:
            raise ValueError(f'unrecognizable accidentals {accidental_b}')
    degree1, degree2 = a.degree, b.degree
    diff1, diff2 = degree1 - degree2, (degree1 - degree2 -
                                       12 if degree1 >= degree2 else degree1 +
                                       12 - degree2)
    if abs(diff1) < abs(diff2):
        diff = diff1
    else:
        diff = diff2
    if diff == 0:
        return b.name
    elif diff == 1:
        return b.name + '#'
    elif diff == 2:
        return b.name + 'x'
    elif diff > 2:
        return b.name + '#' * diff
    elif diff == -1:
        return b.name + 'b'
    elif diff == -2:
        return b.name + 'bb'
    elif diff < -2:
        return b.name + 'b' * abs(diff)


def get_note_name(current_note):
    return current_note[0]


def get_note_num(current_note):
    result = ''.join([i for i in current_note if i.isdigit()])
    return int(result) if result else None


def get_accidental(current_note):
    accidental_part = ''.join([i for i in current_note[1:] if not i.isdigit()])
    if accidental_part.endswith('b'):
        result = accidental_part[accidental_part.index('b'):]
    elif accidental_part.endswith('#'):
        result = accidental_part[accidental_part.index('#'):]
    elif accidental_part.endswith('x'):
        result = 'x'
    elif accidental_part.endswith('♮'):
        result = '♮'
    else:
        result = ''
    return result


def standardize_note(current_note):
    if current_note in standard2:
        return current_note
    elif current_note in standard_dict:
        return standard_dict[current_note]
    else:
        current_note_name = get_note_name(current_note)
        current_accidentals = get_accidental(current_note)
        if not current_accidentals:
            raise ValueError(f'Invalid note name or accidental {current_note}')
        else:
            diff = 0
            for each in current_accidentals:
                if each == '#':
                    diff += 1
                elif each == 'b':
                    diff -= 1
                elif each == 'x':
                    diff += 2
            result = (N(current_note_name) + diff).name
            return result


def is_valid_note(current_note):
    return len(
        current_note) > 0 and current_note[0] in standard and all(
            i in accidentals for i in current_note[1:])


def get_pitch_interval(note1, note2):
    if not isinstance(note1, note):
        note1 = N(note1)
    if not isinstance(note2, note):
        note2 = N(note2)
    direction = 1
    if note1.degree > note2.degree:
        note1, note2 = note2, note1
        direction = -1
    name1 = note1.base_name.upper()
    name2 = note2.base_name.upper()
    pitch_names = standard_pitch_name
    number = ((pitch_names.index(name2) - pitch_names.index(name1)) %
              len(pitch_names)) + 1
    if note2.num > note1.num:
        number += 7 * ((note2.degree - note1.degree) // octave)
    degree_diff = note2.degree - note1.degree
    max_pitch_interval = max(interval_dict.values())
    if number == 1 and degree_diff == 11:
        number = 8
    elif number > max(interval_number_dict):
        down_octave, degree_diff = divmod(degree_diff, octave)
        number -= down_octave * 7
    found = False
    current_number_interval = [
        i for i in interval_dict.values() if i.number == number
    ]
    current_number_interval_values = [i.value for i in current_number_interval]
    if degree_diff in current_number_interval_values:
        result_interval = current_number_interval[
            current_number_interval_values.index(degree_diff)]
        result = Interval(number=result_interval.number,
                                   quality=result_interval.quality,
                                   name=result_interval.name,
                                   direction=direction)
        return result
    else:
        min_interval = min(current_number_interval_values)
        max_interval = max(current_number_interval_values)
        if degree_diff < min_interval:
            result = min(current_number_interval,
                         key=lambda s: s.value).flat(min_interval -
                                                     degree_diff)
        elif degree_diff > max_interval:
            result = max(current_number_interval,
                         key=lambda s: s.value).sharp(degree_diff -
                                                      max_interval)
        else:
            raise ValueError(
                f'cannot find pitch interval for {note1} and {note2}')
        return result


def reset(self, **kwargs):
    temp = copy(self)
    for i, j in kwargs.items():
        setattr(temp, i, j)
    return temp


def closest_note(note1, note2, get_distance=False):
    if isinstance(note1, note):
        note1 = note1.name
    if not isinstance(note2, note):
        note2 = to_note(note2)
    current_note = [
        note(note1, note2.num),
        note(note1, note2.num - 1),
        note(note1, note2.num + 1)
    ]
    if not get_distance:
        result = min(current_note, key=lambda s: abs(s.degree - note2.degree))
        return result
    else:
        distances = [[i, abs(i.degree - note2.degree)] for i in current_note]
        distances.sort(key=lambda s: s[1])
        result = distances[0]
        return result


def closest_note_from_chord(note1, chord1, mode=0, get_distance=False):
    if not isinstance(note1, note):
        note1 = to_note(note1)
    if isinstance(chord1, chord):
        chord1 = chord1.notes
    current_name = standard_dict.get(note1.name, note1.name)
    distances = [(closest_note(note1, each, get_distance=True), i)
                 for i, each in enumerate(chord1)]
    distances.sort(key=lambda s: s[0][1])
    result = chord1[distances[0][1]]
    if mode == 1:
        result = distances[0][0][0]
    if get_distance:
        result = (result, distances[0][0][1])
    return result


def note_range(note1, note2):
    current_range = list(range(note1.degree, note2.degree))
    result = [degree_to_note(i) for i in current_range]
    return result


def adjust_to_scale(current_chord, current_scale):
    temp = copy(current_chord)
    current_notes = current_scale.get_scale()
    for each in temp:
        current_note = closest_note_from_chord(each, current_notes)
        each.name = current_note.name
        each.num = current_note.num
    return temp


def dataclass_repr(s, keywords=None):
    if not keywords:
        result = f"{type(s).__name__}({', '.join([f'{i}={j}' for i, j in vars(s).items()])})"
    else:
        result = f"{type(s).__name__}({', '.join([f'{i}={vars(s)[i]}' for i in keywords])})"
    return result


def to_dict(current_chord,
            bpm=120,
            channel=0,
            start_time=None,
            instrument=None,
            i=None):
    if i is not None:
        instrument = i
    if isinstance(current_chord, note):
        current_chord = chord([current_chord])
    elif isinstance(current_chord, list):
        current_chord = concat(current_chord, '|')
    if isinstance(current_chord, chord):
        if instrument is None:
            instrument = 1
        current_chord = P(
            tracks=[current_chord],
            instruments=[instrument],
            bpm=bpm,
            channels=[channel],
            start_times=[
                current_chord.start_time if start_time is None else start_time
            ],
            other_messages=current_chord.other_messages)
    elif isinstance(current_chord, track):
        current_chord = build(current_chord, bpm=current_chord.bpm)
    elif isinstance(current_chord, drum):
        current_chord = P(tracks=[current_chord.notes],
                          instruments=[current_chord.instrument],
                          bpm=bpm,
                          start_times=[
                              current_chord.notes.start_time
                              if start_time is None else start_time
                          ],
                          channels=[9])
    else:
        current_chord = copy(current_chord)
    result = current_chord.__dict__
    result['tracks'] = [i.__dict__ for i in result['tracks']]
    for each_track in result['tracks']:
        for i, each in enumerate(each_track['notes']):
            each_track['notes'][i] = {
                k1: k2
                for k1, k2 in each.__dict__.items() if k1 not in ['degree']
            }
        for i, each in enumerate(each_track['pitch_bends']):
            each_track['pitch_bends'][i] = {
                k1: k2
                for k1, k2 in each.__dict__.items()
                if k1 not in ['degree', 'volume', 'duration']
            }
            each_track['pitch_bends'][i]['mode'] = 'value'
        for i, each in enumerate(each_track['tempos']):
            each_track['tempos'][i] = {
                k1: k2
                for k1, k2 in each.__dict__.items()
                if k1 not in ['degree', 'volume', 'duration']
            }
        each_track['other_messages'] = [
            k.__dict__ for k in each_track['other_messages']
        ]
        for i, each in enumerate(each_track['notes']):
            each['interval'] = each_track['interval'][i]
        del each_track['interval']
    result['other_messages'] = [i.__dict__ for i in result['other_messages']]
    result['pan'] = [[i.__dict__ for i in j] for j in result['pan']]
    result['volume'] = [[i.__dict__ for i in j] for j in result['volume']]
    for i in result['pan']:
        for j in i:
            j['mode'] = 'value'
            del j['value_percentage']
    for i in result['volume']:
        for j in i:
            j['mode'] = 'value'
            del j['value_percentage']
    return result

def bar_to_real_time(bar, bpm, mode=0):
    # convert bar to time in ms
    return int(
        (60000 / bpm) * (bar * 4)) if mode == 0 else (60000 / bpm) * (bar * 4)


def real_time_to_bar(time, bpm):
    # convert time in ms to bar
    return (time / (60000 / bpm)) / 4


C = trans
N = to_note
S = to_scale
P = piece
arp = arpeggio

for each in [
        note, chord, piece, track, scale, drum, rest, tempo, pitch_bend, pan,
        volume, event, beat, rest_symbol, continue_symbol, rhythm
]:
    each.reset = reset
    each.__hash__ = lambda self: hash(repr(self))
    each.copy = lambda self: copy(self)
    if each.__eq__ == object.__eq__:
        each.__eq__ = lambda self, other: type(other) is type(
            self) and self.__dict__ == other.__dict__

#=================================================================================================
# algorithms.py
#=================================================================================================

def inversion_from(a, b, num=False):
    N = len(b)
    for i in range(1, N):
        temp = b.inversion(i)
        if temp.names(standardize_note=True) == a.names(standardize_note=True):
            return a[0].name if not num else i


def sort_from(a, b):
    a_names = a.names(standardize_note=True)
    b_names = b.names(standardize_note=True)
    order = [b_names.index(j) + 1 for j in a_names]
    return order


def omit_from(a, b):
    a_notes = a.names(standardize_note=True)
    b_notes = b.names(standardize_note=True)
    omitnotes = [i for i in b_notes if i not in a_notes]
    b_first_note = b[0]
    omitnotes_degree = []
    for j in omitnotes:
        current_degree = get_pitch_interval(b_first_note, b[b_notes.index(j)])
        precise_degrees = list(reverse_precise_degree_match.keys())
        if current_degree not in precise_degrees:
            omitnotes_degree.append(j)
        else:
            current_precise_degree = precise_degrees[precise_degrees.index(
                current_degree)]
            omitnotes_degree.append(
                reverse_precise_degree_match[current_precise_degree])
    omitnotes = omitnotes_degree
    return omitnotes


def change_from(a, b, octave_a=False, octave_b=False, same_degree=True):
    '''
    how a is changed from b (flat or sharp some notes of b to get a)
    this is used only when two chords have the same number of notes
    in the detect chord function
    '''
    if octave_a:
        a = a.inoctave()
    if octave_b:
        b = b.inoctave()
    if same_degree:
        b = b.down(12 * (b[0].num - a[0].num))
    N = min(len(a), len(b))
    anotes = [x.degree for x in a.notes]
    bnotes = [x.degree for x in b.notes]
    anames = a.names(standardize_note=True)
    bnames = b.names(standardize_note=True)
    M = min(len(anotes), len(bnotes))
    changes = [(bnames[i], bnotes[i] - anotes[i]) for i in range(M)]
    changes = [x for x in changes if x[1] != 0]
    if any(abs(j[1]) != 1 for j in changes):
        changes = []
    else:
        b_first_note = b[0].degree
        for i, each in enumerate(changes):
            note_name, note_change = each
            if note_change != 0:
                b_root_note = b.notes[0]
                b_current_note = b.notes[bnames.index(note_name)]
                current_b_degree = get_pitch_interval(b_root_note,
                                                      b_current_note)
                precise_degrees = list(
                    reverse_precise_degree_match.keys())
                if current_b_degree not in precise_degrees:
                    current_degree = b_current_note.name
                else:
                    current_precise_degree = precise_degrees[
                        precise_degrees.index(current_b_degree)]
                    current_degree = reverse_precise_degree_match[
                        current_precise_degree]
                if note_change > 0:
                    changes[i] = f'b{current_degree}'
                else:
                    changes[i] = f'#{current_degree}'
    return changes


def contains(a, b):
    '''
    if b contains a (notes), in other words,
    all of a's notes is inside b's notes
    '''
    return set(a.names(standardize_note=True)) < set(
        b.names(standardize_note=True)) and len(a) < len(b)


def inversion_way(a, b):
    if samenotes(a, b):
        result = None
    elif samenote_set(a, b):
        inversion_msg = inversion_from(a, b, num=True)
        if inversion_msg is not None:
            result = inversion_msg
        else:
            sort_msg = sort_from(a, b)
            result = sort_msg
    else:
        result = None
    return result


def samenotes(a, b):
    return a.names(standardize_note=True) == b.names(standardize_note=True)


def samenote_set(a, b):
    return set(a.names(standardize_note=True)) == set(
        b.names(standardize_note=True))


def find_similarity(a,
                    b=None,
                    b_type=None,
                    change_from_first=True,
                    same_note_special=False,
                    similarity_ratio=0.6,
                    custom_mapping=None):
    current_chord_type = chord_type()
    current_chord_type.order = []
    if b is None:
        current_chord_types = chordTypes if custom_mapping is None else custom_mapping[
            2]
        wholeTypes = current_chord_types.keynames()
        self.name = a.names(standardize_note=True)
        root_note = a[0]
        root_note_standardize = note(standardize_note(root_note.name),
                                     root_note.num)
        possible_chords = [(get_chord(root_note_standardize,
                                      each,
                                      custom_mapping=current_chord_types,
                                      pitch_interval=False), each, i)
                           for i, each in enumerate(wholeTypes)]
        lengths = len(possible_chords)
        if same_note_special:
            ratios = [(1 if samenote_set(a, x[0]) else SequenceMatcher(
                None, self.name, x[0].names()).ratio(), x[1], x[2])
                      for x in possible_chords]
        else:
            ratios = [(SequenceMatcher(None, self.name,
                                       x[0].names()).ratio(), x[1], x[2])
                      for x in possible_chords]
        alen = len(a)
        ratios_temp = [
            ratios[k] for k in range(len(ratios))
            if len(possible_chords[k][0]) >= alen
        ]
        if len(ratios_temp) != 0:
            ratios = ratios_temp
        ratios.sort(key=lambda x: x[0], reverse=True)
        first = ratios[0]
        highest = first[0]
        chordfrom = get_chord(root_note,
                              wholeTypes[first[2]],
                              custom_mapping=current_chord_types)
        current_chord_type.highest_ratio = highest
        if highest >= similarity_ratio:
            if change_from_first:
                current_chord_type = find_similarity(
                    a=a,
                    b=chordfrom,
                    b_type=first[1],
                    similarity_ratio=similarity_ratio,
                    custom_mapping=custom_mapping)
                current_chord_type.highest_ratio = highest
                cff_ind = 0
                while current_chord_type.chord_type is None:
                    cff_ind += 1
                    try:
                        first = ratios[cff_ind]
                    except:
                        first = ratios[0]
                        highest = first[0]
                        chordfrom = get_chord(
                            root_note,
                            wholeTypes[first[2]],
                            custom_mapping=current_chord_types)
                        current_chord_type.chord_type = None
                        break
                    highest = first[0]
                    chordfrom = get_chord(root_note,
                                          wholeTypes[first[2]],
                                          custom_mapping=current_chord_types)
                    if highest >= similarity_ratio:
                        current_chord_type = find_similarity(
                            a=a,
                            b=chordfrom,
                            b_type=first[1],
                            similarity_ratio=similarity_ratio,
                            custom_mapping=custom_mapping)
                        current_chord_type.highest_ratio = highest
                    else:
                        first = ratios[0]
                        highest = first[0]
                        chordfrom = get_chord(
                            root_note,
                            wholeTypes[first[2]],
                            custom_mapping=current_chord_types)
                        current_chord_type.chord_type = None
                        break
            if not change_from_first:
                chordfrom_type = first[1]
                current_chord_type = find_similarity(
                    a=a,
                    b=chordfrom,
                    b_type=chordfrom_type,
                    similarity_ratio=similarity_ratio,
                    custom_mapping=custom_mapping)
                current_chord_type.highest_ratio = highest
            return current_chord_type
        else:
            return current_chord_type
    else:
        if b_type is None:
            raise ValueError('requires chord type name of b')
        chordfrom_type = b_type
        chordfrom = b

        if samenotes(a, chordfrom):
            chordfrom_type = detect(current_chord=chordfrom,
                                    change_from_first=change_from_first,
                                    same_note_special=same_note_special,
                                    get_chord_type=True,
                                    custom_mapping=custom_mapping)
            return chordfrom_type

        elif samenote_set(a, chordfrom):
            current_chord_type.root = chordfrom[0].name
            current_chord_type.chord_type = chordfrom_type
            current_inv_msg = inversion_way(a, chordfrom)
            current_chord_type.apply_sort_msg(current_inv_msg,
                                              change_order=True)
        elif contains(a, chordfrom):
            current_omit_msg = omit_from(a, chordfrom)
            current_chord_type.chord_speciality = 'root position'
            current_chord_type.omit = current_omit_msg
            current_chord_type.root = chordfrom[0].name
            current_chord_type.chord_type = chordfrom_type
            current_chord_type._add_order(0)
            current_custom_chord_types = custom_mapping[
                2] if custom_mapping is not None else None
            current_chord_omit = current_chord_type.to_chord(
                custom_mapping=current_custom_chord_types)
            if not samenotes(a, current_chord_omit):
                current_inv_msg = inversion_way(a, current_chord_omit)
                current_chord_type.apply_sort_msg(current_inv_msg,
                                                  change_order=True)
        elif len(a) == len(chordfrom):
            current_change_msg = change_from(a, chordfrom)
            if current_change_msg:
                current_chord_type.chord_speciality = 'altered chord'
                current_chord_type.altered = current_change_msg
                current_chord_type.root = chordfrom[0].name
                current_chord_type.chord_type = chordfrom_type
                current_chord_type._add_order(1)
        return current_chord_type


def detect_variation(current_chord,
                     change_from_first=True,
                     original_first=True,
                     same_note_special=False,
                     similarity_ratio=0.6,
                     N=None,
                     custom_mapping=None):
    current_custom_chord_types = custom_mapping[
        2] if custom_mapping is not None else None
    for each in range(1, N):
        each_current = current_chord.inversion(each)
        each_detect = detect(current_chord=each_current,
                             change_from_first=change_from_first,
                             original_first=original_first,
                             same_note_special=same_note_special,
                             similarity_ratio=similarity_ratio,
                             whole_detect=False,
                             get_chord_type=True,
                             custom_mapping=custom_mapping)
        if each_detect is not None:
            inv_msg = inversion_way(current_chord, each_current)
            if each_detect.voicing is not None:
                change_from_chord = each_detect.to_chord(
                    apply_voicing=False,
                    custom_mapping=current_custom_chord_types)
                inv_msg = inversion_way(current_chord, change_from_chord)
                if inv_msg is None:
                    result = find_similarity(a=current_chord,
                                             b=change_from_chord,
                                             b_type=each_detect.chord_type,
                                             similarity_ratio=similarity_ratio,
                                             custom_mapping=custom_mapping)
                else:
                    result = each_detect
                    result.apply_sort_msg(inv_msg, change_order=True)
            else:
                result = each_detect
                result.apply_sort_msg(inv_msg, change_order=True)
            return result
    for each2 in range(1, N):
        each_current = current_chord.inversion_highest(each2)
        each_detect = detect(current_chord=each_current,
                             change_from_first=change_from_first,
                             original_first=original_first,
                             same_note_special=same_note_special,
                             similarity_ratio=similarity_ratio,
                             whole_detect=False,
                             get_chord_type=True,
                             custom_mapping=custom_mapping)
        if each_detect is not None:
            inv_msg = inversion_way(current_chord, each_current)
            if each_detect.voicing is not None:
                change_from_chord = each_detect.to_chord(
                    apply_voicing=False,
                    custom_mapping=current_custom_chord_types)
                inv_msg = inversion_way(current_chord, change_from_chord)
                if inv_msg is None:
                    result = find_similarity(a=current_chord,
                                             b=change_from_chord,
                                             b_type=each_detect.chord_type,
                                             similarity_ratio=similarity_ratio,
                                             custom_mapping=custom_mapping)
                else:
                    result = each_detect
                    result.apply_sort_msg(inv_msg, change_order=True)
            else:
                result = each_detect
                result.apply_sort_msg(inv_msg, change_order=True)
            return result


def detect_split(current_chord, N=None, **detect_args):
    if N is None:
        N = len(current_chord)
    result = chord_type(chord_speciality='polychord')
    if N < 6:
        splitind = 1
        lower = chord_type(note_name=current_chord.notes[0].name, type='note')
        upper = detect(current_chord.notes[splitind:],
                       get_chord_type=True,
                       **detect_args)
        result.polychords = [lower, upper]
    else:
        splitind = N // 2
        lower = detect(current_chord.notes[:splitind],
                       get_chord_type=True,
                       **detect_args)
        upper = detect(current_chord.notes[splitind:],
                       get_chord_type=True,
                       **detect_args)
        result.polychords = [lower, upper]
    return result


def interval_check(current_chord, custom_mapping=None):
    times, dist = divmod(
        (current_chord.notes[1].degree - current_chord.notes[0].degree), 12)
    if times > 0:
        dist = 12 + dist
    current_interval_dict = INTERVAL if custom_mapping is None else custom_mapping[
        0]
    if dist in current_interval_dict:
        interval_name = current_interval_dict[dist]
    else:
        interval_name = current_interval_dict[dist % 12]
    root_note_name = current_chord[0].name
    return root_note_name, interval_name


def _detect_helper(current_chord_type,
                   get_chord_type=False,
                   show_degree=True,
                   custom_mapping=None):
    return current_chord_type.to_text(
        show_degree=show_degree,
        custom_mapping=custom_mapping if current_chord_type.type == 'chord'
        else None) if not get_chord_type else current_chord_type


@method_wrapper(chord)
def detect(current_chord,
           change_from_first=True,
           original_first=True,
           same_note_special=False,
           whole_detect=True,
           poly_chord_first=False,
           root_preference=False,
           show_degree=False,
           get_chord_type=False,
           original_first_ratio=0.86,
           similarity_ratio=0.6,
           custom_mapping=None,
           standardize_note=False):
    current_chord_type = chord_type()
    if not isinstance(current_chord, chord):
        current_chord = chord(current_chord)
    N = len(current_chord)
    if N == 1:
        current_chord_type.type = 'note'
        current_chord_type.note_name = str(current_chord.notes[0])
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)
    if N == 2:
        current_root_note_name, current_interval_name = interval_check(
            current_chord, custom_mapping=custom_mapping)
        current_chord_type.type = 'interval'
        current_chord_type.root = current_root_note_name
        current_chord_type.interval_name = current_interval_name
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)
    current_chord = current_chord.standardize(
        standardize_note=standardize_note)
    N = len(current_chord)
    if N == 1:
        current_chord_type.type = 'note'
        current_chord_type.note_name = str(current_chord.notes[0])
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)
    if N == 2:
        current_root_note_name, current_interval_name = interval_check(
            current_chord, custom_mapping=custom_mapping)
        current_chord_type.type = 'interval'
        current_chord_type.root = current_root_note_name
        current_chord_type.interval_name = current_interval_name
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)
    current_chord_type.order = []
    root = current_chord[0].degree
    root_note = current_chord[0].name
    distance = tuple(i.degree - root for i in current_chord[1:])
    current_detect_types = detectTypes if custom_mapping is None else custom_mapping[
        1]
    current_custom_chord_types = custom_mapping[
        2] if custom_mapping is not None else None
    if distance in current_detect_types:
        findTypes = current_detect_types[distance]
        current_chord_type.root = root_note
        current_chord_type.chord_type = findTypes[0]
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)

    if root_preference:
        current_chord_type_root_preference = detect_chord_by_root(
            current_chord,
            get_chord_type=True,
            custom_mapping=custom_mapping,
            inner=True)
        if current_chord_type_root_preference is not None:
            return _detect_helper(
                current_chord_type=current_chord_type_root_preference,
                get_chord_type=get_chord_type,
                show_degree=show_degree,
                custom_mapping=custom_mapping)

    current_chord_inoctave = current_chord.inoctave()
    root = current_chord_inoctave[0].degree
    distance = tuple(i.degree - root for i in current_chord_inoctave[1:])
    if distance in current_detect_types:
        result = current_detect_types[distance]
        current_chord_type.clear()
        current_invert_msg = inversion_way(current_chord,
                                           current_chord_inoctave)
        current_chord_type.root = current_chord_inoctave[0].name
        current_chord_type.chord_type = result[0]
        current_chord_type.apply_sort_msg(current_invert_msg,
                                          change_order=True)
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)

    current_chord_type = find_similarity(a=current_chord,
                                         change_from_first=change_from_first,
                                         same_note_special=same_note_special,
                                         similarity_ratio=similarity_ratio,
                                         custom_mapping=custom_mapping)

    if current_chord_type.chord_type is not None:
        if (original_first
                and current_chord_type.highest_ratio >= original_first_ratio
            ) or current_chord_type.highest_ratio == 1:
            return _detect_helper(current_chord_type=current_chord_type,
                                  get_chord_type=get_chord_type,
                                  show_degree=show_degree,
                                  custom_mapping=custom_mapping)

    current_chord_type_inoctave = find_similarity(
        a=current_chord_inoctave,
        change_from_first=change_from_first,
        same_note_special=same_note_special,
        similarity_ratio=similarity_ratio,
        custom_mapping=custom_mapping)

    if current_chord_type_inoctave.chord_type is not None:
        if (original_first and current_chord_type_inoctave.highest_ratio
                >= original_first_ratio
            ) or current_chord_type_inoctave.highest_ratio == 1:
            current_invert_msg = inversion_way(current_chord,
                                               current_chord_inoctave)
            current_chord_type_inoctave.apply_sort_msg(current_invert_msg,
                                                       change_order=True)
            return _detect_helper(
                current_chord_type=current_chord_type_inoctave,
                get_chord_type=get_chord_type,
                show_degree=show_degree,
                custom_mapping=custom_mapping)

    for i in range(1, N):
        current = chord(current_chord.inversion(i).names())
        distance = current.intervalof()
        if distance not in current_detect_types:
            current = current.inoctave()
            distance = current.intervalof()
        if distance in current_detect_types:
            result = current_detect_types[distance]
            inversion_result = inversion_way(current_chord, current)
            if not isinstance(inversion_result, int):
                continue
            else:
                current_chord_type.clear()
                current_chord_type.chord_speciality = 'inverted chord'
                current_chord_type.inversion = inversion_result
                current_chord_type.root = current[0].name
                current_chord_type.chord_type = result[0]
                current_chord_type._add_order(2)
                return _detect_helper(current_chord_type=current_chord_type,
                                      get_chord_type=get_chord_type,
                                      show_degree=show_degree,
                                      custom_mapping=custom_mapping)
    for i in range(1, N):
        current = chord(current_chord.inversion_highest(i).names())
        distance = current.intervalof()
        if distance not in current_detect_types:
            current = current.inoctave()
            distance = current.intervalof()
        if distance in current_detect_types:
            result = current_detect_types[distance]
            inversion_high_result = inversion_way(current_chord, current)
            if not isinstance(inversion_high_result, int):
                continue
            else:
                current_chord_type.clear()
                current_chord_type.chord_speciality = 'inverted chord'
                current_chord_type.inversion = inversion_high_result
                current_chord_type.root = current[0].name
                current_chord_type.chord_type = result[0]
                current_chord_type._add_order(2)
                return _detect_helper(current_chord_type=current_chord_type,
                                      get_chord_type=get_chord_type,
                                      show_degree=show_degree,
                                      custom_mapping=custom_mapping)
    if poly_chord_first and N > 3:
        current_chord_type = detect_split(current_chord=current_chord,
                                          N=N,
                                          change_from_first=change_from_first,
                                          original_first=original_first,
                                          same_note_special=same_note_special,
                                          whole_detect=whole_detect,
                                          poly_chord_first=poly_chord_first,
                                          show_degree=show_degree,
                                          custom_mapping=custom_mapping)
        return _detect_helper(current_chord_type=current_chord_type,
                              get_chord_type=get_chord_type,
                              show_degree=show_degree,
                              custom_mapping=custom_mapping)
    inversion_final = True
    possibles = [(find_similarity(a=current_chord.inversion(j),
                                  change_from_first=change_from_first,
                                  same_note_special=same_note_special,
                                  similarity_ratio=similarity_ratio,
                                  custom_mapping=custom_mapping), j)
                 for j in range(1, N)]
    possibles = [x for x in possibles if x[0].chord_type is not None]
    if len(possibles) == 0:
        possibles = [(find_similarity(a=current_chord.inversion_highest(j),
                                      change_from_first=change_from_first,
                                      same_note_special=same_note_special,
                                      similarity_ratio=similarity_ratio,
                                      custom_mapping=custom_mapping), j)
                     for j in range(1, N)]
        possibles = [x for x in possibles if x[0].chord_type is not None]
        inversion_final = False
    if len(possibles) == 0:
        if current_chord_type.chord_type is not None:
            return _detect_helper(current_chord_type=current_chord_type,
                                  get_chord_type=get_chord_type,
                                  show_degree=show_degree,
                                  custom_mapping=custom_mapping)
        else:
            if not whole_detect:
                return
    else:
        possibles.sort(key=lambda x: x[0].highest_ratio, reverse=True)
        highest_chord_type, current_inversion = possibles[0]
        if current_chord_type.chord_type is not None:
            if current_chord_type.highest_ratio >= similarity_ratio and (
                    current_chord_type.highest_ratio
                    >= highest_chord_type.highest_ratio
                    or highest_chord_type.voicing is not None):
                return _detect_helper(current_chord_type=current_chord_type,
                                      get_chord_type=get_chord_type,
                                      show_degree=show_degree,
                                      custom_mapping=custom_mapping)
        if highest_chord_type.highest_ratio >= similarity_ratio:
            if inversion_final:
                current_invert = current_chord.inversion(current_inversion)
            else:
                current_invert = current_chord.inversion_highest(
                    current_inversion)
            invfrom_current_invert = inversion_way(current_chord,
                                                   current_invert)
            if highest_chord_type.voicing is not None and not isinstance(
                    invfrom_current_invert, int):
                current_root_position = highest_chord_type.get_root_position()
                current_chord_type = find_similarity(
                    a=current_chord,
                    b=C(current_root_position,
                        custom_mapping=current_custom_chord_types),
                    b_type=highest_chord_type.chord_type,
                    similarity_ratio=similarity_ratio,
                    custom_mapping=custom_mapping)
                current_chord_type.chord_speciality = 'chord voicings'
                current_chord_type.voicing = invfrom_current_invert
                current_chord_type._add_order(3)
            else:
                current_invert_msg = inversion_way(
                    current_chord,
                    highest_chord_type.to_chord(
                        apply_voicing=False,
                        custom_mapping=current_custom_chord_types))
                current_chord_type = highest_chord_type
                current_chord_type.apply_sort_msg(current_invert_msg,
                                                  change_order=True)
            return _detect_helper(current_chord_type=current_chord_type,
                                  get_chord_type=get_chord_type,
                                  show_degree=show_degree,
                                  custom_mapping=custom_mapping)

    if not whole_detect:
        return
    else:
        detect_var = detect_variation(current_chord=current_chord,
                                      change_from_first=change_from_first,
                                      original_first=original_first,
                                      same_note_special=same_note_special,
                                      similarity_ratio=similarity_ratio,
                                      N=N,
                                      custom_mapping=custom_mapping)
        if detect_var is None:
            current_chord_type = detect_split(
                current_chord=current_chord,
                N=N,
                change_from_first=change_from_first,
                original_first=original_first,
                same_note_special=same_note_special,
                whole_detect=whole_detect,
                poly_chord_first=poly_chord_first,
                show_degree=show_degree,
                custom_mapping=custom_mapping)
            return _detect_helper(current_chord_type=current_chord_type,
                                  get_chord_type=get_chord_type,
                                  show_degree=show_degree,
                                  custom_mapping=custom_mapping)
        else:
            current_chord_type = detect_var
            return _detect_helper(current_chord_type=current_chord_type,
                                  get_chord_type=get_chord_type,
                                  show_degree=show_degree,
                                  custom_mapping=custom_mapping)


def detect_chord_by_root(current_chord,
                         get_chord_type=False,
                         show_degree=False,
                         custom_mapping=None,
                         return_mode=0,
                         inner=False,
                         standardize_note=False):
    if not inner:
        current_chord = current_chord.standardize(
            standardize_note=standardize_note)
        if len(current_chord) < 3:
            return detect(current_chord,
                          get_chord_type=get_chord_type,
                          custom_mapping=custom_mapping)
    current_chord_types = []
    current_custom_chord_types = custom_mapping[
        2] if custom_mapping is not None else None
    current_match_chord = _detect_chord_by_root_helper(
        current_chord, custom_mapping=custom_mapping, inner=inner)
    if current_match_chord:
        current_chord_type = find_similarity(
            a=current_chord,
            b=C(f'{current_chord[0].name}{current_match_chord}',
                custom_mapping=current_custom_chord_types),
            b_type=current_match_chord,
            custom_mapping=custom_mapping)
        current_chord_types.append(current_chord_type)
    current_chord_inoctave = current_chord.inoctave()
    if not samenotes(current_chord_inoctave, current_chord):
        current_match_chord_inoctave = _detect_chord_by_root_helper(
            current_chord_inoctave, custom_mapping=custom_mapping, inner=inner)
        if current_match_chord_inoctave and current_match_chord_inoctave != current_match_chord:
            current_chord_type_inoctave = find_similarity(
                a=current_chord,
                b=C(f'{current_chord[0].name}{current_match_chord_inoctave}',
                    custom_mapping=current_custom_chord_types),
                b_type=current_match_chord_inoctave,
                custom_mapping=custom_mapping)
            current_chord_types.append(current_chord_type_inoctave)
    if return_mode == 0:
        if current_chord_types:
            current_chord_types = min(current_chord_types,
                                      key=lambda s: s.get_complexity())
            return current_chord_types if get_chord_type else current_chord_types.to_text(
                show_degree=show_degree)
    else:
        return current_chord_types if get_chord_type else [
            i.to_text(show_degree=show_degree) for i in current_chord_types
        ]


def _detect_chord_by_root_helper(current_chord,
                                 custom_mapping=None,
                                 inner=False):
    current_match_chord = None
    current_note_interval = current_chord.intervalof(translate=True)
    current_note_interval.sort()
    current_note_interval = tuple(current_note_interval)
    current_detect_types = detectTypes if not custom_mapping else custom_mapping[
        1]
    current_chord_types = chordTypes if not custom_mapping else custom_mapping[
        2]
    if not inner and current_note_interval in current_detect_types:
        return current_detect_types[current_note_interval][0]
    if not any(i in current_note_interval
               for i in non_standard_intervals):
        chord_type_intervals = list(current_chord_types.values())
        match_chords = [(current_detect_types[i][0], i)
                        for i in chord_type_intervals
                        if all((each in i or each - octave in i)
                               for each in current_note_interval)]
        if match_chords:
            match_chords.sort(key=lambda s: len(s[1]))
            current_match_chord = match_chords[0][0]
    return current_match_chord


def detect_scale_type(current_scale, mode='scale'):
    if mode == 'scale':
        interval = tuple(current_scale.interval)
    elif mode == 'interval':
        interval = tuple(current_scale)
    if interval not in detectScale:
        if mode == 'scale':
            current_notes = current_scale.get_scale()
        elif mode == 'interval':
            current_notes = get_chord_by_interval('C',
                                                  current_scale,
                                                  cumulative=False)
        result = detect_in_scale(current_notes,
                                 get_scales=True,
                                 match_len=True)
        if not result:
            return None
        else:
            return result[0].mode
    else:
        scales = detectScale[interval]
        return scales[0]


def _random_composing_choose_melody(focused, now_focus, focus_ratio,
                                    focus_notes, remained_notes, pick,
                                    avoid_dim_5, chordinner, newchord,
                                    choose_from_chord):
    if focused:
        now_focus = random.choices([1, 0], [focus_ratio, 1 - focus_ratio])[0]
        if now_focus == 1:
            firstmelody = random.choice(focus_notes)
        else:
            firstmelody = random.choice(remained_notes)
    else:
        if choose_from_chord:
            current = random.randint(0, 1)
            if current == 0:
                # pick up melody notes outside chord inner notes
                firstmelody = random.choice(pick)
                # avoid to choose a melody note that appears a diminished fifth interval with the current chord
                if avoid_dim_5:
                    while any((firstmelody.degree - x.degree) %
                              diminished_fifth == 0
                              for x in newchord.notes):
                        firstmelody = random.choice(pick)
            else:
                # pick up melody notes from chord inner notes
                firstmelody = random.choice(chordinner)
        else:
            firstmelody = random.choice(pick)
            if avoid_dim_5:
                while any((firstmelody.degree - x.degree) %
                          diminished_fifth == 0
                          for x in newchord.notes):
                    firstmelody = random.choice(pick)
    return firstmelody


def random_composing(current_scale,
                     length,
                     pattern=None,
                     focus_notes=None,
                     focus_ratio=0.7,
                     avoid_dim_5=True,
                     num=3,
                     left_hand_velocity=70,
                     right_hand_velocity=80,
                     left_hand_meter=4,
                     choose_intervals=[1 / 8, 1 / 4, 1 / 2],
                     choose_durations=[1 / 8, 1 / 4, 1 / 2],
                     melody_interval_tol=perfect_fourth,
                     choose_from_chord=False):
    '''
    compose a piece of music randomly from a given scale with custom preferences to some degrees in the scale
    '''
    if pattern is not None:
        pattern = [int(x) for x in pattern]
    standard = current_scale.notes[:-1]
    # pick is the sets of notes from the required scales which used to pick up notes for melody
    pick = [x.up(2 * octave) for x in standard]
    focused = False
    if focus_notes is not None:
        focused = True
        focus_notes = [pick[i - 1] for i in focus_notes]
        remained_notes = [j for j in pick if j not in focus_notes]
        now_focus = 0
    else:
        focus_notes = None
        remained_notes = None
        now_focus = 0
    # the chord part and melody part will be written separately,
    # but still with some revelations. (for example, avoiding dissonant intervals)
    # the draft of the piece of music would be generated first,
    # and then modify the details of the music (durations, intervals,
    # notes volume, rests and so on)
    basechord = current_scale.get_all_chord(num=num)
    # count is the counter for the total number of notes in the piece
    count = 0
    patterncount = 0
    result = chord([])
    while count <= length:
        if pattern is None:
            newchordnotes = random.choice(basechord)
        else:
            newchordnotes = basechord[pattern[patterncount] - 1]
            patterncount += 1
            if patterncount == len(pattern):
                patterncount = 0
        newduration = random.choice(choose_durations)
        newinterval = random.choice(choose_intervals)
        newchord = newchordnotes.set(newduration, newinterval)
        newchord_len = len(newchord)
        if newchord_len < left_hand_meter:
            choose_more = [x for x in current_scale if x not in newchord]
            for g in range(left_hand_meter - newchord_len):
                current_choose = random.choice(choose_more)
                if current_choose.degree < newchord[-1].degree:
                    current_choose = current_choose.up(octave)
                newchord += current_choose
        do_inversion = random.randint(0, 1)
        if do_inversion == 1:
            newchord = newchord.inversion_highest(
                random.randint(2, left_hand_meter - 1))
        for each in newchord.notes:
            each.volume = left_hand_velocity
        chord_temp.names = newchord.names()
        chordinner = [x for x in pick if x.name in chord_temp.names]
        while True:
            firstmelody = _random_composing_choose_melody(
                focused, now_focus, focus_ratio, focus_notes, remained_notes,
                pick, avoid_dim_5, chordinner, newchord, choose_from_chord)
            firstmelody.volume = right_hand_velocity
            newmelody = [firstmelody]
            length_of_chord = sum(newchord.interval)
            intervals = [random.choice(choose_intervals)]
            firstmelody.duration = random.choice(choose_durations)
            while sum(intervals) <= length_of_chord:
                currentmelody = _random_composing_choose_melody(
                    focused, now_focus, focus_ratio, focus_notes,
                    remained_notes, pick, avoid_dim_5, chordinner, newchord,
                    choose_from_chord)
                while abs(currentmelody.degree -
                          newmelody[-1].degree) > melody_interval_tol:
                    currentmelody = _random_composing_choose_melody(
                        focused, now_focus, focus_ratio, focus_notes,
                        remained_notes, pick, avoid_dim_5, chordinner,
                        newchord, choose_from_chord)
                currentmelody.volume = right_hand_velocity
                newinter = random.choice(choose_intervals)
                intervals.append(newinter)
                currentmelody.duration = random.choice(choose_durations)
                newmelody.append(currentmelody)

            distance = [
                abs(x.degree - y.degree) for x in newmelody for y in newmelody
            ]
            if diminished_fifth in distance:
                continue
            else:
                break
        newmelodyall = chord(newmelody, interval=intervals)
        while sum(newmelodyall.interval) > length_of_chord:
            newmelodyall.notes.pop()
            newmelodyall.interval.pop()
        newcombination = newchord.add(newmelodyall, mode='head')
        result = result.add(newcombination)
        count += len(newcombination)
    return result


def perm(n, k=None):
    '''
    return all of the permutations of the elements in x
    '''
    if isinstance(n, int):
        n = list(range(1, n + 1))
    if isinstance(n, str):
        n = list(n)
    if k is None:
        k = len(n)
    result = list(itertools.permutations(n, k))
    return result


def negative_harmony(key,
                     current_chord=None,
                     sort=False,
                     get_map=False,
                     keep_root=True):
    notes_dict = [
        'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'
    ] * 2
    key_tonic = key[0].name
    if key_tonic in standard_dict:
        key_tonic = standard_dict[key_tonic]
    inds = notes_dict.index(key_tonic) + 1
    right_half = notes_dict[inds:inds + 6]
    left_half = notes_dict[inds + 6:inds + 12]
    left_half.reverse()
    map_dict = {
        **{
            left_half[i]: right_half[i]
            for i in range(6)
        },
        **{
            right_half[i]: left_half[i]
            for i in range(6)
        }
    }
    if get_map:
        return map_dict
    if current_chord:
        if isinstance(current_chord, chord):
            temp = copy(current_chord)
            notes = temp.notes
            for each in range(len(notes)):
                current = notes[each]
                if isinstance(current, note):
                    if current.name in standard_dict:
                        current.name = standard_dict[current.name]
                    current_note = closest_note(map_dict[current.name],
                                                current)
                    notes[each] = current.reset(name=current_note.name,
                                                num=current_note.num)
            if sort:
                temp.notes.sort(key=lambda s: s.degree)
            return temp
        else:
            raise ValueError('requires a chord object')
    else:
        temp = copy(key)
        if temp.notes[-1].degree - temp.notes[0].degree == octave:
            temp.notes = temp.notes[:-1]
        notes = temp.notes
        for each in range(len(notes)):
            current = notes[each]
            if current.name in standard_dict:
                current.name = standard_dict[current.name]
            notes[each] = current.reset(name=map_dict[current.name])
        if keep_root:
            root_note = key[0].name
            if root_note in standard_dict:
                root_note = standard_dict[root_note]
            root_note_ind = [i.name for i in notes].index(root_note)
            new_notes = [
                i.name
                for i in notes[root_note_ind + 1:] + notes[:root_note_ind + 1]
            ]
            new_notes.reverse()
            new_notes.append(new_notes[0])
        else:
            new_notes = [i.name for i in notes]
            new_notes.append(new_notes[0])
            new_notes.reverse()
        result = scale(notes=chord(new_notes))
        return result


def guitar_chord(frets,
                 return_chord=True,
                 tuning=guitar_standard_tuning,
                 duration=1 / 4,
                 interval=0,
                 **detect_args):
    '''
    the default tuning is the standard tuning E-A-D-G-B-E,
    you can set the tuning to whatever you want
    the parameter frets is a list contains the frets of each string of
    the guitar you want to press in this chord, sorting from 6th string
    to 1st string (which is from E2 string to E4 string in standard tuning),
    the fret of a string is an integer, if it is 0, then it means you
    play that string open (not press any fret on that string),
    if it is 3 for example, then it means you press the third fret on that
    string, if it is None, then that means you did not play that string
    (mute or just not touch that string)
    this function will return the chord types that form by the frets pressing
    at the strings on a guitar, or you can choose to just return the chord
    '''
    tuning = [N(i) if isinstance(i, str) else i for i in tuning]
    length = len(tuning)
    guitar_notes = [
        tuning[j].up(frets[j]) for j in range(length) if frets[j] is not None
    ]
    result = chord(guitar_notes, duration, interval)
    if return_chord:
        return result
    return detect(result.sortchord(), **detect_args)


def guitar_pattern(frets,
                   tuning=guitar_standard_tuning,
                   default_duration=1 / 8,
                   default_interval=1 / 8,
                   default_volume=100,
                   mapping=drum_mapping):
    tuning = [N(i) if isinstance(i, str) else i for i in tuning]
    length = len(tuning)
    current = [i.strip() for i in frets.split(',')]
    new_current = []
    current_string_ind = length - 1
    current_string_root_note = tuning[current_string_ind]
    for i, each in enumerate(current):
        if each == '':
            continue
        if each.startswith('s'):
            if '(' in each and ')' in each:
                relative_pitch_ind = each.index('(')
                current_temp_string_ind = length - int(
                    each[:relative_pitch_ind].split('s', 1)[1])
                current_settings = each[relative_pitch_ind:]
                each = str(tuning[current_temp_string_ind]) + current_settings
                new_current.append(each)
            else:
                current_string_ind = length - int(each.split('s', 1)[1])
                current_string_root_note = tuning[current_string_ind]
        elif each[0].isnumeric():
            current_fret_ind = len(each) - 1
            for i, j in enumerate(each):
                if not j.isnumeric():
                    current_fret_ind = i - 1
                    break
            current_fret = each[:current_fret_ind + 1]
            current_settings = each[current_fret_ind + 1:]
            current_fret = f'{current_string_root_note}(+{current_fret})'
            each = current_fret + current_settings
            new_current.append(each)
        else:
            new_current.append(each)
    new_frets = ','.join(new_current)
    current_chord = translate(new_frets,
                              default_duration=default_duration,
                              default_interval=default_interval,
                              default_volume=default_volume,
                              mapping=mapping)
    return current_chord


@method_wrapper(chord)
def find_chords_for_melody(melody,
                           mode=None,
                           num=3,
                           chord_num=8,
                           get_pattern=False,
                           chord_length=None,
                           down_octave=1):
    if isinstance(melody, (str, list)):
        melody = chord(melody)
    possible_scales = detect_in_scale(melody, num, get_scales=True)
    if not possible_scales:
        raise ValueError('cannot find a scale suitable for this melody')
    current_scale = possible_scales[0]
    if current_scale.mode != 'major' and current_scale.mode in diatonic_modes:
        current_scale = current_scale.inversion(
            8 - diatonic_modes.index(current_scale.mode))
    current_chordTypes = list(chordTypes.dic.keys())
    result = []
    if get_pattern:
        choose_patterns = [
            '6451', '1645', '6415', '1564', '4565', '4563', '6545', '6543',
            '4536', '6251'
        ]
        roots = [
            current_scale[i]
            for i in [int(k) for k in random.choice(choose_patterns)]
        ]
        length = len(roots)
        counter = 0
    for i in range(chord_num):
        if not get_pattern:
            current_root = random.choice(current_scale.notes[:6])
        else:
            current_root = roots[counter]
            counter += 1
            if counter >= length:
                counter = 0
        current_chord_type = random.choice(current_chordTypes)[0]
        current_chord = get_chord(current_root, current_chord_type)
        while current_chord not in current_scale or current_chord_type == '5' or current_chord in result or (
                chord_length is not None
                and len(current_chord) < chord_length):
            current_chord_type = random.choice(current_chordTypes)[0]
            current_chord = get_chord(current_root, current_chord_type)
        result.append(current_chord)
    if chord_length is not None:
        result = [each[:chord_length + 1] for each in result]
    result = [each - octave * down_octave for each in result]
    return result


@method_wrapper(chord)
def detect_in_scale(current_chord,
                    most_like_num=3,
                    get_scales=False,
                    search_all=True,
                    search_all_each_num=2,
                    major_minor_preference=True,
                    find_altered=True,
                    altered_max_number=1,
                    match_len=False):
    '''
    detect the most possible scales that a set of notes are in,
    this algorithm can also detect scales with altered notes based on
    existing scale definitions
    '''
    current_chord = current_chord.remove_duplicates()
    if not isinstance(current_chord, chord):
        current_chord = chord([trans_note(i) for i in current_chord])
    whole_notes = current_chord.names(standardize_note=True)
    note_names = list(set(whole_notes))
    note_names = [standardize_note(i) for i in note_names]
    first_note = whole_notes[0]
    results = []
    if find_altered:
        altered_scales = []
    for each in scaleTypes:
        scale_name = each[0]
        if scale_name != '12':
            current_scale = scale(first_note, scale_name)
            current_scale_notes = current_scale.names(standardize_note=True)
            if all(i in current_scale_notes for i in note_names):
                results.append(current_scale)
                if not search_all:
                    break
            else:
                if find_altered:
                    altered = [
                        i for i in note_names if i not in current_scale_notes
                    ]
                    if len(altered) <= altered_max_number:
                        altered = [trans_note(i) for i in altered]
                        if all((j.up().name in current_scale_notes
                                or j.down().name in current_scale_notes)
                               for j in altered):
                            altered_msg = []
                            for k in altered:
                                altered_note = k.up().name
                                header = 'b'
                                if not (altered_note in current_scale_notes
                                        and altered_note not in note_names):
                                    altered_note = k.down().name
                                    header = '#'
                                if altered_note in current_scale_notes and altered_note not in note_names:
                                    inds = current_scale_notes.index(
                                        altered_note) + 1
                                    test_scale_exist = copy(
                                        current_scale.notes)
                                    if k.degree - test_scale_exist[
                                            inds - 2].degree < 0:
                                        k = k.up(octave)
                                    test_scale_exist[inds - 1] = k
                                    if chord(test_scale_exist).intervalof(
                                            cumulative=False
                                    ) not in scaleTypes.values():
                                        altered_msg.append(f'{header}{inds}')
                                        altered_scales.append(
                                            f"{current_scale.start.name} {current_scale.mode} {', '.join(altered_msg)}"
                                        )
    if search_all:
        current_chord_len = len(current_chord)
        results.sort(key=lambda s: current_chord_len / len(s), reverse=True)
    if results:
        first_note_scale = results[0]
        inversion_scales = [
            first_note_scale.inversion(i)
            for i in range(2, len(first_note_scale))
        ]
        inversion_scales = [i for i in inversion_scales
                            if i.mode is not None][:search_all_each_num]
        results += inversion_scales
        if major_minor_preference:
            major_or_minor_inds = [
                i for i in range(len(results))
                if results[i].mode in ['major', 'minor']
            ]
            if len(major_or_minor_inds) > 1:
                results.insert(1, results.pop(major_or_minor_inds[1]))
            else:
                if len(major_or_minor_inds) > 0:
                    first_major_minor_ind = major_or_minor_inds[0]
                    if results[first_major_minor_ind].mode == 'major':
                        results.insert(
                            first_major_minor_ind + 1,
                            results[first_major_minor_ind].relative_key())
                    elif results[first_major_minor_ind].mode == 'minor':
                        results.insert(
                            first_major_minor_ind + 1,
                            results[first_major_minor_ind].relative_key())

    results = results[:most_like_num]
    if find_altered:
        for i, each in enumerate(altered_scales):
            current_start, current_mode = each.split(' ', 1)
            current_mode, current_altered = current_mode.rsplit(' ', 1)
            current_scale = scale(current_start, mode=current_mode)
            altered_scales[i] = scale(notes=current_scale.notes,
                                      mode=f'{current_mode} {current_altered}')
        results.extend(altered_scales)
    if match_len:
        results = [
            i for i in results if len(i.get_scale()) == len(current_chord)
        ]
    if get_scales:
        return results
    else:
        results = [f"{each.start.name} {each.mode}" for each in results]
        return results


def _most_appear_notes_detect_scale(current_chord, most_appeared_note):
    third_degree_major = most_appeared_note.up(major_third).name
    third_degree_minor = most_appeared_note.up(minor_third).name
    if current_chord.count(third_degree_major) > current_chord.count(
            third_degree_minor):
        current_mode = 'major'
        if current_chord.count(
                most_appeared_note.up(
                    augmented_fourth).name) > current_chord.count(
                        most_appeared_note.up(perfect_fourth).name):
            current_mode = 'lydian'
        else:
            if current_chord.count(
                    most_appeared_note.up(
                        minor_seventh).name) > current_chord.count(
                            most_appeared_note.up(
                                major_seventh).name):
                current_mode = 'mixolydian'
    else:
        current_mode = 'minor'
        if current_chord.count(
                most_appeared_note.up(
                    major_sixth).name) > current_chord.count(
                        most_appeared_note.up(minor_sixth).name):
            current_mode = 'dorian'
        else:
            if current_chord.count(
                    most_appeared_note.up(
                        minor_second).name) > current_chord.count(
                            most_appeared_note.up(major_second).name):
                current_mode = 'phrygian'
                if current_chord.count(
                        most_appeared_note.up(diminished_fifth).name
                ) > current_chord.count(
                        most_appeared_note.up(perfect_fifth).name):
                    current_mode = 'locrian'
    return scale(most_appeared_note.name, current_mode)


@method_wrapper(chord)
def detect_scale(current_chord,
                 get_scales=False,
                 most_appear_num=5,
                 major_minor_preference=True,
                 is_chord=True):
    '''
    Receive a piece of music and analyze what modes it is using,
    return a list of most likely and exact modes the music has.

    newly added on 2020/4/25, currently in development
    '''
    if not is_chord:
        original_chord = current_chord
        current_chord = concat(current_chord, mode='|')
    current_chord = current_chord.only_notes()
    counts = current_chord.count_appear(sort=True)
    most_appeared_note = [N(each[0]) for each in counts[:most_appear_num]]
    result_scales = [
        _most_appear_notes_detect_scale(current_chord, each)
        for each in most_appeared_note
    ]
    if major_minor_preference:
        major_minor_inds = [
            i for i in range(len(result_scales))
            if result_scales[i].mode in ['major', 'minor']
        ]
        result_scales = [result_scales[i] for i in major_minor_inds] + [
            result_scales[i]
            for i in range(len(result_scales)) if i not in major_minor_inds
        ]
        if major_minor_inds:
            major_minor_inds = [
                i for i in range(len(result_scales))
                if result_scales[i].mode in ['major', 'minor']
            ]
            major_inds = [
                i for i in major_minor_inds if result_scales[i].mode == 'major'
            ]
            minor_inds = [
                i for i in major_minor_inds if result_scales[i].mode == 'minor'
            ]
            current_chord_analysis = chord_analysis(
                current_chord,
                is_chord=True,
                get_original_order=True,
                mode='chords') if is_chord else original_chord
            if current_chord_analysis:
                first_chord = current_chord_analysis[0]
                first_chord_info = first_chord.info()
                if first_chord_info.type == 'chord' and first_chord_info.chord_type:
                    if first_chord_info.chord_type.startswith('maj'):
                        major_scales = [result_scales[i] for i in major_inds]
                        major_scales = [
                            i for i in major_scales
                            if i.start.name == first_chord_info.root
                        ] + [
                            i for i in major_scales
                            if i.start.name != first_chord_info.root
                        ]
                        result_scales = major_scales + [
                            result_scales[j] for j in range(len(result_scales))
                            if j not in major_inds
                        ]
                    elif first_chord_info.chord_type.startswith('m'):
                        minor_scales = [result_scales[i] for i in minor_inds]
                        minor_scales = [
                            i for i in minor_scales
                            if i.start.name == first_chord_info.root
                        ] + [
                            i for i in minor_scales
                            if i.start.name != first_chord_info.root
                        ]
                        result_scales = minor_scales + [
                            result_scales[j] for j in range(len(result_scales))
                            if j not in minor_inds
                        ]

    if get_scales:
        return result_scales
    else:
        return f'most likely scales: {", ".join([f"{i.start.name} {i.mode}" for i in result_scales])}'


@method_wrapper(chord)
def detect_scale2(current_chord,
                  get_scales=False,
                  most_appear_num=3,
                  major_minor_preference=True,
                  is_chord=True):
    '''
    Receive a piece of music and analyze what modes it is using,
    return a list of most likely and exact modes the music has.
    
    This algorithm uses different detect factors from detect_scale function,
    which are the appearance rate of the notes in the tonic chord.
    '''
    if not is_chord:
        current_chord = concat(current_chord, mode='|')
    current_chord = current_chord.only_notes()
    counts = current_chord.count_appear(sort=True)
    counts_dict = {i[0]: i[1] for i in counts}
    appeared_note = [N(each[0]) for each in counts]
    note_scale_count = [
        (i,
         sum([
             counts_dict[k]
             for k in scale(i, 'major').names(standardize_note=True)
         ]) / len(current_chord)) for i in appeared_note
    ]
    note_scale_count.sort(key=lambda s: s[1], reverse=True)
    most_appeared_note, current_key_rate = note_scale_count[0]
    current_scale = scale(most_appeared_note, 'major')
    current_scale_names = current_scale.names(standardize_note=True)
    current_scale_num = len(current_scale_names)
    tonic_chords = [[
        current_scale_names[i],
        current_scale_names[(i + 2) % current_scale_num],
        current_scale_names[(i + 4) % current_scale_num]
    ] for i in range(current_scale_num)]
    scale_notes_counts = [(current_scale_names[k],
                           sum([counts_dict[i] for i in tonic_chords[k]]))
                          for k in range(current_scale_num)]
    scale_notes_counts.sort(key=lambda s: s[1], reverse=True)
    if major_minor_preference:
        scale_notes_counts = [
            i for i in scale_notes_counts
            if i[0] in [current_scale_names[0], current_scale_names[5]]
        ]
        result_scale = [
            scale(i[0],
                  diatonic_modes[current_scale_names.index(i[0])])
            for i in scale_notes_counts
        ]
    else:
        current_tonic = [i[0] for i in scale_notes_counts[:most_appear_num]]
        current_ind = [current_scale_names.index(i) for i in current_tonic]
        current_mode = [diatonic_modes[i] for i in current_ind]
        result_scale = [
            scale(current_tonic[i], current_mode[i])
            for i in range(len(current_tonic))
        ]
    if get_scales:
        return result_scale
    else:
        return ', '.join([f"{i.start.name} {i.mode}" for i in result_scale])


@method_wrapper(chord)
def detect_scale3(current_chord,
                  get_scales=False,
                  most_appear_num=3,
                  major_minor_preference=True,
                  unit=5,
                  key_accuracy_tol=0.9,
                  is_chord=True):
    '''
    Receive a piece of music and analyze what modes it is using,
    return a list of most likely and exact modes the music has.
    
    This algorithm uses the same detect factors as detect_scale2 function,
    but detect the key of the piece in units, which makes modulation detections possible.
    '''
    if not is_chord:
        current_chord = concat(current_chord, mode='|')
    current_chord = current_chord.only_notes()
    result_scale = []
    total_bars = current_chord.bars()
    current_key = None
    current_key_range = [0, 0]
    for i in range(math.ceil(total_bars / unit)):
        current_range = [unit * i, unit * (i + 1)]
        if current_range[1] >= total_bars:
            current_range[1] = total_bars
        current_part = current_chord.cut(*current_range)
        if not current_part:
            current_key_range[1] = current_range[1]
            if result_scale:
                result_scale[-1][0][1] = current_range[1]
            continue
        counts = current_part.count_appear(sort=True)
        counts_dict = {i[0]: i[1] for i in counts}
        appeared_note = [N(each[0]) for each in counts]
        note_scale_count = [
            (i,
             sum([
                 counts_dict[k]
                 for k in scale(i, 'major').names(standardize_note=True)
             ]) / len(current_part)) for i in appeared_note
        ]
        note_scale_count.sort(key=lambda s: s[1], reverse=True)
        most_appeared_note, current_key_rate = note_scale_count[0]
        if current_key_rate < key_accuracy_tol:
            current_key_range[1] = current_range[1]
            if result_scale:
                result_scale[-1][0][1] = current_range[1]
            continue
        current_scale = scale(most_appeared_note, 'major')
        current_scale_names = current_scale.names(standardize_note=True)
        current_scale_num = len(current_scale_names)
        tonic_chords = [[
            current_scale_names[i],
            current_scale_names[(i + 2) % current_scale_num],
            current_scale_names[(i + 4) % current_scale_num]
        ] for i in range(current_scale_num)]
        scale_notes_counts = [(current_scale_names[k],
                               sum([counts_dict[i] for i in tonic_chords[k]]))
                              for k in range(current_scale_num)]
        scale_notes_counts.sort(key=lambda s: s[1], reverse=True)
        if major_minor_preference:
            scale_notes_counts = [
                i for i in scale_notes_counts
                if i[0] in [current_scale_names[0], current_scale_names[5]]
            ]
            current_result_scale = [
                scale(i[0],
                      diatonic_modes[current_scale_names.index(i[0])])
                for i in scale_notes_counts
            ]
        else:
            current_tonic = [
                i[0] for i in scale_notes_counts[:most_appear_num]
            ]
            current_ind = [current_scale_names.index(i) for i in current_tonic]
            current_mode = [diatonic_modes[i] for i in current_ind]
            current_result_scale = [
                scale(current_tonic[i], current_mode[i])
                for i in range(len(current_tonic))
            ]
        if not current_key:
            current_key = current_result_scale
        if not result_scale:
            result_scale.append([current_key_range, current_result_scale])
        if set(current_result_scale) != set(current_key):
            current_key_range = current_range
            current_key = current_result_scale
            result_scale.append([current_key_range, current_result_scale])
        else:
            current_key_range[1] = current_range[1]
            if result_scale:
                result_scale[-1][0][1] = current_range[1]
    if get_scales:
        return result_scale
    else:
        return ', '.join([
            str(i[0]) + ' ' +
            ', '.join([f"{j.start.name} {j.mode}" for j in i[1]])
            for i in result_scale
        ])


def get_chord_root_note(current_chord,
                        get_chord_types=False,
                        to_standard=False):
    if current_chord.type == 'note':
        result = N(current_chord.note_name).name
    else:
        if current_chord.chord_speciality == 'polychord':
            result = get_chord_root_note(current_chord.polychords[0],
                                         to_standard=to_standard)
        else:
            result = current_chord.root
    current_chord_type = current_chord.chord_type
    if current_chord_type is None:
        current_chord_type = ''
    if to_standard:
        result = standard_dict.get(result, result)
    if get_chord_types:
        return result, current_chord_type
    else:
        return result


def get_chord_type_location(current_chord, mode='functions'):
    if current_chord in chordTypes:
        chord_types = [
            i for i in list(chordTypes.keys()) if current_chord in i
        ][0]
        if mode == 'functions':
            for each, value in chord_function_dict.items():
                if each in chord_types:
                    return value
        elif mode == 'notations':
            for each, value in chord_notation_dict.items():
                if each in chord_types:
                    return value


def get_note_degree_in_scale(root_note, current_scale):
    header = ''
    note_names = current_scale.names()
    if root_note not in note_names:
        current_scale_standard = current_scale.standard()
        root_note = standard_dict.get(root_note, root_note)
        if any(get_accidental(i) == 'b' for i in current_scale_standard):
            root_note = N(root_note).flip_accidental().name
        scale_degree = [i[0] for i in current_scale_standard].index(root_note)
        scale_degree_diff = N(root_note).degree - N(
            standardize_note(current_scale_standard[scale_degree])).degree
        if scale_degree_diff == -1:
            header = 'b'
        elif scale_degree_diff == 1:
            header = '#'
    else:
        scale_degree = note_names.index(root_note)
    return scale_degree, header


def get_chord_functions(chords,
                        current_scale,
                        as_list=False,
                        functions_interval=1):
    if not isinstance(chords, list):
        chords = [chords]
    note_names = current_scale.names()
    root_note_list = [
        get_chord_root_note(i, get_chord_types=True) for i in chords
        if i.type == 'chord'
    ]
    functions = []
    for i, each in enumerate(root_note_list):
        current_chord_type = chords[i]
        if current_chord_type.inversion is not None or current_chord_type.non_chord_bass_note is not None:
            current_note = current_chord_type.root
            current_chord_type.order = [2, 4]
            inversion_note = current_chord_type.to_text().rsplit('/', 1)[1]
            current_chord_type.inversion = None
            current_chord_type.non_chord_bass_note = None
            current_inversion_note_degree, current_inversion_note_header = get_note_degree_in_scale(
                inversion_note, current_scale)
            current_function = f'{get_chord_functions(current_chord_type, current_scale)}/{current_inversion_note_header}{current_inversion_note_degree+1}'
        else:
            root_note, chord_types = each
            root_note_obj = note(root_note, 5)
            scale_degree, header = get_note_degree_in_scale(
                root_note, current_scale)
            current_function = chord_functions_roman_numerals[
                scale_degree + 1]
            if chord_types == '' or chord_types == '5':
                original_chord = current_scale(scale_degree)
                third_type = original_chord[1].degree - original_chord[0].degree
                if third_type == minor_third:
                    current_function = current_function.lower()
            else:
                if chord_types in chordTypes:
                    current_chord = get_chord(root_note, chord_types)
                    current_chord_names = current_chord.names()
                else:
                    if chord_types[5:] not in NAME_OF_INTERVAL:
                        current_chord_names = None
                    else:
                        current_chord_names = [
                            root_note_obj.name,
                            root_note_obj.up(NAME_OF_INTERVAL[
                                chord_types[5:]]).name
                        ]
                if chord_types in chord_function_dict:
                    to_lower, function_name = chord_function_dict[
                        chord_types]
                    if to_lower:
                        current_function = current_function.lower()
                    current_function += function_name
                else:
                    function_result = get_chord_type_location(chord_types,
                                                              mode='functions')
                    if function_result:
                        to_lower, function_name = function_result
                        if to_lower:
                            current_function = current_function.lower()
                        current_function += function_name
                    else:
                        if current_chord_names:
                            M3 = root_note_obj.up(major_third).name
                            m3 = root_note_obj.up(minor_third).name
                            if m3 in current_chord_names:
                                current_function = current_function.lower()
                            if len(current_chord_names) >= 3:
                                current_function += '?'
                        else:
                            current_function += '?'
            current_function = header + current_function
        functions.append(current_function)
    if as_list:
        return functions
    return (' ' * functions_interval + '–' +
            ' ' * functions_interval).join(functions)


def get_chord_notations(chords,
                        as_list=False,
                        functions_interval=1,
                        split_symbol='|'):
    if not isinstance(chords, list):
        chords = [chords]
    root_note_list = [
        get_chord_root_note(i, get_chord_types=True) for i in chords
        if i.type == 'chord'
    ]
    notations = []
    for i, each in enumerate(root_note_list):
        current_chord_type = chords[i]
        if current_chord_type.inversion is not None or current_chord_type.non_chord_bass_note is not None:
            current_note = current_chord_type.root
            current_chord_type.order = [2, 4]
            inversion_note = current_chord_type.to_text().rsplit('/', 1)[1]
            current_chord_type.inversion = None
            current_chord_type.non_chord_bass_note = None
            current_notation = f'{get_chord_notations(current_chord_type)}/{inversion_note}'
        else:
            root_note, chord_types = each
            current_notation = root_note
            root_note_obj = note(root_note, 5)
            if chord_types in chord_notation_dict:
                current_notation += chord_notation_dict[chord_types]
            else:
                notation_result = get_chord_type_location(chord_types,
                                                          mode='notations')
                if notation_result:
                    current_notation += notation_result
                else:
                    if chord_types in chordTypes:
                        current_chord = get_chord(root_note, chord_types)
                        current_chord_names = current_chord.names()
                    else:
                        if chord_types[5:] not in NAME_OF_INTERVAL:
                            current_chord_names = None
                        else:
                            current_chord_names = [
                                root_note_obj.name,
                                root_note_obj.up(NAME_OF_INTERVAL[
                                    chord_types[5:]]).name
                            ]
                    if current_chord_names:
                        M3 = root_note_obj.up(major_third).name
                        m3 = root_note_obj.up(minor_third).name
                        if m3 in current_chord_names:
                            current_notation += '-'
                        if len(current_chord_names) >= 3:
                            current_notation += '?'
                    else:
                        current_notation += '?'
        notations.append(current_notation)
    if as_list:
        return notations
    return (' ' * functions_interval + split_symbol +
            ' ' * functions_interval).join(notations)


@method_wrapper(chord)
def chord_functions_analysis(current_chord,
                             functions_interval=1,
                             function_symbol='-',
                             split_symbol='|',
                             chord_mode='function',
                             fixed_scale_type=None,
                             return_scale_degrees=False,
                             write_to_file=False,
                             filename='chords functions analysis result.txt',
                             each_line_chords_number=15,
                             space_lines=2,
                             full_chord_msg=False,
                             is_chord_analysis=True,
                             detect_scale_function=detect_scale2,
                             major_minor_preference=True,
                             is_detect=True,
                             detect_args={},
                             chord_analysis_args={}):
    '''
    analysis the chord functions of a chord instance
    '''
    if is_chord_analysis:
        current_chord = current_chord.only_notes()
    else:
        if isinstance(current_chord, chord):
            current_chord = [current_chord]
    if fixed_scale_type:
        scales = fixed_scale_type
    else:
        scales = detect_scale_function(
            current_chord,
            major_minor_preference=major_minor_preference,
            get_scales=True,
            is_chord=is_chord_analysis)[0]
    if is_chord_analysis:
        result = chord_analysis(current_chord,
                                mode='chords',
                                **chord_analysis_args)
        result = [i.standardize() for i in result]
    else:
        result = current_chord
    if is_detect:
        actual_chords = [
            detect(i, get_chord_type=True, **detect_args) for i in result
        ]
    else:
        actual_chords = current_chord
    if chord_mode == 'function':
        chord_progressions = get_chord_functions(
            chords=actual_chords,
            current_scale=scales,
            as_list=True,
            functions_interval=functions_interval)
        if full_chord_msg:
            chord_progressions = [
                f'{actual_chords[i].to_text()} {chord_progressions[i]}'
                for i in range(len(chord_progressions))
            ]
        if return_scale_degrees:
            return chord_progressions
        if write_to_file:
            num = (len(chord_progressions) // each_line_chords_number) + 1
            delimiter = ' ' * functions_interval + function_symbol + ' ' * functions_interval
            chord_progressions = [
                delimiter.join(chord_progressions[each_line_chords_number *
                                                  i:each_line_chords_number *
                                                  (i + 1)]) + delimiter
                for i in range(num)
            ]
            chord_progressions[-1] = chord_progressions[-1][:-len(delimiter)]
            chord_progressions = ('\n' * space_lines).join(chord_progressions)
        else:
            chord_progressions = f' {function_symbol} '.join(
                chord_progressions)
    elif chord_mode == 'notation':
        if full_chord_msg:
            num = (len(actual_chords) // each_line_chords_number) + 1
            delimiter = ' ' * functions_interval + split_symbol + ' ' * functions_interval
            chord_progressions = [
                delimiter.join([
                    j.to_text()
                    for j in actual_chords[each_line_chords_number *
                                           i:each_line_chords_number * (i + 1)]
                ]) + delimiter for i in range(num)
            ]
            chord_progressions[-1] = chord_progressions[-1][:-len(delimiter)]
            chord_progressions = ('\n' * space_lines).join(chord_progressions)
        elif not write_to_file:
            chord_progressions = get_chord_notations(
                chords=actual_chords,
                as_list=True,
                functions_interval=functions_interval,
                split_symbol=split_symbol)
            if return_scale_degrees:
                return chord_progressions
            chord_progressions = f' {split_symbol} '.join(chord_progressions)
        else:
            chord_progressions = get_chord_notations(actual_chords, True,
                                                     functions_interval,
                                                     split_symbol)
            if return_scale_degrees:
                return chord_progressions
            num = (len(chord_progressions) // each_line_chords_number) + 1
            delimiter = ' ' * functions_interval + split_symbol + ' ' * functions_interval
            chord_progressions = [
                delimiter.join(chord_progressions[each_line_chords_number *
                                                  i:each_line_chords_number *
                                                  (i + 1)]) + delimiter
                for i in range(num)
            ]
            chord_progressions[-1] = chord_progressions[-1][:-len(delimiter)]
            chord_progressions = ('\n' * space_lines).join(chord_progressions)
    else:
        raise ValueError("chord mode must be 'function' or 'notation'")
    spaces = '\n' * space_lines
    analysis_result = f'key: {scales[0].name} {scales.mode}{spaces}{chord_progressions}'
    if write_to_file:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(analysis_result)
        analysis_result += spaces + f"Successfully write the chord analysis result as a text file, please see '{filename}'."
        return analysis_result
    else:
        return analysis_result


@method_wrapper(chord)
def split_melody(current_chord,
                 mode='chord',
                 melody_tol=minor_seventh,
                 chord_tol=major_sixth,
                 get_off_overlap_notes=True,
                 get_off_same_time=True,
                 average_degree_length=8,
                 melody_degree_tol='B4'):
    '''
    split the melody part of a chord instance
    if mode == 'notes', return a list of main melody notes
    if mode == 'index', return a list of indexes of main melody notes
    if mode == 'chord', return a chord with main melody notes with original places
    '''
    if melody_degree_tol is not None and not isinstance(
            melody_degree_tol, note):
        melody_degree_tol = to_note(melody_degree_tol)
    if mode == 'notes':
        result = split_melody(current_chord=current_chord,
                              mode='index',
                              melody_tol=melody_tol,
                              chord_tol=chord_tol,
                              get_off_overlap_notes=get_off_overlap_notes,
                              get_off_same_time=get_off_same_time,
                              average_degree_length=average_degree_length,
                              melody_degree_tol=melody_degree_tol)
        current_chord_notes = current_chord.notes
        melody = [current_chord_notes[t] for t in result]
        return melody
    elif mode == 'chord':
        result = split_melody(current_chord=current_chord,
                              mode='index',
                              melody_tol=melody_tol,
                              chord_tol=chord_tol,
                              get_off_overlap_notes=get_off_overlap_notes,
                              get_off_same_time=get_off_same_time,
                              average_degree_length=average_degree_length,
                              melody_degree_tol=melody_degree_tol)
        return current_chord.pick(result)

    elif mode == 'index':
        current_chord_notes = current_chord.notes
        current_chord_interval = current_chord.interval
        whole_length = len(current_chord)
        for k in range(whole_length):
            current_chord_notes[k].number = k
        other_messages_inds = [
            i for i in range(whole_length)
            if not isinstance(current_chord_notes[i], note)
        ]
        temp = current_chord.only_notes()
        N = len(temp)
        whole_notes = temp.notes
        whole_interval = temp.interval
        if get_off_overlap_notes:
            for j in range(N):
                current_note = whole_notes[j]
                current_interval = whole_interval[j]
                if current_interval != 0:
                    if current_note.duration >= current_interval:
                        current_note.duration = current_interval
                else:
                    for y in range(j + 1, N):
                        next_interval = whole_interval[y]
                        if next_interval != 0:
                            if current_note.duration >= next_interval:
                                current_note.duration = next_interval
                            break
            unit_duration = min([i.duration for i in whole_notes])
            for each in whole_notes:
                each.duration = unit_duration
            whole_interval = [
                current_chord_interval[j.number] for j in whole_notes
            ]
            k = 0
            while k < len(whole_notes) - 1:
                current_note = whole_notes[k]
                next_note = whole_notes[k + 1]
                current_interval = whole_interval[k]
                if current_note.degree == next_note.degree:
                    if current_interval == 0:
                        del whole_notes[k + 1]
                        del whole_interval[k]
                k += 1
        if get_off_same_time:
            play_together = find_all_continuous(whole_interval, 0)
            for each in play_together:
                max_ind = max(each, key=lambda t: whole_notes[t].degree)
                get_off = set(each) - {max_ind}
                for each_ind in get_off:
                    whole_notes[each_ind] = None
                    whole_interval[each_ind] = None
            whole_notes = [x for x in whole_notes if x is not None]
            whole_interval = [x for x in whole_interval if x is not None]
        N = len(whole_notes) - 1
        start = 0
        if whole_notes[1].degree - whole_notes[0].degree >= chord_tol:
            start = 1
        i = start + 1
        melody = [whole_notes[start]]
        notes_num = 1
        melody_interval = [whole_interval[start]]
        while i < N:
            current_note = whole_notes[i]
            next_note = whole_notes[i + 1]
            current_note_interval = whole_interval[i]
            next_degree_diff = next_note.degree - current_note.degree
            recent_notes = add_to_index(melody_interval, average_degree_length,
                                        notes_num - 1, -1, -1)
            if recent_notes:
                current_average_degree = sum(
                    [melody[j].degree
                     for j in recent_notes]) / len(recent_notes)
                average_diff = current_average_degree - current_note.degree
                if average_diff <= melody_tol:
                    if melody[-1].degree - current_note.degree < chord_tol:
                        melody.append(current_note)
                        notes_num += 1
                        melody_interval.append(current_note_interval)
                    else:
                        if abs(next_degree_diff) < chord_tol and not (
                                melody_degree_tol is not None
                                and current_note.degree
                                < melody_degree_tol.degree):
                            melody.append(current_note)
                            notes_num += 1
                            melody_interval.append(current_note_interval)
                else:

                    if (melody[-1].degree - current_note.degree < chord_tol
                            and next_degree_diff < chord_tol
                            and all(k.degree - current_note.degree < chord_tol
                                    for k in melody[-2:])):
                        melody.append(current_note)
                        notes_num += 1
                        melody_interval.append(current_note_interval)
                    else:
                        if (abs(next_degree_diff) < chord_tol
                                and not (melody_degree_tol is not None
                                         and current_note.degree
                                         < melody_degree_tol.degree) and
                                all(k.degree - current_note.degree < chord_tol
                                    for k in melody[-2:])):
                            melody.append(current_note)
                            notes_num += 1
                            melody_interval.append(current_note_interval)
            i += 1
        melody_inds = [each.number for each in melody]
        whole_inds = melody_inds + other_messages_inds
        whole_inds.sort()
        return whole_inds


@method_wrapper(chord)
def split_chord(current_chord, mode='chord', **args):
    '''
    split the chord part of a chord instance
    '''
    melody_ind = split_melody(current_chord=current_chord,
                              mode='index',
                              **args)
    N = len(current_chord)
    whole_notes = current_chord.notes
    other_messages_inds = [
        i for i in range(N) if not isinstance(whole_notes[i], note)
    ]
    chord_ind = [
        i for i in range(N)
        if (i not in melody_ind) or (i in other_messages_inds)
    ]
    if mode == 'index':
        return chord_ind
    elif mode == 'notes':
        return [whole_notes[k] for k in chord_ind]
    elif mode == 'chord':
        return current_chord.pick(chord_ind)


@method_wrapper(chord)
def split_all(current_chord, mode='chord', **args):
    '''
    split the main melody and chords part of a piece of music,
    return both of main melody and chord part
    '''
    melody_ind = split_melody(current_chord=current_chord,
                              mode='index',
                              **args)
    N = len(current_chord)
    whole_notes = current_chord.notes
    chord_ind = [
        i for i in range(N)
        if (i not in melody_ind) or (not isinstance(whole_notes[i], note))
    ]
    if mode == 'index':
        return [melody_ind, chord_ind]
    elif mode == 'notes':
        return [[whole_notes[j] for j in melody_ind],
                [whole_notes[k] for k in chord_ind]]
    elif mode == 'chord':
        result_chord = current_chord.pick(chord_ind)
        result_melody = current_chord.pick(melody_ind)
        return [result_melody, result_chord]


@method_wrapper(chord)
def chord_analysis(chords,
                   mode='chord names',
                   is_chord=False,
                   new_chord_tol=minor_seventh,
                   get_original_order=False,
                   formatted=False,
                   formatted_mode=1,
                   output_as_file=False,
                   each_line_chords_number=5,
                   functions_interval=1,
                   split_symbol='|',
                   space_lines=2,
                   detect_args={},
                   split_chord_args={}):
    '''
    analysis the chord progressions of a chord instance
    '''
    chords = chords.only_notes()
    if not is_chord:
        chord_notes = split_chord(chords, 'chord', **split_chord_args)
    else:
        chord_notes = chords
    if formatted or (mode in ['inds', 'bars', 'bars start']):
        get_original_order = True
    whole_notes = chord_notes.notes
    chord_ls = []
    current_chord = [whole_notes[0]]
    if get_original_order:
        chord_inds = []
    N = len(whole_notes) - 1
    for i in range(N):
        current_note = whole_notes[i]
        next_note = whole_notes[i + 1]
        if current_note.degree <= next_note.degree:
            if i > 0 and chord_notes.interval[
                    i - 1] == 0 and chord_notes.interval[i] != 0:
                chord_ls.append(chord(current_chord).sortchord())
                if get_original_order:
                    chord_inds.append([i + 1 - len(current_chord), i + 1])
                current_chord = []
                current_chord.append(next_note)

            else:
                current_chord.append(next_note)
        elif chord_notes.interval[i] == 0:
            current_chord.append(next_note)
        elif current_note.degree > next_note.degree:
            if len(current_chord) < 3:
                if len(current_chord) == 2:
                    if next_note.degree > min(
                        [k.degree for k in current_chord]):
                        current_chord.append(next_note)
                    else:
                        chord_ls.append(chord(current_chord).sortchord())
                        if get_original_order:
                            chord_inds.append(
                                [i + 1 - len(current_chord), i + 1])
                        current_chord = []
                        current_chord.append(next_note)
                else:
                    current_chord.append(next_note)
            else:
                current_chord_degrees = sorted(
                    [k.degree for k in current_chord])
                if next_note.degree >= current_chord_degrees[2]:
                    if current_chord_degrees[
                            -1] - next_note.degree >= new_chord_tol:
                        chord_ls.append(chord(current_chord).sortchord())
                        if get_original_order:
                            chord_inds.append(
                                [i + 1 - len(current_chord), i + 1])
                        current_chord = []
                        current_chord.append(next_note)
                    else:
                        current_chord.append(next_note)
                else:
                    chord_ls.append(chord(current_chord).sortchord())
                    if get_original_order:
                        chord_inds.append([i + 1 - len(current_chord), i + 1])
                    current_chord = []
                    current_chord.append(next_note)
    chord_ls.append(chord(current_chord).sortchord())
    if get_original_order:
        chord_inds.append([N + 1 - len(current_chord), N + 1])
    current_chord = []
    if formatted:
        result = [detect(each, **detect_args) for each in chord_ls]
        result = [i if not isinstance(i, list) else i[0] for i in result]
        result_notes = [chord_notes[k[0]:k[1]] for k in chord_inds]
        result_notes = [
            each.sortchord() if all(j == 0
                                    for j in each.interval[:-1]) else each
            for each in result_notes
        ]
        if formatted_mode == 0:
            chords_formatted = '\n\n'.join([
                f'chord {i+1}: {result[i]}    notes: {result_notes[i]}'
                for i in range(len(result))
            ])
        elif formatted_mode == 1:
            num = (len(result) // each_line_chords_number) + 1
            delimiter = ' ' * functions_interval + split_symbol + ' ' * functions_interval
            chords_formatted = [
                delimiter.join(result[each_line_chords_number *
                                      i:each_line_chords_number * (i + 1)]) +
                delimiter for i in range(num)
            ]
            chords_formatted[-1] = chords_formatted[-1][:-len(delimiter)]
            chords_formatted = ('\n' * space_lines).join(chords_formatted)
        if output_as_file:
            with open('chord analysis result.txt', 'w', encoding='utf-8') as f:
                f.write(chords_formatted)
            chords_formatted += "\n\nSuccessfully write the chord analysis result as a text file, please see 'chord analysis result.txt'."
        return chords_formatted
    if mode == 'chords':
        if get_original_order:
            return [chord_notes[k[0]:k[1]] for k in chord_inds]
        return chord_ls
    elif mode == 'chord names':
        result = [detect(each, **detect_args) for each in chord_ls]
        return [i if not isinstance(i, list) else i[0] for i in result]
    elif mode == 'inds':
        return [[i[0], i[1]] for i in chord_inds]
    elif mode == 'bars':
        inds = [[i[0], i[1]] for i in chord_inds]
        return [chord_notes.count_bars(k[0], k[1]) for k in inds]
    elif mode == 'bars start':
        inds = [[i[0], i[1]] for i in chord_inds]
        return [chord_notes.count_bars(k[0], k[1])[0] for k in inds]


def find_continuous(current_chord, value, start=None, stop=None):
    if start is None:
        start = 0
    if stop is None:
        stop = len(current_chord)
    inds = []
    appear = False
    for i in range(start, stop):
        if not appear:
            if current_chord[i] == value:
                appear = True
                inds.append(i)
        else:
            if current_chord[i] == value:
                inds.append(i)
            else:
                break
    return inds


def find_all_continuous(current_chord, value, start=None, stop=None):
    if start is None:
        start = 0
    if stop is None:
        stop = len(current_chord)
    result = []
    inds = []
    appear = False
    for i in range(start, stop):
        if current_chord[i] == value:
            if appear:
                inds.append(i)
            else:
                if inds:
                    inds.append(inds[-1] + 1)
                    result.append(inds)
                appear = True
                inds = [i]
        else:
            appear = False
    if inds:
        result.append(inds)
    try:
        if result[-1][-1] >= len(current_chord):
            del result[-1][-1]
    except:
        pass
    return result


def add_to_index(current_chord, value, start=None, stop=None, step=1):
    if start is None:
        start = 0
    if stop is None:
        stop = len(current_chord)
    inds = []
    counter = 0
    for i in range(start, stop, step):
        counter += current_chord[i]
        inds.append(i)
        if counter == value:
            inds.append(i + step)
            break
        elif counter > value:
            break
    if not inds:
        inds = [0]
    return inds


def add_to_last_index(current_chord, value, start=None, stop=None, step=1):
    if start is None:
        start = 0
    if stop is None:
        stop = len(current_chord)
    ind = 0
    counter = 0
    for i in range(start, stop, step):
        counter += current_chord[i]
        ind = i
        if counter == value:
            ind += step
            break
        elif counter > value:
            break
    return ind


def humanize(current_chord,
             timing_range=[-1 / 128, 1 / 128],
             velocity_range=[-10, 10]):
    '''
    add random dynamic changes in given ranges to timing and velocity of notes to a piece
    '''
    temp = copy(current_chord)
    if isinstance(temp, piece):
        temp.tracks = [
            humanize(each, timing_range, velocity_range)
            for each in temp.tracks
        ]
        return temp
    elif isinstance(temp, chord):
        if velocity_range:
            for each in temp.notes:
                each.volume += random.uniform(*velocity_range)
                if each.volume < 0:
                    each.volume = 0
                elif each.volume > 127:
                    each.volume = 127
                each.volume = int(each.volume)
        if timing_range:
            places = [0] + [
                sum(temp.interval[:i]) for i in range(1,
                                                      len(temp.notes) + 1)
            ]
            places = [places[0]] + [
                each + random.choices([random.uniform(*timing_range), 0])[0]
                for each in places[1:]
            ]
            temp.interval = [
                abs(places[i] - places[i - 1]) for i in range(1, len(places))
            ]
        return temp

def write_pop(midi_file_name='pop.mid',
        scale_type=scale('C', 'major'),
        length=[10, 20],
        melody_ins=1,
        chord_ins=1,
        bpm=120,
        scale_type2=None,
        choose_chord_notes_num=[4],
        default_chord_durations=1 / 2,
        inversion_highest_num=2,
        choose_chord_intervals=[1 / 8],
        choose_melody_durations=[1 / 8, 1 / 16, beat(1 / 8, 1)],
        choose_start_times=[0],
        choose_chord_progressions=None,
        current_choose_chord_progressions_list=None,
        melody_chord_octave_diff=2,
        choose_melody_rhythms=None,
        with_drum_beats=True,
        drum_ins=1,
        with_bass=True,
        bass_octave=2,
        choose_bass_rhythm=default_choose_bass_rhythm,
        choose_bass_techniques=default_choose_bass_playing_techniques
):
    '''
    write a pop/dance song with melody, chords, bass and drum in a given key,
    currently in development
    '''
    if isinstance(length, list):
        length = random.randint(*length)
    if isinstance(bpm, list):
        bpm = random.randint(*bpm)
    if isinstance(melody_ins, list):
        melody_ins = random.choice(melody_ins)
    if isinstance(chord_ins, list):
        chord_ins = random.choice(chord_ins)
    melody_octave = scale_type[0].num + melody_chord_octave_diff
    if 'minor' in scale_type.mode:
        scale_type = scale_type.relative_key()

    if choose_chord_progressions is None:
        choose_chord_progressions = random.choice(
            choose_chord_progressions_list
            if current_choose_chord_progressions_list is
            None else current_choose_chord_progressions_list)
    choose_chords = scale_type % (choose_chord_progressions,
                                  default_chord_durations, 0,
                                  random.choice(choose_chord_notes_num))
    for i in range(len(choose_chords)):
        each = choose_chords[i]
        if each[0] == scale_type[4]:
            each = C(f'{scale_type[4].name}',
                     each[0].num,
                     duration=default_chord_durations) @ [1, 2, 3, 1.1]
            choose_chords[i] = each

    if inversion_highest_num is not None:
        choose_chords = [i ^ inversion_highest_num for i in choose_chords]
    chord_num = len(choose_chords)
    length_count = 0
    chord_ind = 0
    melody = chord([])
    chords_part = chord([])
    if with_bass:
        bass_part = chord([])
        current_bass_techniques = None
        if choose_bass_techniques is not None:
            current_bass_techniques = random.choice(choose_bass_techniques)
    while length_count < length:
        current_chord = choose_chords[chord_ind]
        current_chord_interval = random.choice(choose_chord_intervals)
        if isinstance(current_chord_interval, beat):
            current_chord_interval = current_chord_interval.get_duration()
        current_chord = current_chord.set(interval=current_chord_interval)
        current_chord_length = current_chord.bars(mode=0)
        chords_part |= current_chord
        length_count = chords_part.bars(mode=0)
        if with_bass:
            current_chord_tonic = note(
                scale_type[int(choose_chord_progressions[chord_ind]) - 1].name,
                bass_octave)
            if choose_bass_rhythm is None:
                current_bass_part = chord([current_chord_tonic]) % (
                    current_chord_length, current_chord_length)
            else:
                current_bass_part = get_chords_from_rhythm(
                    chord([current_chord_tonic]),
                    rhythm(*random.choice(choose_bass_rhythm)))
                if current_bass_techniques:
                    if current_bass_techniques == 'octaves':
                        if len(current_bass_part) > 1:
                            for i in range(len(current_bass_part)):
                                if i % 2 != 0:
                                    current_bass_part[i] += 12
            bass_part |= current_bass_part
        while melody.bars(mode=0) < length_count:
            if scale_type2:
                current_melody = copy(random.choice(scale_type2.notes))
            else:
                current_melody = copy(
                    random.choice(current_chord.notes + scale_type.notes))
                current_melody.num = melody_octave
            current_chord_duration = random.choice(choose_melody_durations)
            if isinstance(current_chord_duration, beat):
                current_chord_duration = current_chord_duration.get_duration()
            current_melody.duration = current_chord_duration
            melody.notes.append(current_melody)
            melody.interval.append(copy(current_melody.duration))
        chord_ind += 1
        if chord_ind >= chord_num:
            chord_ind = 0
    chords_part.set_volume(70)
    result = piece(tracks=[melody, chords_part],
                   instruments=[melody_ins, chord_ins],
                   bpm=bpm,
                   start_times=[0, random.choice(choose_start_times)],
                   track_names=['melody', 'chords'],
                   channels=[0, 1])
    result.choose_chord_progressions = choose_chord_progressions
    if with_drum_beats:
        current_drum_beats = drum(
            random.choice(default_choose_drum_beats))
        current_drum_beat_repeat_num = math.ceil(
            length / current_drum_beats.notes.bars())
        current_drum_beats *= current_drum_beat_repeat_num
        current_drum_beats.notes.set_volume(70)
        result.append(
            track(content=current_drum_beats.notes,
                  instrument=drum_ins,
                  start_time=result.start_times[1],
                  track_name='drum',
                  channel=9))
    if with_bass:
        bass_part.set_volume(80)
        result.append(
            track(content=bass_part,
                  instrument=34,
                  start_time=result.start_times[1],
                  track_name='bass',
                  channel=2))
        
    if midi_file_name:
        write(result, name=midi_file_name)
        
    else:
        return result

#=================================================================================================
# This is the end of musicpy_pop_generator Python module
#=================================================================================================