from __future__ import division
from fractions import Fraction
import constants
from exception import InvalidNoteDurationTypeException


class NoteDuration(object):
  """Internal representation of a MusicXML note's duration properties."""

  TYPE_RATIO_MAP = {'maxima': Fraction(8, 1), 'long': Fraction(4, 1),
                    'breve': Fraction(2, 1), 'whole': Fraction(1, 1),
                    'half': Fraction(1, 2), 'quarter': Fraction(1, 4),
                    'eighth': Fraction(1, 8), '16th': Fraction(1, 16),
                    '32nd': Fraction(1, 32), '64th': Fraction(1, 64),
                    '128th': Fraction(1, 128), '256th': Fraction(1, 256),
                    '512th': Fraction(1, 512), '1024th': Fraction(1, 1024)}

  def __init__(self, state):
    self.duration = 0  # MusicXML duration
    self.midi_ticks = 0  # Duration in MIDI ticks
    self.seconds = 0  # Duration in seconds
    self.time_position = 0  # Onset time in seconds
    self.xml_position = 0
    self.dots = 0  # Number of augmentation dots
    self._type = 'quarter'  # MusicXML duration type
    self.tuplet_ratio = Fraction(1, 1)  # Ratio for tuplets (default to 1)
    self.is_grace_note = True  # Assume true until not found
    self.state = state
    self.preceded_by_grace_note = False  # The note is preceded by a grace note(s)
    self.grace_order = 0  # If there are multiple grace notes, record the order of notes (-1, -2)
    self.num_grace = 0
    self.is_first_grace_note = False

  def parse_duration(self, is_in_chord, is_grace_note, duration):
    """Parse the duration of a note and compute timings."""
    self.duration = int(duration)
    # Due to an error in Sibelius' export, force this note to have the
    # duration of the previous note if it is in a chord
    if is_in_chord:
      self.duration = self.state.previous_note_duration

    self.midi_ticks = self.duration
    self.midi_ticks *= (constants.STANDARD_PPQ / self.state.divisions)

    self.seconds = (self.midi_ticks / constants.STANDARD_PPQ)
    self.seconds *= self.state.seconds_per_quarter

    self.time_position = float("{0:.8f}".format(self.state.time_position))
    self.xml_position = self.state.xml_position

    # Not sure how to handle durations of grace notes yet as they
    # steal time from subsequent notes and they do not have a
    # <duration> tag in the MusicXML
    self.is_grace_note = is_grace_note

    if is_in_chord:
      # If this is a chord, set the time position to the time position
      # of the previous note (i.e. all the notes in the chord will have
      # the same time position)
      self.time_position = self.state.previous_note_time_position
      self.xml_position = self.state.previous_note_xml_position
      # pass
    else:
      # Only increment time positions once in chord
      self.state.time_position += self.seconds
      self.state.xml_position += self.duration

  def _convert_type_to_ratio(self):
    """Convert the MusicXML note-type-value to a Python Fraction.

    Examples:
    - whole = 1/1
    - half = 1/2
    - quarter = 1/4
    - 32nd = 1/32

    Returns:
      A Fraction object representing the note type.
    """
    return self.TYPE_RATIO_MAP[self.type]

  def duration_ratio(self):
    """Compute the duration ratio of the note as a Python Fraction.

    Examples:
    - Whole Note = 1
    - Quarter Note = 1/4
    - Dotted Quarter Note = 3/8
    - Triplet eighth note = 1/12

    Returns:
      The duration ratio as a Python Fraction.
    """
    # Get ratio from MusicXML note type
    duration_ratio = Fraction(1, 1)
    type_ratio = self._convert_type_to_ratio()

    # Compute tuplet ratio
    duration_ratio /= self.tuplet_ratio
    type_ratio /= self.tuplet_ratio

    # Add augmentation dots
    one_half = Fraction(1, 2)
    dot_sum = Fraction(0, 1)
    for dot in range(self.dots):
      dot_sum += (one_half ** (dot + 1)) * type_ratio

    duration_ratio = type_ratio + dot_sum

    # If the note is a grace note, force its ratio to be 0
    # because it does not have a <duration> tag
    if self.is_grace_note:
      duration_ratio = Fraction(0, 1)
    return duration_ratio

  def duration_float(self):
    """Return the duration ratio as a float."""
    ratio = self.duration_ratio()
    return ratio.numerator / ratio.denominator

  @property
  def type(self):
    return self._type

  @type.setter
  def type(self, new_type):
    if new_type not in self.TYPE_RATIO_MAP:
      raise InvalidNoteDurationTypeException(
        'Note duration type "{}" is not valid'.format(new_type))
    self._type = new_type
