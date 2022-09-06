from exception import KeyParseException
import copy


class KeySignature(object):
  """Internal representation of a MusicXML key signature."""

  def __init__(self, state, xml_key=None):
    self.xml_key = xml_key
    # MIDI and MusicXML identify key by using "fifths":
    # -1 = F, 0 = C, 1 = G etc.
    self.key = 0
    # mode is "major" or "minor" only: MIDI only supports major and minor
    self.mode = 'major'
    self.time_position = -1
    self.xml_position = -1
    self.state = copy.copy(state)
    if xml_key is not None:
      self._parse()

  def _parse(self):
    """Parse the MusicXML <key> element into a MIDI compatible key.

    If the mode is not minor (e.g. dorian), default to "major"
    because MIDI only supports major and minor modes.


    Raises:
      KeyParseException: If the fifths element is missing.
    """
    fifths = self.xml_key.find('fifths')
    if fifths is None:
      raise KeyParseException(
        'Could not find fifths attribute in key signature.')
    self.key = int(self.xml_key.find('fifths').text)
    mode = self.xml_key.find('mode')
    # Anything not minor will be interpreted as major
    if mode != 'minor':
      mode = 'major'
    self.mode = mode
    self.time_position = self.state.time_position
    self.xml_position = self.state.xml_position

  def __str__(self):
    keys = (['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D',
             'A', 'E', 'B', 'F#', 'C#'])
    key_string = keys[self.key + 7] + ' ' + self.mode
    key_string += ' (@time: ' + str(self.time_position) + ')'
    return key_string

  def __eq__(self, other):
    isequal = self.key == other.key
    isequal = isequal and (self.mode == other.mode)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal
