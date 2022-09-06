from exception import AlternatingTimeSignatureException, TimeSignatureParseException
import copy


class TimeSignature(object):
  """Internal representation of a MusicXML time signature.

  Does not support:
  - Composite time signatures: 3+2/8
  - Alternating time signatures 2/4 + 3/8
  - Senza misura
  """

  def __init__(self, state, xml_time=None):
    self.xml_time = xml_time
    self.numerator = -1
    self.denominator = -1
    self.time_position = 0
    self.xml_position = 0
    self.state = copy.copy(state)
    if xml_time is not None:
      self._parse()

  def _parse(self):
    """Parse the MusicXML <time> element."""
    if (len(self.xml_time.findall('beats')) > 1 or
        len(self.xml_time.findall('beat-type')) > 1):
      # If more than 1 beats or beat-type found, this time signature is
      # not supported (ex: alternating meter)
      raise AlternatingTimeSignatureException('Alternating Time Signature')

    if 'symbol' in self.xml_time.attrib.keys():
      symbol = self.xml_time.attrib['symbol']
      if symbol == 'cut':
        self.numerator = 2
        self.denominator = 2
      elif symbol == 'common':
        self.numerator = 4
        self.denominator = 4
      else:
        print('Unknown time signature symbol: ', symbol)
    else:
      beats = self.xml_time.find('beats').text
      beat_type = self.xml_time.find('beat-type').text
      try:
        self.numerator = int(beats)
        self.denominator = int(beat_type)
      except ValueError:
        raise TimeSignatureParseException(
          'Could not parse time signature: {}/{}'.format(beats, beat_type))
    self.time_position = self.state.time_position
    self.xml_position = self.state.xml_position

  def __str__(self):
    time_sig_str = str(self.numerator) + '/' + str(self.denominator)
    time_sig_str += ' (@time: ' + str(self.time_position) + ')'
    return time_sig_str

  def __eq__(self, other):
    isequal = self.numerator == other.numerator
    isequal = isequal and (self.denominator == other.denominator)
    isequal = isequal and (self.time_position == other.time_position)
    return isequal

  def __ne__(self, other):
    return not self.__eq__(other)
