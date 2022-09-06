from tempo import Tempo
from chord_symbol import ChordSymbol
from note import Note

DEFAULT_MIDI_PROGRAM = 0  # Default MIDI Program (0 = grand piano)
DEFAULT_MIDI_CHANNEL = 0  # Default MIDI Channel (0 = first channel)


class ScorePart(object):
  """"Internal representation of a MusicXML <score-part>.

  A <score-part> element contains MIDI program and channel info
  for the <part> elements in the MusicXML document.

  If no MIDI info is found for the part, use the default MIDI channel (0)
  and default to the Grand Piano program (MIDI Program #1).
  """

  def __init__(self, xml_score_part=None):
    self.id = ''
    self.part_name = ''
    self.midi_channel = DEFAULT_MIDI_CHANNEL
    self.midi_program = DEFAULT_MIDI_PROGRAM
    if xml_score_part is not None:
      self._parse(xml_score_part)

  def _parse(self, xml_score_part):
    """Parse the <score-part> element to an in-memory representation."""
    self.id = xml_score_part.attrib['id']

    if xml_score_part.find('part-name') is not None:
      self.part_name = xml_score_part.find('part-name').text or ''

    xml_midi_instrument = xml_score_part.find('midi-instrument')
    if (xml_midi_instrument is not None and
        xml_midi_instrument.find('midi-channel') is not None and
        xml_midi_instrument.find('midi-program') is not None):
      self.midi_channel = int(xml_midi_instrument.find('midi-channel').text)
      self.midi_program = int(xml_midi_instrument.find('midi-program').text)
    else:
      # If no MIDI info, use the default MIDI channel.
      self.midi_channel = DEFAULT_MIDI_CHANNEL
      # Use the default MIDI program
      self.midi_program = DEFAULT_MIDI_PROGRAM

  def __str__(self):
    score_str = 'ScorePart: ' + self.part_name
    score_str += ', Channel: ' + str(self.midi_channel)
    score_str += ', Program: ' + str(self.midi_program)
    return score_str
