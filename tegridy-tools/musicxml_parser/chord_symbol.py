import constants
from exception import ChordSymbolParseException


class ChordSymbol(object):
  """Internal representation of a MusicXML chord symbol <harmony> element.

  This represents a chord symbol with four components:

  1) Root: a string representing the chord root pitch class, e.g. "C#".
  2) Kind: a string representing the chord kind, e.g. "m7" for minor-seventh,
      "9" for dominant-ninth, or the empty string for major triad.
  3) Scale degree modifications: a list of strings representing scale degree
      modifications for the chord, e.g. "add9" to add an unaltered ninth scale
      degree (without the seventh), "b5" to flatten the fifth scale degree,
      "no3" to remove the third scale degree, etc.
  4) Bass: a string representing the chord bass pitch class, or None if the bass
      pitch class is the same as the root pitch class.

  There's also a special chord kind "N.C." representing no harmony, for which
  all other fields should be None.

  Use the `get_figure_string` method to get a string representation of the chord
  symbol as might appear in a lead sheet. This string representation is what we
  use to represent chord symbols in NoteSequence protos, as text annotations.
  While the MusicXML representation has more structure, using an unstructured
  string provides more flexibility and allows us to ingest chords from other
  sources, e.g. guitar tabs on the web.
  """

  # The below dictionary maps chord kinds to an abbreviated string as would
  # appear in a chord symbol in a standard lead sheet. There are often multiple
  # standard abbreviations for the same chord type, e.g. "+" and "aug" both
  # refer to an augmented chord, and "maj7", "M7", and a Delta character all
  # refer to a major-seventh chord; this dictionary attempts to be consistent
  # but the choice of abbreviation is somewhat arbitrary.
  #
  # The MusicXML-defined chord kinds are listed here:
  # http://usermanuals.musicxml.com/MusicXML/Content/ST-MusicXML-kind-value.htm

  CHORD_KIND_ABBREVIATIONS = {
    # These chord kinds are in the MusicXML spec.
    'major': '',
    'minor': 'm',
    'augmented': 'aug',
    'diminished': 'dim',
    'dominant': '7',
    'major-seventh': 'maj7',
    'minor-seventh': 'm7',
    'diminished-seventh': 'dim7',
    'augmented-seventh': 'aug7',
    'half-diminished': 'm7b5',
    'major-minor': 'm(maj7)',
    'major-sixth': '6',
    'minor-sixth': 'm6',
    'dominant-ninth': '9',
    'major-ninth': 'maj9',
    'minor-ninth': 'm9',
    'dominant-11th': '11',
    'major-11th': 'maj11',
    'minor-11th': 'm11',
    'dominant-13th': '13',
    'major-13th': 'maj13',
    'minor-13th': 'm13',
    'suspended-second': 'sus2',
    'suspended-fourth': 'sus',
    'pedal': 'ped',
    'power': '5',
    'none': 'N.C.',

    # These are not in the spec, but show up frequently in the wild.
    'dominant-seventh': '7',
    'augmented-ninth': 'aug9',
    'minor-major': 'm(maj7)',

    # Some abbreviated kinds also show up frequently in the wild.
    '': '',
    'min': 'm',
    'aug': 'aug',
    'dim': 'dim',
    '7': '7',
    'maj7': 'maj7',
    'min7': 'm7',
    'dim7': 'dim7',
    'm7b5': 'm7b5',
    'minMaj7': 'm(maj7)',
    '6': '6',
    'min6': 'm6',
    'maj69': '6(add9)',
    '9': '9',
    'maj9': 'maj9',
    'min9': 'm9',
    'sus47': 'sus7'
  }

  def __init__(self, xml_harmony, state):
    self.xml_harmony = xml_harmony
    self.time_position = -1
    self.xml_position = -1
    self.root = None
    self.kind = ''
    self.degrees = []
    self.bass = None
    self.state = state
    self._parse()

  def _alter_to_string(self, alter_text):
    """Parse alter text to a string of one or two sharps/flats.

    Args:
      alter_text: A string representation of an integer number of semitones.

    Returns:
      A string, one of 'bb', 'b', '#', '##', or the empty string.

    Raises:
      ChordSymbolParseException: If `alter_text` cannot be parsed to an integer,
          or if the integer is not a valid number of semitones between -2 and 2
          inclusive.
    """
    # Parse alter text to an integer number of semitones.
    try:
      alter_semitones = int(alter_text)
    except ValueError:
      raise ChordSymbolParseException('Non-integer alter: ' + str(alter_text))

    # Visual alter representation
    if alter_semitones == -2:
      alter_string = 'bb'
    elif alter_semitones == -1:
      alter_string = 'b'
    elif alter_semitones == 0:
      alter_string = ''
    elif alter_semitones == 1:
      alter_string = '#'
    elif alter_semitones == 2:
      alter_string = '##'
    else:
      raise ChordSymbolParseException('Invalid alter: ' + str(alter_semitones))

    return alter_string

  def _parse(self):
    """Parse the MusicXML <harmony> element."""
    self.time_position = self.state.time_position
    self.xml_position = self.state.xml_position
    for child in self.xml_harmony:
      if child.tag == 'root':
        self._parse_root(child)
      elif child.tag == 'kind':
        if child.text is None:
          # Seems like this shouldn't happen but frequently does in the wild...
          continue
        kind_text = str(child.text).strip()
        if kind_text not in self.CHORD_KIND_ABBREVIATIONS:
          raise ChordSymbolParseException('Unknown chord kind: ' + kind_text)
        self.kind = self.CHORD_KIND_ABBREVIATIONS[kind_text]
      elif child.tag == 'degree':
        self.degrees.append(self._parse_degree(child))
      elif child.tag == 'bass':
        self._parse_bass(child)
      elif child.tag == 'offset':
        # Offset tag moves chord symbol time position.
        try:
          offset = int(child.text)
        except ValueError:
          raise ChordSymbolParseException('Non-integer offset: ' +
                                          str(child.text))
        midi_ticks = offset * constants.STANDARD_PPQ / self.state.divisions
        seconds = (midi_ticks / constants.STANDARD_PPQ *
                   self.state.seconds_per_quarter)
        self.time_position += seconds
        self.xml_position += offset
      else:
        # Ignore other tag types because they are not relevant to mxp.
        pass

    if self.root is None and self.kind != 'N.C.':
      raise ChordSymbolParseException('Chord symbol must have a root')

  def _parse_pitch(self, xml_pitch, step_tag, alter_tag):
    """Parse and return the pitch-like <root> or <bass> element."""
    if xml_pitch.find(step_tag) is None:
      raise ChordSymbolParseException('Missing pitch step')
    step = xml_pitch.find(step_tag).text

    alter_string = ''
    if xml_pitch.find(alter_tag) is not None:
      alter_text = xml_pitch.find(alter_tag).text
      alter_string = self._alter_to_string(alter_text)

    if self.state.transpose:
      raise ChordSymbolParseException(
        'Transposition of chord symbols currently unsupported')

    return step + alter_string

  def _parse_root(self, xml_root):
    """Parse the <root> tag for a chord symbol."""
    self.root = self._parse_pitch(xml_root, step_tag='root-step',
                                  alter_tag='root-alter')

  def _parse_bass(self, xml_bass):
    """Parse the <bass> tag for a chord symbol."""
    self.bass = self._parse_pitch(xml_bass, step_tag='bass-step',
                                  alter_tag='bass-alter')

  def _parse_degree(self, xml_degree):
    """Parse and return the <degree> scale degree modification element."""
    if xml_degree.find('degree-value') is None:
      raise ChordSymbolParseException('Missing scale degree value in harmony')
    value_text = xml_degree.find('degree-value').text
    if value_text is None:
      raise ChordSymbolParseException('Missing scale degree')
    try:
      value = int(value_text)
    except ValueError:
      raise ChordSymbolParseException('Non-integer scale degree: ' +
                                      str(value_text))

    alter_string = ''
    if xml_degree.find('degree-alter') is not None:
      alter_text = xml_degree.find('degree-alter').text
      alter_string = self._alter_to_string(alter_text)

    if xml_degree.find('degree-type') is None:
      raise ChordSymbolParseException('Missing degree modification type')
    type_text = xml_degree.find('degree-type').text

    if type_text == 'add':
      if not alter_string:
        # When adding unaltered scale degree, use "add" string.
        type_string = 'add'
      else:
        # When adding altered scale degree, "add" not necessary.
        type_string = ''
    elif type_text == 'subtract':
      type_string = 'no'
      # Alter should be irrelevant when removing scale degree.
      alter_string = ''
    elif type_text == 'alter':
      if not alter_string:
        raise ChordSymbolParseException('Degree alteration by zero semitones')
      # No type string necessary as merely appending e.g. "#9" suffices.
      type_string = ''
    else:
      raise ChordSymbolParseException('Invalid degree modification type: ' +
                                      str(type_text))

    # Return a scale degree modification string that can be appended to a chord
    # symbol figure string.
    return type_string + alter_string + str(value)

  def __str__(self):
    if self.kind == 'N.C.':
      note_string = '{kind: ' + self.kind + '} '
    else:
      note_string = '{root: ' + self.root
      note_string += ', kind: ' + self.kind
      note_string += ', degrees: [%s]' % ', '.join(degree
                                                   for degree in self.degrees)
      note_string += ', bass: ' + self.bass + '} '
    note_string += '(@time: ' + str(self.time_position) + ')'
    return note_string

  def get_figure_string(self):
    """Return a chord symbol figure string."""
    if self.kind == 'N.C.':
      return self.kind
    else:
      degrees_string = ''.join('(%s)' % degree for degree in self.degrees)
      figure = self.root + self.kind + degrees_string
      if self.bass:
        figure += '/' + self.bass
      return figure
