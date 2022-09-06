from calendar import c
import copy

class Direction(object):
  """Internal representation of a MusicXML Measure's Direction properties.
  
  This represents musical dynamic symbols, expressions with six components:
  1) dynamic               # 'ppp', 'pp', 'p', 'mp' 'mf', 'f', 'ff' 'fff
  2) pedal                 # 'start' or 'stop' or 'change' 'continue' or None
  3) tempo                 # integer
  4) wedge                 # 'crescendo' or 'diminuendo' or 'stop' or None
  5) words                 # string e.g)  Andantino
  6) velocity              # integer
  7) octave-shift
  8) metronome

  It parses the standard of the marking point of note.
  """
  def __init__(self, xml_direction, state):
    self.xml_direction = xml_direction
    self.type = {'type': None, 'content': None}
    self.state = copy.copy(state)
    self.placement = None
    self.staff = None
    self.time_position = state.time_position
    self.xml_position = state.xml_position
    self._parse()


  def _parse(self):
    """Parse the MusicXML <direction> element."""
    direction = self.xml_direction
    child_list = list(direction.find('direction-type'))
    if len(child_list) == 0:
      return
    staff = direction.find('staff')
    if staff is not None:
      self.staff = staff.text
    if 'placement' in direction.attrib.keys():
      self.placement = direction.attrib['placement']
    for child in child_list:
      if child is not None:
        if child.tag == "dynamics":
          self._parse_dynamics(child)
        elif child.tag == "pedal":
          self._parse_pedal(child)
        elif child.tag == "wedge":
          self._parse_wedge(child)
        elif child.tag == "words" or child.tag=="other-dynamics":
          self._parse_words(child)
        elif child.tag=='octave-shift':
          self._parse_octave_shift(child)
        elif child.tag=='metronome':
          self._parse_metronome(child)


  def _parse_pedal(self, xml_pedal):
    """Parse the MusicXML <pedal> element.
    
    Args:
      xml_pedal: XML element with tag type 'pedal'.
    """
    pedal = xml_pedal.attrib['type']
    self.type = {'type': 'pedal', 'content': pedal}

  def _parse_sound(self, xml_direction):
    """Parse the MusicXML <sound> element.
    
    Args:
      xml_direction: XML element with tag type 'direction'.
    """
    sound_tag = xml_direction.find('sound')
    if sound_tag is not None:
      attrib = sound_tag.attrib
      if 'dynamics' in attrib:
        velocity = attrib['dynamics']
        self.type = {'type':'velocity', 'content': velocity}

      elif 'tempo' in attrib:
        tempo = attrib['tempo']
        self.type = {'type':'tempo', 'content': tempo}

  def _parse_dynamics(self, xml_dynamics):
    """Parse the MusicXML <dynamics> element.

    Args:
      xml_dynamics: XML element with tag type 'dynamics'.
    """
    dynamic = list(xml_dynamics)[0].tag
    if dynamic == 'other-dynamics':
      content = list(xml_dynamics)[0].text
      if content:
        while '<sym>dynamicPiano</sym>' in content:
          content = content.replace('<sym>dynamicPiano</sym>', 'p')
        while '<sym>dynamicForte</sym>' in content:
          content = content.replace('<sym>dynamicForte</sym>', 'f')
        while '<sym>dynamicMezzo</sym>' in content:
          content = content.replace('<sym>dynamicMezzo</sym>', 'm')
        while '<sym>dynamicSforzando</sym>' in content:
          content = content.replace('<sym>dynamicSforzando</sym>', 'sf')
        while '<sym>dynamicRinforzando</sym>' in content:
          content = content.replace('<sym>dynamicRinforzando</sym>', 'r')
        while '<sym>dynamicNiente</sym>' in content:
          content = content.replace('<sym>dynamicNiente</sym>', 'n')
        while '<sym>dynamicZ</sym>' in content:
          content = content.replace('<sym>dynamicZ</sym>', 'z')
      if content is not None:
        self.type = {'type':'words', 'content': content}
    else:
      self.type = {'type':'dynamic', 'content': dynamic}

  def _parse_wedge(self, xml_wedge):
    """Parse the MusicXML <wedge> element.
    
    Args:
      xml_wedge: XML element with tag type 'wedge'.
    """
    wedge_type_labels = ['crescendo', 'diminuendo']
    wedge_type = xml_wedge.attrib['type']
    if 'number' in xml_wedge.attrib.keys():
      wedge_index = xml_wedge.attrib['number']
    else:
      wedge_index = None

    if wedge_type in wedge_type_labels:
      # Add "start" at the point of a wedge starting point
      self.type = {'type':wedge_type, 'content': 'start', 'number': wedge_index}

    else:
      # if wedge_type == 'stop':
      #   if self.state.previous_direction.type['type'] is not None:
      #     previous_type = list(self.state.previous_direction.type['type'])[0]
      #
      #     if previous_type in wedge_type_labels:
      #       self.type = {'type':previous_type, 'content': wedge_type, 'number': wedge_index}
      #   else:
      self.type = {'type':'none', 'content': wedge_type, 'number': wedge_index}

  def _parse_words(self, xml_words):
    """Parse the MusicXML <words> element.
    
    Args:
      xml_wedge: XML element with tag type 'wedge'.
    """
    # self.type = {'type':'words', 'content': xml_words.text.decode('utf-8')}
    if self.type['content'] is None:
      self.type = {'type': 'words', 'content': xml_words.text}
    else:
      self.type['content'] += xml_words.text

  def _parse_octave_shift(self, xml_shift):
    """Parse the MusicXML <octave-shift> element.

    """
    self.type = {'type': 'octave-shift', 'content': xml_shift.attrib['type'], 'size':  xml_shift.attrib['size']}

  def _parse_metronome(self, xml_metronome):
    """Parse the MusicXML <metronome> element.

    """
    self.type = {'type':'metronome', 'content': xml_metronome.find('per-minute'), 'beat-unit': xml_metronome.find('beat-unit')}

  def __str__(self):
    direction_string = '{type: ' + str(self.type['type']) + ' - ' + str(self.type['content'].encode('utf-8'))
    direction_string += ', xml_position: ' + str(self.xml_position)
    direction_string += ', staff: ' + str(self.staff) + '}'
    return direction_string


