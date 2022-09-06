from fractions import Fraction
import xml.etree.ElementTree as ET
import zipfile
from exception import UnpitchedNoteException, PitchStepParseException


class Notations(object):
  """Internal representation of a MusicXML Note's Notations properties.
  
  This represents musical notations symbols, articulationsNo with ten components:

  1) accent
  2) arpeggiate
  3) fermata
  4) mordent
  5) staccato
  6) tenuto
  7) tie
  8) tied
  9) trill
  10) tuplet
  11) cue (small note)

  """

  def __init__(self, xml_notations=None):
    self.xml_notations = xml_notations
    self.is_accent = False
    self.is_arpeggiate = False
    self.is_fermata = False
    self.is_mordent = False
    self.is_staccato = False
    self.is_tenuto = False
    self.tie = None  # 'start' or 'stop' or None
    self.tied_start = False
    self.tied_stop = False
    self.is_trill = False
    self.is_tuplet = False
    self.is_strong_accent = False
    self.is_cue = False
    self.is_beam_start = False
    self.is_beam_continue = False
    self.is_beam_stop = False
    self.wavy_line = None
    self.slurs = []
    self.is_slur_start = False
    self.is_slur_stop = False
    self.is_slur_continue = False
    self.is_slash = False

  def parse_notations(self, xml_notations):
    """Parse the MusicXML <Notations> element."""
    self.xml_notations = xml_notations
    if self.xml_notations is not None:
      notations = list(self.xml_notations)
      for child in notations:
        if child.tag == 'articulations':
          self._parse_articulations(child)
        elif child.tag == 'arpeggiate':
          self.is_arpeggiate = True
        elif child.tag == 'fermata':
          self.is_fermata = True
        elif child.tag == 'tie':
          self.tie = child.attrib['type']
        elif child.tag == 'tied':
          if child.attrib['type'] == 'start':
            self.tied_start = True
          elif child.attrib['type'] == 'stop':
            self.tied_stop = True
        elif child.tag == 'ornaments':
          self._parse_ornaments(child)
        elif child.tag == 'slur':
          self._parse_slur(child)

  def _parse_articulations(self, xml_articulation):
    """Parse the MusicXML <Articulations> element.

    Args:
      xml_articulation: XML element with tag type 'articulation'.
    """
    tag = list(xml_articulation)[0].tag
    if tag == 'arpeggiate':
      self.is_arpeggiate = True
    elif tag == 'accent':
      self.is_accent = True
    elif tag == 'fermata':
      self.is_fermata = True
    elif tag == 'staccato':
      self.is_staccato = True
    elif tag == 'tenuto':
      self.is_tenuto = True
    elif tag == 'tuplet':
      self.is_tuplet = True
    elif tag == 'strong-accent':
      self.is_strong_accent = True

  def _parse_ornaments(self, xml_ornaments):
    """Parse the MusicXML <ornaments> element.

    Args:
      xml_ornaments: XML element with tag type 'ornaments'.
    """
    children = list(xml_ornaments)
    for child in children:
      tag = child.tag
      if tag == 'trill-mark':
        self.is_trill = True
      if tag == 'inverted-mordent' or tag == 'mordent':
        self.is_mordent = True
      if tag == 'wavy-line':
        type = child.attrib['type']
        if 'number' in child.attrib:
          number = child.attrib['number']
        else:
          number = 1
        self.wavy_line = WavyLine(type, number)

  def _parse_slur(self, xml_slurs):
    type = xml_slurs.attrib['type']
    if 'number' in xml_slurs.attrib:
      number = xml_slurs.attrib['number']
    else:
      number=1
    self.slurs.append(Slur(type, number))


class WavyLine:
  def __init__(self, type, number):
    self.type = type  # start or stop
    self.number = number
    self.xml_position = 0
    self.end_xml_position = 0
    self.pitch = 0

class Slur:
  def __init__(self, type, number):
    self.type = type  # start or stop
    self.number = number
    self.xml_position = 0
    self.end_xml_position = 0
    self.index = 0
    self.voice = 0

