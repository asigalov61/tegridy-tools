"""MusicXML parser.
"""
from fractions import Fraction
import xml.etree.ElementTree as ET
import zipfile
import math
from exception import MusicXMLParseException, MultipleTimeSignatureException

import six
import constants

from measure import Measure
from tempo import Tempo
from key_signature import KeySignature
from score_part import ScorePart
from part import Part
from playable_notes import get_playable_notes

DEFAULT_MIDI_PROGRAM = 0  # Default MIDI Program (0 = grand piano)
DEFAULT_MIDI_CHANNEL = 0  # Default MIDI Channel (0 = first channel)
MUSICXML_MIME_TYPE = 'application/vnd.recordare.musicxml+xml'


class MusicXMLParserState(object):
  """Maintains internal state of the MusicXML parser."""

  def __init__(self):
    # Default to one division per measure
    # From the MusicXML documentation: "The divisions element indicates
    # how many divisions per quarter note are used to indicate a note's
    # duration. For example, if duration = 1 and divisions = 2,
    # this is an eighth note duration."
    self.divisions = 1

    # Default to a tempo of 120 quarter notes per minute
    # MusicXML calls this tempo, but mxp calls this qpm
    # Therefore, the variable is called qpm, but reads the
    # MusicXML tempo attribute
    # (120 qpm is the default tempo according to the
    # Standard MIDI Files 1.0 Specification)
    self.qpm = 120

    # Duration of a single quarter note in seconds
    self.seconds_per_quarter = 0.5

    # Running total of time for the current event in seconds.
    # Resets to 0 on every part. Affected by <forward> and <backup> elements
    self.time_position = 0
    self.xml_position = 0

    # Default to a MIDI velocity of 64 (mf)
    self.velocity = 64

    # Default MIDI program (0 = grand piano)
    self.midi_program = DEFAULT_MIDI_PROGRAM

    # Current MIDI channel (usually equal to the part number)
    self.midi_channel = DEFAULT_MIDI_CHANNEL

    # Keep track of previous note to get chord timing correct
    # This variable stores an instance of the Note class (defined below)
    self.previous_note_duration = 0
    self.previous_note_time_position = 0
    self.previous_note_xml_position = 0

    # Keep track of previous direction
    # self.previous_direction = None

    # Keep track of current transposition level in +/- semitones.
    self.transpose = 0

    # Keep track of current time signature. Does not support polymeter.
    self.time_signature = None

    # Keep track of previous (unsolved) grace notes
    self.previous_grace_notes = []

    # Keep track of chord index
    self.chord_index = 0

    # Keep track of measure number
    self.measure_number = 0

    # Keep track of unsolved ending bracket
    self.first_ending_discontinue = False

    # Keep track of beam status
    self.is_beam_start = False
    self.is_beam_continue = False
    self.is_beam_stop = False



class MusicXMLDocument(object):
  """Internal representation of a MusicXML Document.

  Represents the top level object which holds the MusicXML document
  Responsible for loading the .xml or .mxl file using the _get_score method
  If the file is .mxl, this class uncompresses it

  After the file is loaded, this class then parses the document into memory
  using the parse method.
  """

  def __init__(self, filename):
    self._score = self._get_score(filename)
    self.parts = []
    # ScoreParts indexed by id.
    self._score_parts = {}
    self.midi_resolution = constants.STANDARD_PPQ
    self._state = MusicXMLParserState()
    # Total time in seconds
    self.total_time_secs = 0
    self.total_time_duration = 0
    self._parse()
    self._recalculate_time_position()

  @staticmethod
  def _get_score(filename):
    """Given a MusicXML file, return the score as an xml.etree.ElementTree.

    Given a MusicXML file, return the score as an xml.etree.ElementTree
    If the file is compress (ends in .mxl), uncompress it first

    Args:
        filename: The path of a MusicXML file

    Returns:
      The score as an xml.etree.ElementTree.

    Raises:
      MusicXMLParseException: if the file cannot be parsed.
    """
    score = None
    if not isinstance(filename, str):
      filename = str(filename)
    if filename.endswith('.mxl'):
      # Compressed MXL file. Uncompress in memory.
      try:
        mxlzip = zipfile.ZipFile(filename)
      except zipfile.BadZipfile as exception:
        raise MusicXMLParseException(exception)

      # A compressed MXL file may contain multiple files, but only one
      # MusicXML file. Read the META-INF/container.xml file inside of the
      # MXL file to locate the MusicXML file within the MXL file
      # http://www.musicxml.com/tutorial/compressed-mxl-files/zip-archive-structure/

      # Raise a MusicXMLParseException if multiple MusicXML files found

      infolist = mxlzip.infolist()
      if six.PY3:
        # In py3, instead of returning raw bytes, ZipFile.infolist() tries to
        # guess the filenames' encoding based on file headers, and decodes using
        # this encoding in order to return a list of strings. If the utf-8
        # header is missing, it decodes using the DOS code page 437 encoding
        # which is almost definitely wrong. Here we need to explicitly check
        # for when this has occurred and change the encoding to utf-8.
        # https://stackoverflow.com/questions/37723505/namelist-from-zipfile-returns-strings-with-an-invalid-encoding
        zip_filename_utf8_flag = 0x800
        for info in infolist:
          if info.flag_bits & zip_filename_utf8_flag == 0:
            filename_bytes = info.filename.encode('437')
            filename = filename_bytes.decode('utf-8', 'replace')
            info.filename = filename

      container_file = [x for x in infolist
                        if x.filename == 'META-INF/container.xml']
      compressed_file_name = ''

      if container_file:
        try:
          container = ET.fromstring(mxlzip.read(container_file[0]))
          for rootfile_tag in container.findall('./rootfiles/rootfile'):
            if 'media-type' in rootfile_tag.attrib:
              if rootfile_tag.attrib['media-type'] == MUSICXML_MIME_TYPE:
                if not compressed_file_name:
                  compressed_file_name = rootfile_tag.attrib['full-path']
                else:
                  raise MusicXMLParseException(
                    'Multiple MusicXML files found in compressed archive')
            else:
              # No media-type attribute, so assume this is the MusicXML file
              if not compressed_file_name:
                compressed_file_name = rootfile_tag.attrib['full-path']
              else:
                raise MusicXMLParseException(
                  'Multiple MusicXML files found in compressed archive')
        except ET.ParseError as exception:
          raise MusicXMLParseException(exception)

      if not compressed_file_name:
        raise MusicXMLParseException(
          'Unable to locate main .xml file in compressed archive.')
      if six.PY2:
        # In py2, the filenames in infolist are utf-8 encoded, so
        # we encode the compressed_file_name as well in order to
        # be able to lookup compressed_file_info below.
        compressed_file_name = compressed_file_name.encode('utf-8')
      try:
        compressed_file_info = [x for x in infolist
                                if x.filename == compressed_file_name][0]
      except IndexError:
        raise MusicXMLParseException(
          'Score file %s not found in zip archive' % compressed_file_name)
      score_string = mxlzip.read(compressed_file_info)
      try:
        score = ET.fromstring(score_string)
      except ET.ParseError as exception:
        raise MusicXMLParseException(exception)
    else:
      # Uncompressed XML file.
      try:
        tree = ET.parse(filename)
        score = tree.getroot()
      except ET.ParseError as exception:
        raise MusicXMLParseException(exception)

    return score

  def _parse(self):
    """Parse the uncompressed MusicXML document."""
    # Parse part-list
    xml_part_list = self._score.find('part-list')
    if xml_part_list is not None:
      for element in xml_part_list:
        if element.tag == 'score-part':
          score_part = ScorePart(element)
          self._score_parts[score_part.id] = score_part

    # Parse parts
    for score_part_index, child in enumerate(self._score.findall('part')):
      part = Part(child, self._score_parts, self._state)
      self.parts.append(part)
      score_part_index += 1
      if self._state.time_position > self.total_time_secs:
        self.total_time_secs = self._state.time_position
      if self._state.xml_position > self.total_time_duration:
        self.total_time_duration = self._state.xml_position

  def _recalculate_time_position(self):
    """ Sometimes, the tempo marking is not located in the first voice.
    Therefore, the time position of each object should be calculate after parsing the entire tempo objects.

    """
    tempos = self.get_tempos()

    tempos.sort(key=lambda x: x.xml_position)
    if tempos[0].xml_position != 0:
      default_tempo = Tempo(self._state)
      default_tempo.xml_position = 0
      default_tempo.time_position = 0
      default_tempo.qpm = constants.DEFAULT_QUARTERS_PER_MINUTE
      default_tempo.state.divisions = tempos[0].state.divisions
      tempos.insert(0, default_tempo)
    new_time_position = 0
    for i in range(len(tempos)):
      tempos[i].time_position = new_time_position
      if i + 1 < len(tempos):
        new_time_position += (tempos[i + 1].xml_position - tempos[i].xml_position) / tempos[i].qpm * 60 / tempos[
          i].state.divisions

    for part in self.parts:
      for measure in part.measures:
        for note in measure.notes:
          for i in range(len(tempos)):
            if i + 1 == len(tempos):
              current_tempo = tempos[i].qpm / 60 * tempos[i].state.divisions
              break
            else:
              if tempos[i].xml_position <= note.note_duration.xml_position and tempos[
                i + 1].xml_position > note.note_duration.xml_position:
                current_tempo = tempos[i].qpm / 60 * tempos[i].state.divisions
                break
          note.note_duration.time_position = tempos[i].time_position + (
                  note.note_duration.xml_position - tempos[i].xml_position) / current_tempo
          note.note_duration.seconds = note.note_duration.duration / current_tempo

  def get_chord_symbols(self):
    """Return a list of all the chord symbols used in this score."""
    chord_symbols = []
    for part in self.parts:
      for measure in part.measures:
        for chord_symbol in measure.chord_symbols:
          if chord_symbol not in chord_symbols:
            # Prevent duplicate chord symbols
            chord_symbols.append(chord_symbol)
    return chord_symbols

  def get_time_signatures(self):
    """Return a list of all the time signatures used in this score.

    Does not support polymeter (i.e. assumes all parts have the same
    time signature, such as Part 1 having a time signature of 6/8
    while Part 2 has a simultaneous time signature of 2/4).

    Ignores duplicate time signatures to prevent mxp duplicate
    time signature error. This happens when multiple parts have the
    same time signature is used in multiple parts at the same time.

    Example: If Part 1 has a time siganture of 4/4 and Part 2 also
    has a time signature of 4/4, then only instance of 4/4 is sent
    to mxp.

    Returns:
      A list of all TimeSignature objects used in this score.
    """
    time_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.time_signature is not None:
          if measure.time_signature not in time_signatures:
            # Prevent duplicate time signatures
            time_signatures.append(measure.time_signature)

    return time_signatures

  def get_key_signatures(self):
    """Return a list of all the key signatures used in this score.

    Support different key signatures in different parts (score in
    written pitch).

    Ignores duplicate key signatures to prevent mxp duplicate key
    signature error. This happens when multiple parts have the same
    key signature at the same time.

    Example: If the score is in written pitch and the
    flute is written in the key of Bb major, the trombone will also be
    written in the key of Bb major. However, the clarinet and trumpet
    will be written in the key of C major because they are Bb transposing
    instruments.

    If no key signatures are found, create a default key signature of
    C major.

    Returns:
      A list of all KeySignature objects used in this score.
    """
    key_signatures = []
    for part in self.parts:
      for measure in part.measures:
        if measure.key_signature is not None:
          if measure.key_signature not in key_signatures:
            # Prevent duplicate key signatures
            key_signatures.append(measure.key_signature)

    if not key_signatures:
      # If there are no key signatures, add C major at the beginning
      key_signature = KeySignature(self._state)
      key_signature.time_position = 0
      key_signature.xml_position = 0
      key_signatures.append(key_signature)

    return key_signatures

  def get_tempos(self):
    """Return a list of all tempos in this score.

    If no tempos are found, create a default tempo of 120 qpm.

    Returns:
      A list of all Tempo objects used in this score.
    """
    tempos = []

    if self.parts:
      part = self.parts[0]  # Use only first part
      for measure in part.measures:
        for tempo in measure.tempos:
          tempos.append(tempo)

    # If no tempos, add a default of 120 at beginning
    if not tempos:
      tempo = Tempo(self._state)
      tempo.qpm = self._state.qpm
      tempo.time_position = 0
      tempo.xml_position = 0
      tempos.append(tempo)

    return tempos

  def recalculate_time_position(self):
    tempos = self.get_tempos()

    tempos.sort(key=lambda x: x.xml_position)
    new_time_position = 0
    for i in range(len(tempos)):
      tempos[i].time_position = new_time_position
      if i + 1 < len(tempos):
        new_time_position += (tempos[i + 1].xml_position - tempos[i].xml_position) / tempos[i].qpm * 60 / tempos[
          i].state.divisions

    for part in self.parts:
      for measure in part.measures:
        for note in measure.notes:
          for i in range(len(tempos)):
            if i + 1 == len(tempos):
              current_tempo = tempos[i].qpm / 60 * tempos[i].state.divisions
              break
            else:
              if tempos[i].xml_position <= note.note_duration.xml_position and tempos[
                i + 1].xml_position > note.note_duration.xml_position:
                current_tempo = tempos[i].qpm / 60 * tempos[i].state.divisions
                break
          note.note_duration.time_position = tempos[i].time_position + (
                note.note_duration.xml_position - tempos[i].xml_position) / current_tempo
          note.note_duration.seconds = note.note_duration.duration / current_tempo
  def get_measure_positions(self):
    part = self.parts[0]
    measure_positions = []

    for measure in part.measures:
      measure_positions.append(measure.start_xml_position)

    return measure_positions


  def get_notes(self, melody_only=False, grace_note=True):
    notes = []
    rests = []
    num_parts = len(self.parts)
    for instrument_index in range(num_parts):
      part = self.parts[instrument_index]

      notes_part, rests_part = get_playable_notes(part)
      notes.extend(notes_part)
      rests.extend(rests_part)
    notes.sort(key=lambda x: (x.note_duration.xml_position,
              x.note_duration.grace_order, -x.pitch[1]))
    return notes, rests


  def find(self, f, seq):
    items_list = []
    for item in seq:
      if f(item):
        items_list.append(item)
    return items_list

  def rearrange_chord_index(self, xml_notes):
    # assert all(xml_notes[i].pitch[1] >= xml_notes[i + 1].pitch[1] for i in range(len(xml_notes) - 1)
    #            if xml_notes[i].note_duration.xml_position ==xml_notes[i+1].note_duration.xml_position)

    previous_position = [-1]
    max_chord_index = [0]
    for note in xml_notes:
      voice = note.voice - 1
      while voice >= len(previous_position):
        previous_position.append(-1)
        max_chord_index.append(0)
      if note.note_duration.is_grace_note:
        continue
      if note.staff == 1:
        if note.note_duration.xml_position > previous_position[voice]:
          previous_position[voice] = note.note_duration.xml_position
          max_chord_index[voice] = note.chord_index
          note.chord_index = 0
        else:
          note.chord_index = (max_chord_index[voice] - note.chord_index)
      else:  # note staff ==2
        pass

    return xml_notes

  def get_directions(self):
    directions = []
    for part in self.parts:
        for measure in part.measures:
            for direction in measure.directions:
                directions.append(direction)

    directions.sort(key=lambda x: x.xml_position)
    cleaned_direction = []
    for i in range(len(directions)):
        dir = directions[i]
        if not dir.type == None:
            if dir.type['type'] == "none":
                for j in range(i):
                    prev_dir = directions[i-j-1]
                    if 'number' in prev_dir.type.keys():
                        prev_key = prev_dir.type['type']
                        prev_num = prev_dir.type['number']
                    else:
                        continue
                    if prev_num == dir.type['number']:
                        if prev_key == "crescendo":
                            dir.type['type'] = 'crescendo'
                            break
                        elif prev_key == "diminuendo":
                            dir.type['type'] = 'diminuendo'
                            break
            cleaned_direction.append(dir)
        else:
            print(vars(dir.xml_direction))

    return cleaned_direction

  def get_beat_positions(self, in_measure_level=False):
    piano = self.parts[0]
    num_measure = len(piano.measures)
    time_signatures = self.get_time_signatures()
    time_sig_position = [time.xml_position for time in time_signatures]
    beat_piece = []
    for i in range(num_measure):
      measure = piano.measures[i]
      measure_start = measure.start_xml_position
      corresp_time_sig_idx = self.binary_index(time_sig_position, measure_start)
      corresp_time_sig = time_signatures[corresp_time_sig_idx]
      # corresp_time_sig = measure.time_signature
      full_measure_length = corresp_time_sig.state.divisions * corresp_time_sig.numerator / corresp_time_sig.denominator * 4
      if i < num_measure - 1:
        actual_measure_length = piano.measures[i + 1].start_xml_position - measure_start
      else:
        actual_measure_length = full_measure_length

      # if i +1 < num_measure:
      #     measure_length = piano.measures[i+1].start_xml_position - measure_start
      # else:
      #     measure_length = measure_start - piano.measures[i-1].start_xml_position

      num_beat_in_measure = corresp_time_sig.numerator
      if in_measure_level:
        num_beat_in_measure = 1
      elif num_beat_in_measure == 6:
        num_beat_in_measure = 2
      elif num_beat_in_measure == 9:
        num_beat_in_measure = 3
      elif num_beat_in_measure == 12:
        num_beat_in_measure = 4
      elif num_beat_in_measure == 18:
        num_beat_in_measure = 3
      elif num_beat_in_measure == 24:
        num_beat_in_measure = 4
      inter_beat_interval = full_measure_length / num_beat_in_measure
      if actual_measure_length != full_measure_length:
        measure.implicit = True
      else:
        measure.implicit = False

      if measure.implicit:
        length_ratio = actual_measure_length / full_measure_length
        minimum_beat = 1 / num_beat_in_measure
        num_beat_in_measure = int(math.ceil(length_ratio / minimum_beat))
        if i == 0:
          for j in range(-num_beat_in_measure, 0):
            beat = piano.measures[i + 1].start_xml_position + j * inter_beat_interval
            if len(beat_piece) > 0 and beat > beat_piece[-1]:
              beat_piece.append(beat)
            elif len(beat_piece) == 0:
              beat_piece.append(beat)
        else:
          for j in range(0, num_beat_in_measure):
            beat = piano.measures[i].start_xml_position + j * inter_beat_interval
            if beat > beat_piece[-1]:
              beat_piece.append(beat)
      else:
        for j in range(num_beat_in_measure):
          beat = measure_start + j * inter_beat_interval
          beat_piece.append(beat)
        #
      # for note in measure.notes:
      #     note.on_beat = check_note_on_beat(note, measure_start, measure_length)
    return beat_piece

  def get_interval_positions(self, interval_in_16th):
    piano = self.parts[0]
    num_measure = len(piano.measures)
    time_signatures = self.get_time_signatures()
    time_sig_position = [time.xml_position for time in time_signatures]
    beat_piece = []
    for i in range(num_measure):
      measure = piano.measures[i]
      measure_start = measure.start_xml_position
      corresp_time_sig_idx = self.binary_index(time_sig_position, measure_start)
      corresp_time_sig = time_signatures[corresp_time_sig_idx]

      num_beat_in_measure = int(corresp_time_sig.numerator * 16 / (corresp_time_sig.denominator * interval_in_16th))
      if num_beat_in_measure == 0:
        raise Exception(f'measure is longer than a interval: {corresp_time_sig.numerator} / {corresp_time_sig.denominator}')
      inter_beat_interval = corresp_time_sig.state.divisions * interval_in_16th / 4 

      full_measure_length = corresp_time_sig.state.divisions * corresp_time_sig.numerator / corresp_time_sig.denominator * 4
      if i < num_measure - 1:
        actual_measure_length = piano.measures[i + 1].start_xml_position - measure_start
      else:
        actual_measure_length = full_measure_length

      if actual_measure_length != full_measure_length:
        measure.implicit = True
      else:
        measure.implicit = False

      if measure.implicit:
        length_ratio = actual_measure_length / full_measure_length
        minimum_beat = 1 / num_beat_in_measure
        num_beat_in_measure = int(math.ceil(length_ratio / minimum_beat))
        if i == 0:
          for j in range(-num_beat_in_measure, 0):
            beat = piano.measures[i + 1].start_xml_position + j * inter_beat_interval
            if len(beat_piece) > 0 and beat > beat_piece[-1]:
              beat_piece.append(beat)
            elif len(beat_piece) == 0:
              beat_piece.append(beat)
        else:
          for j in range(0, num_beat_in_measure):
            beat = piano.measures[i].start_xml_position + j * inter_beat_interval
            if beat > beat_piece[-1]:
              beat_piece.append(beat)
      else:
        for j in range(num_beat_in_measure):
          beat = measure_start + j * inter_beat_interval
          beat_piece.append(beat)

    return beat_piece
  def get_accidentals(self):
    directions = []
    accs = ['#', '♭', '♮']
    # accs = ' # ♭ ♮ '
    for part in self.parts:
      for measure in part.measures:
        for direction in measure.directions:
          if direction.type['type'] == 'words' and direction.type['content'] in accs:
            directions.append(direction)
    return directions

  def binary_index(self, alist, item):
    # better to move : utils.py
    first = 0
    last = len(alist)-1
    midpoint = 0

    if(item< alist[first]):
        return 0

    while first<last:
        midpoint = (first + last)//2
        currentElement = alist[midpoint]

        if currentElement < item:
            if alist[midpoint+1] > item:
                return midpoint
            else: first = midpoint +1
            if first == last and alist[last] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint -1
        else:
            if midpoint +1 ==len(alist):
                return midpoint
            while alist[midpoint+1] == item:
                midpoint += 1
                if midpoint + 1 == len(alist):
                    return midpoint
            return midpoint
    return last
