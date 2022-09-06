from measure import Measure
from score_part import ScorePart
import xml.etree.ElementTree as ET
import copy



class Part(object):
  """Internal represention of a MusicXML <part> element."""

  def __init__(self, xml_part, score_parts, state):
    self.id = ''
    self.score_part = None
    self.measures = []
    self._state = state
    self._parse(xml_part, score_parts)

  def _parse(self, xml_part, score_parts):
    """Parse the <part> element."""
    if 'id' in xml_part.attrib:
      self.id = xml_part.attrib['id']
    if self.id in score_parts:
      self.score_part = score_parts[self.id]
    else:
      # If this part references a score-part id that was not found in the file,
      # construct a default score-part.
      self.score_part = ScorePart()

    # Reset the time position when parsing each part
    self._state.time_position = 0
    self._state.xml_position = 0
    self._state.midi_channel = self.score_part.midi_channel
    self._state.midi_program = self.score_part.midi_program
    self._state.transpose = 0

    xml_measures = xml_part.findall('measure')
    measure_length = len(xml_measures)
    current_measure_number = 0
    segno_measure = None
    previous_forward_repeats = []
    resolved_repeats = []
    resolved_first_ending = []
    end_measure_of_first_ending = []
    fine_activated = False

    while current_measure_number < measure_length:
      measure = xml_measures[current_measure_number]

      self._repair_empty_measure(measure)
      self._state.measure_number = current_measure_number
      old_state = copy.copy(self._state)
      parsed_measure = Measure(measure, self._state)

      if parsed_measure.first_ending_start:
        if current_measure_number in resolved_first_ending:
          ending_index = resolved_first_ending.index(current_measure_number)
          current_measure_number = end_measure_of_first_ending[ending_index] + 1
          self._state = old_state
          continue
        else:
          resolved_first_ending.append(current_measure_number)

      self.measures.append(parsed_measure)

      if parsed_measure.first_ending_stop:
        end_measure_of_first_ending.append(current_measure_number)

      if parsed_measure.repeat == 'start':
        previous_forward_repeats.append(current_measure_number)
        current_measure_number += 1
      elif parsed_measure.repeat == 'jump' and current_measure_number not in resolved_repeats:
        resolved_repeats.append(current_measure_number)
        if len(resolved_first_ending) != len(end_measure_of_first_ending):
          end_measure_of_first_ending.append(current_measure_number)
        if len(previous_forward_repeats) == 0:
          current_measure_number = 0
        else:
          current_measure_number = previous_forward_repeats[-1]
          del previous_forward_repeats[-1]
      elif parsed_measure.dacapo == 'jump':
        current_measure_number = 0
        fine_activated = True
      elif parsed_measure.fine and fine_activated:
        break
      else:
        current_measure_number += 1


    #
    # for (measure_number, measure) in enumerate(xml_measures):
    #   # Issue #674: Repair measures that do not contain notes
    #   # by inserting a whole measure rest
    #   self._repair_empty_measure(measure)
    #   self._state.measure_number = measure_number
    #   parsed_measure = Measure(measure, self._state)
    #   self.measures.append(parsed_measure)

  def _repair_empty_measure(self, measure):
    """Repair a measure if it is empty by inserting a whole measure rest.

    If a <measure> only consists of a <forward> element that advances
    the time cursor, remove the <forward> element and replace
    with a whole measure rest of the same duration.

    Args:
      measure: The measure to repair.
    """
    # Issue #674 - If the <forward> element is in a measure without
    # any <note> elements, treat it as if it were a whole measure
    # rest by inserting a rest of that duration
    forward_count = len(measure.findall('forward'))
    note_count = len(measure.findall('note'))
    if note_count == 0 and forward_count == 1:
      # Get the duration of the <forward> element
      xml_forward = measure.find('forward')
      xml_duration = xml_forward.find('duration')
      forward_duration = int(xml_duration.text)

      # Delete the <forward> element
      measure.remove(xml_forward)

      # Insert the new note
      new_note = '<note>'
      new_note += '<rest /><duration>' + str(forward_duration) + '</duration>'
      new_note += '<voice>1</voice><type>whole</type><staff>1</staff>'
      new_note += '</note>'
      new_note_xml = ET.fromstring(new_note)
      measure.append(new_note_xml)

  def __str__(self):
    part_str = 'Part: ' + self.score_part.part_name
    return part_str
