import six

DEFAULT_MIDI_PROGRAM = 0  # Default MIDI Program (0 = grand piano)
DEFAULT_MIDI_CHANNEL = 0  # Default MIDI Channel (0 = first channel)
MUSICXML_MIME_TYPE = 'application/vnd.recordare.musicxml+xml'


class MusicXMLParseException(Exception):
  """Exception thrown when the MusicXML contents cannot be parsed."""
  pass


class PitchStepParseException(MusicXMLParseException):
  """Exception thrown when a pitch step cannot be parsed.

  Will happen if pitch step is not one of A, B, C, D, E, F, or G
  """
  pass


class ChordSymbolParseException(MusicXMLParseException):
  """Exception thrown when a chord symbol cannot be parsed."""
  pass


class MultipleTimeSignatureException(MusicXMLParseException):
  """Exception thrown when multiple time signatures found in a measure."""
  pass


class AlternatingTimeSignatureException(MusicXMLParseException):
  """Exception thrown when an alternating time signature is encountered."""
  pass


class TimeSignatureParseException(MusicXMLParseException):
  """Exception thrown when the time signature could not be parsed."""
  pass


class UnpitchedNoteException(MusicXMLParseException):
  """Exception thrown when an unpitched note is encountered.

  We do not currently support parsing files with unpitched notes (e.g.,
  percussion scores).

  http://www.musicxml.com/tutorial/percussion/unpitched-notes/
  """
  pass


class KeyParseException(MusicXMLParseException):
  """Exception thrown when a key signature cannot be parsed."""
  pass


class InvalidNoteDurationTypeException(MusicXMLParseException):
  """Exception thrown when a note's duration type is invalid."""
  pass
