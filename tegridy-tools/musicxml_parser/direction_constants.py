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


"""

DIRECTION_TYPES = [
  'dynamic',
  'pedal',
  'tempo',
  'wedge',
  'words',
  'velocity',
  'octave-shift',
  'metronome'
]
