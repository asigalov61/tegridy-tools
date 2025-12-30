#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#
#	Tegridy MIDI X Module (TMIDI X / tee-midi eks)
#
#   NOTE: TMIDI X Module starts after the partial MIDI.py module @ line 1450
#
#	Based upon MIDI.py module v.6.7. by Peter Billam / pjb.com.au
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
###################################################################################
###################################################################################
#
#	PARTIAL MIDI.py Module v.6.7. by Peter Billam
#   Please see TMIDI 2.3/tegridy-tools repo for full MIDI.py module code
# 
#   Or you can always download the latest full version from:
#
#   https://pjb.com.au/
#   https://peterbillam.gitlab.io/miditools/
#	
#	Copyright 2020 Peter Billam
#
###################################################################################
###################################################################################
'''

###################################################################################

__version__ = "25.12.29"

print('=' * 70)
print('TMIDIX Python module')
print('Version:', __version__)
print('=' * 70)
print('Loading module...')

###################################################################################

import sys, struct, copy

Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4
_no_warning = False

#------------------------------- Encoding stuff --------------------------

def opus2midi(opus=[], text_encoding='ISO-8859-1'):
    r'''The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of midi-events, and each event is itself a list; see above.
opus2midi() returns a bytestring of the MIDI, which can then be
written either to a file opened in binary mode (mode='wb'),
or to stdout by means of:   sys.stdout.buffer.write()

my_opus = [
    96, 
    [   # track 0:
        ['patch_change', 0, 1, 8],   # and these are the events...
        ['note_on',   5, 1, 25, 96],
        ['note_off', 96, 1, 25, 0],
        ['note_on',   0, 1, 29, 96],
        ['note_off', 96, 1, 29, 0],
    ],   # end of track 0
]
my_midi = opus2midi(my_opus)
sys.stdout.buffer.write(my_midi)
'''
    if len(opus) < 2:
        opus=[1000, [],]
    tracks = copy.deepcopy(opus)
    ticks = int(tracks.pop(0))
    ntracks = len(tracks)
    if ntracks == 1:
        format = 0
    else:
        format = 1

    my_midi = b"MThd\x00\x00\x00\x06"+struct.pack('>HHH',format,ntracks,ticks)
    for track in tracks:
        events = _encode(track, text_encoding=text_encoding)
        my_midi += b'MTrk' + struct.pack('>I',len(events)) + events
    _clean_up_warnings()
    return my_midi


def score2opus(score=None, text_encoding='ISO-8859-1'):
    r'''
The argument is a list: the first item in the list is the "ticks"
parameter, the others are the tracks. Each track is a list
of score-events, and each event is itself a list.  A score-event
is similar to an opus-event (see above), except that in a score:
 1) the times are expressed as an absolute number of ticks
    from the track's start time
 2) the pairs of 'note_on' and 'note_off' events in an "opus"
    are abstracted into a single 'note' event in a "score":
    ['note', start_time, duration, channel, pitch, velocity]
score2opus() returns a list specifying the equivalent "opus".

my_score = [
    96,
    [   # track 0:
        ['patch_change', 0, 1, 8],
        ['note', 5, 96, 1, 25, 96],
        ['note', 101, 96, 1, 29, 96]
    ],   # end of track 0
]
my_opus = score2opus(my_score)
'''
    if len(score) < 2:
        score=[1000, [],]
    tracks = copy.deepcopy(score)
    ticks = int(tracks.pop(0))
    opus_tracks = []
    for scoretrack in tracks:
        time2events = dict([])
        for scoreevent in scoretrack:
            if scoreevent[0] == 'note':
                note_on_event = ['note_on',scoreevent[1],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                note_off_event = ['note_off',scoreevent[1]+scoreevent[2],
                 scoreevent[3],scoreevent[4],scoreevent[5]]
                if time2events.get(note_on_event[1]):
                   time2events[note_on_event[1]].append(note_on_event)
                else:
                   time2events[note_on_event[1]] = [note_on_event,]
                if time2events.get(note_off_event[1]):
                   time2events[note_off_event[1]].append(note_off_event)
                else:
                   time2events[note_off_event[1]] = [note_off_event,]
                continue
            if time2events.get(scoreevent[1]):
               time2events[scoreevent[1]].append(scoreevent)
            else:
               time2events[scoreevent[1]] = [scoreevent,]

        sorted_times = []  # list of keys
        for k in time2events.keys():
            sorted_times.append(k)
        sorted_times.sort()

        sorted_events = []  # once-flattened list of values sorted by key
        for time in sorted_times:
            sorted_events.extend(time2events[time])

        abs_time = 0
        for event in sorted_events:  # convert abs times => delta times
            delta_time = event[1] - abs_time
            abs_time = event[1]
            event[1] = delta_time
        opus_tracks.append(sorted_events)
    opus_tracks.insert(0,ticks)
    _clean_up_warnings()
    return opus_tracks

def score2midi(score=None, text_encoding='ISO-8859-1'):
    r'''
Translates a "score" into MIDI, using score2opus() then opus2midi()
'''
    return opus2midi(score2opus(score, text_encoding), text_encoding)

#--------------------------- Decoding stuff ------------------------

def midi2opus(midi=b'', do_not_check_MIDI_signature=False):
    r'''Translates MIDI into a "opus".  For a description of the
"opus" format, see opus2midi()
'''
    my_midi=bytearray(midi)
    if len(my_midi) < 4:
        _clean_up_warnings()
        return [1000,[],]
    id = bytes(my_midi[0:4])
    if id != b'MThd':
        _warn("midi2opus: midi starts with "+str(id)+" instead of 'MThd'")
        _clean_up_warnings()
        if do_not_check_MIDI_signature == False:
          return [1000,[],]
    [length, format, tracks_expected, ticks] = struct.unpack(
     '>IHHH', bytes(my_midi[4:14]))
    if length != 6:
        _warn("midi2opus: midi header length was "+str(length)+" instead of 6")
        _clean_up_warnings()
        return [1000,[],]
    my_opus = [ticks,]
    my_midi = my_midi[14:]
    track_num = 1   # 5.1
    while len(my_midi) >= 8:
        track_type   = bytes(my_midi[0:4])
        if track_type != b'MTrk':
            #_warn('midi2opus: Warning: track #'+str(track_num)+' type is '+str(track_type)+" instead of b'MTrk'")
            pass
        [track_length] = struct.unpack('>I', my_midi[4:8])
        my_midi = my_midi[8:]
        if track_length > len(my_midi):
            _warn('midi2opus: track #'+str(track_num)+' length '+str(track_length)+' is too large')
            _clean_up_warnings()
            return my_opus   # 5.0
        my_midi_track = my_midi[0:track_length]
        my_track = _decode(my_midi_track)
        my_opus.append(my_track)
        my_midi = my_midi[track_length:]
        track_num += 1   # 5.1
    _clean_up_warnings()
    return my_opus

def opus2score(opus=[]):
    r'''For a description of the "opus" and "score" formats,
see opus2midi() and score2opus().
'''
    if len(opus) < 2:
        _clean_up_warnings()
        return [1000,[],]
    tracks = copy.deepcopy(opus)  # couple of slices probably quicker...
    ticks = int(tracks.pop(0))
    score = [ticks,]
    for opus_track in tracks:
        ticks_so_far = 0
        score_track = []
        chapitch2note_on_events = dict([])   # 4.0
        for opus_event in opus_track:
            ticks_so_far += opus_event[1]
            if opus_event[0] == 'note_off' or (opus_event[0] == 'note_on' and opus_event[4] == 0):  # 4.8
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                if chapitch2note_on_events.get(key):
                    new_event = chapitch2note_on_events[key].pop(0)
                    new_event[2] = ticks_so_far - new_event[1]
                    score_track.append(new_event)
                elif pitch > 127:
                    pass #_warn('opus2score: note_off with no note_on, bad pitch='+str(pitch))
                else:
                    pass #_warn('opus2score: note_off with no note_on cha='+str(cha)+' pitch='+str(pitch))
            elif opus_event[0] == 'note_on':
                cha = opus_event[2]
                pitch = opus_event[3]
                key = cha*128 + pitch
                new_event = ['note',ticks_so_far,0,cha,pitch, opus_event[4]]
                if chapitch2note_on_events.get(key):
                    chapitch2note_on_events[key].append(new_event)
                else:
                    chapitch2note_on_events[key] = [new_event,]
            else:
                opus_event[1] = ticks_so_far
                score_track.append(opus_event)
        # check for unterminated notes (Oisín) -- 5.2
        for chapitch in chapitch2note_on_events:
            note_on_events = chapitch2note_on_events[chapitch]
            for new_e in note_on_events:
                new_e[2] = ticks_so_far - new_e[1]
                score_track.append(new_e)
                pass #_warn("opus2score: note_on with no note_off cha="+str(new_e[3])+' pitch='+str(new_e[4])+'; adding note_off at end')
        score.append(score_track)
    _clean_up_warnings()
    return score

def midi2score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score", using midi2opus() then opus2score()
'''
    return opus2score(midi2opus(midi, do_not_check_MIDI_signature))

def midi2ms_score(midi=b'', do_not_check_MIDI_signature=False):
    r'''
Translates MIDI into a "score" with one beat per second and one
tick per millisecond, using midi2opus() then to_millisecs()
then opus2score()
'''
    return opus2score(to_millisecs(midi2opus(midi, do_not_check_MIDI_signature)))

def midi2single_track_ms_score(midi_path_or_bytes, 
                                recalculate_channels = False, 
                                pass_old_timings_events= False, 
                                verbose = False, 
                                do_not_check_MIDI_signature=False
                                ):
    r'''
Translates MIDI into a single track "score" with 16 instruments and one beat per second and one
tick per millisecond
'''

    if type(midi_path_or_bytes) == bytes:
      midi_data = midi_path_or_bytes

    elif type(midi_path_or_bytes) == str:
      midi_data = open(midi_path_or_bytes, 'rb').read() 

    score = midi2score(midi_data, do_not_check_MIDI_signature)

    if recalculate_channels:

      events_matrixes = []

      itrack = 1
      events_matrixes_channels = []
      while itrack < len(score):
          events_matrix = []
          for event in score[itrack]:
              if event[0] == 'note' and event[3] != 9:
                event[3] = (16 * (itrack-1)) + event[3]
                if event[3] not in events_matrixes_channels:
                  events_matrixes_channels.append(event[3])

              events_matrix.append(event)
          events_matrixes.append(events_matrix)
          itrack += 1

      events_matrix1 = []
      for e in events_matrixes:
        events_matrix1.extend(e)

      if verbose:
        if len(events_matrixes_channels) > 16:
          print('MIDI has', len(events_matrixes_channels), 'instruments!', len(events_matrixes_channels) - 16, 'instrument(s) will be removed!')

      for e in events_matrix1:
        if e[0] == 'note' and e[3] != 9:
          if e[3] in events_matrixes_channels[:15]:
            if events_matrixes_channels[:15].index(e[3]) < 9:
              e[3] = events_matrixes_channels[:15].index(e[3])
            else:
              e[3] = events_matrixes_channels[:15].index(e[3])+1
          else:
            events_matrix1.remove(e)
        
        if e[0] in ['patch_change', 'control_change', 'channel_after_touch', 'key_after_touch', 'pitch_wheel_change'] and e[2] != 9:
          if e[2] in [e % 16 for e in events_matrixes_channels[:15]]:
            if [e % 16 for e in events_matrixes_channels[:15]].index(e[2]) < 9:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])
            else:
              e[2] = [e % 16 for e in events_matrixes_channels[:15]].index(e[2])+1
          else:
            events_matrix1.remove(e)
    
    else:
      events_matrix1 = []
      itrack = 1
     
      while itrack < len(score):
          for event in score[itrack]:
            events_matrix1.append(event)
          itrack += 1    

    opus = score2opus([score[0], events_matrix1])
    ms_score = opus2score(to_millisecs(opus, pass_old_timings_events=pass_old_timings_events))

    return ms_score

#------------------------ Other Transformations ---------------------

def to_millisecs(old_opus=None, desired_time_in_ms=1, pass_old_timings_events = False):
    r'''Recallibrates all the times in an "opus" to use one beat
per second and one tick per millisecond.  This makes it
hard to retrieve any information about beats or barlines,
but it does make it easy to mix different scores together.
'''
    if old_opus == None:
        return [1000 * desired_time_in_ms,[],]
    try:
        old_tpq  = int(old_opus[0])
    except IndexError:   # 5.0
        _warn('to_millisecs: the opus '+str(type(old_opus))+' has no elements')
        return [1000 * desired_time_in_ms,[],]
    new_opus = [1000 * desired_time_in_ms,]
    # 6.7 first go through building a table of set_tempos by absolute-tick
    ticks2tempo = {}
    itrack = 1
    while itrack < len(old_opus):
        ticks_so_far = 0
        for old_event in old_opus[itrack]:
            if old_event[0] == 'note':
                raise TypeError('to_millisecs needs an opus, not a score')
            ticks_so_far += old_event[1]
            if old_event[0] == 'set_tempo':
                ticks2tempo[ticks_so_far] = old_event[2]
        itrack += 1
    # then get the sorted-array of their keys
    tempo_ticks = []  # list of keys
    for k in ticks2tempo.keys():
        tempo_ticks.append(k)
    tempo_ticks.sort()
    # then go through converting to millisec, testing if the next
    # set_tempo lies before the next track-event, and using it if so.
    itrack = 1
    while itrack < len(old_opus):
        ms_per_old_tick = 400 / old_tpq  # float: will round later 6.3
        i_tempo_ticks = 0
        ticks_so_far = 0
        ms_so_far = 0.0
        previous_ms_so_far = 0.0

        if pass_old_timings_events:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],['old_tpq', 0, old_tpq]]  # new "crochet" is 1 sec
        else:
          new_track = [['set_tempo',0,1000000 * desired_time_in_ms],]  # new "crochet" is 1 sec
        for old_event in old_opus[itrack]:
            # detect if ticks2tempo has something before this event
            # 20160702 if ticks2tempo is at the same time, leave it
            event_delta_ticks = old_event[1] * desired_time_in_ms
            if (i_tempo_ticks < len(tempo_ticks) and
              tempo_ticks[i_tempo_ticks] < (ticks_so_far + old_event[1]) * desired_time_in_ms):
                delta_ticks = tempo_ticks[i_tempo_ticks] - ticks_so_far
                ms_so_far += (ms_per_old_tick * delta_ticks * desired_time_in_ms)
                ticks_so_far = tempo_ticks[i_tempo_ticks]
                ms_per_old_tick = ticks2tempo[ticks_so_far] / (1000.0*old_tpq * desired_time_in_ms)
                i_tempo_ticks += 1
                event_delta_ticks -= delta_ticks
            new_event = copy.deepcopy(old_event)  # now handle the new event
            ms_so_far += (ms_per_old_tick * old_event[1] * desired_time_in_ms)
            new_event[1] = round(ms_so_far - previous_ms_so_far)

            if pass_old_timings_events:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
              else:
                  new_event[0] = 'old_set_tempo'
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            else:
              if old_event[0] != 'set_tempo':
                  previous_ms_so_far = ms_so_far
                  new_track.append(new_event)
            ticks_so_far += event_delta_ticks
        new_opus.append(new_track)
        itrack += 1
    _clean_up_warnings()
    return new_opus

def event2alsaseq(event=None):   # 5.5
    r'''Converts an event into the format needed by the alsaseq module,
http://pp.com.mx/python/alsaseq
The type of track (opus or score) is autodetected.
'''
    pass

def grep(score=None, channels=None):
    r'''Returns a "score" containing only the channels specified
'''
    if score == None:
        return [1000,[],]
    ticks = score[0]
    new_score = [ticks,]
    if channels == None:
        return new_score
    channels = set(channels)
    global Event2channelindex
    itrack = 1
    while itrack < len(score):
        new_score.append([])
        for event in score[itrack]:
            channel_index = Event2channelindex.get(event[0], False)
            if channel_index:
                if event[channel_index] in channels:
                    new_score[itrack].append(event)
            else:
                new_score[itrack].append(event)
        itrack += 1
    return new_score

def score2stats(opus_or_score=None):
    r'''Returns a dict of some basic stats about the score, like
bank_select (list of tuples (msb,lsb)),
channels_by_track (list of lists), channels_total (set),
general_midi_mode (list),
ntracks, nticks, patch_changes_by_track (list of dicts),
num_notes_by_channel (list of numbers),
patch_changes_total (set),
percussion (dict histogram of channel 9 events),
pitches (dict histogram of pitches on channels other than 9),
pitch_range_by_track (list, by track, of two-member-tuples),
pitch_range_sum (sum over tracks of the pitch_ranges),
'''
    bank_select_msb = -1
    bank_select_lsb = -1
    bank_select = []
    channels_by_track = []
    channels_total    = set([])
    general_midi_mode = []
    num_notes_by_channel = dict([])
    patches_used_by_track  = []
    patches_used_total     = set([])
    patch_changes_by_track = []
    patch_changes_total    = set([])
    percussion = dict([]) # histogram of channel 9 "pitches"
    pitches    = dict([]) # histogram of pitch-occurrences channels 0-8,10-15
    pitch_range_sum = 0   # u pitch-ranges of each track
    pitch_range_by_track = []
    is_a_score = True
    if opus_or_score == None:
        return {'bank_select':[], 'channels_by_track':[], 'channels_total':[],
         'general_midi_mode':[], 'ntracks':0, 'nticks':0,
         'num_notes_by_channel':dict([]),
         'patch_changes_by_track':[], 'patch_changes_total':[],
         'percussion':{}, 'pitches':{}, 'pitch_range_by_track':[],
         'ticks_per_quarter':0, 'pitch_range_sum':0}
    ticks_per_quarter = opus_or_score[0]
    i = 1   # ignore first element, which is ticks
    nticks = 0
    while i < len(opus_or_score):
        highest_pitch = 0
        lowest_pitch = 128
        channels_this_track = set([])
        patch_changes_this_track = dict({})
        for event in opus_or_score[i]:
            if event[0] == 'note':
                num_notes_by_channel[event[3]] = num_notes_by_channel.get(event[3],0) + 1
                if event[3] == 9:
                    percussion[event[4]] = percussion.get(event[4],0) + 1
                else:
                    pitches[event[4]]    = pitches.get(event[4],0) + 1
                    if event[4] > highest_pitch:
                        highest_pitch = event[4]
                    if event[4] < lowest_pitch:
                        lowest_pitch = event[4]
                channels_this_track.add(event[3])
                channels_total.add(event[3])
                finish_time = event[1] + event[2]
                if finish_time > nticks:
                    nticks = finish_time
            elif event[0] == 'note_off' or (event[0] == 'note_on' and event[4] == 0):  # 4.8
                finish_time = event[1]
                if finish_time > nticks:
                    nticks = finish_time
            elif event[0] == 'note_on':
                is_a_score = False
                num_notes_by_channel[event[2]] = num_notes_by_channel.get(event[2],0) + 1
                if event[2] == 9:
                    percussion[event[3]] = percussion.get(event[3],0) + 1
                else:
                    pitches[event[3]]    = pitches.get(event[3],0) + 1
                    if event[3] > highest_pitch:
                        highest_pitch = event[3]
                    if event[3] < lowest_pitch:
                        lowest_pitch = event[3]
                channels_this_track.add(event[2])
                channels_total.add(event[2])
            elif event[0] == 'patch_change':
                patch_changes_this_track[event[2]] = event[3]
                patch_changes_total.add(event[3])
            elif event[0] == 'control_change':
                if event[3] == 0:  # bank select MSB
                    bank_select_msb = event[4]
                elif event[3] == 32:  # bank select LSB
                    bank_select_lsb = event[4]
                if bank_select_msb >= 0 and bank_select_lsb >= 0:
                    bank_select.append((bank_select_msb,bank_select_lsb))
                    bank_select_msb = -1
                    bank_select_lsb = -1
            elif event[0] == 'sysex_f0':
                if _sysex2midimode.get(event[2], -1) >= 0:
                    general_midi_mode.append(_sysex2midimode.get(event[2]))
            if is_a_score:
                if event[1] > nticks:
                    nticks = event[1]
            else:
                nticks += event[1]
        if lowest_pitch == 128:
            lowest_pitch = 0
        channels_by_track.append(channels_this_track)
        patch_changes_by_track.append(patch_changes_this_track)
        pitch_range_by_track.append((lowest_pitch,highest_pitch))
        pitch_range_sum += (highest_pitch-lowest_pitch)
        i += 1

    return {'bank_select':bank_select,
            'channels_by_track':channels_by_track,
            'channels_total':channels_total,
            'general_midi_mode':general_midi_mode,
            'ntracks':len(opus_or_score)-1,
            'nticks':nticks,
            'num_notes_by_channel':num_notes_by_channel,
            'patch_changes_by_track':patch_changes_by_track,
            'patch_changes_total':patch_changes_total,
            'percussion':percussion,
            'pitches':pitches,
            'pitch_range_by_track':pitch_range_by_track,
            'pitch_range_sum':pitch_range_sum,
            'ticks_per_quarter':ticks_per_quarter}

#----------------------------- Event stuff --------------------------

_sysex2midimode = {
    "\x7E\x7F\x09\x01\xF7": 1,
    "\x7E\x7F\x09\x02\xF7": 0,
    "\x7E\x7F\x09\x03\xF7": 2,
}

# Some public-access tuples:
MIDI_events = tuple('''note_off note_on key_after_touch
control_change patch_change channel_after_touch
pitch_wheel_change'''.split())

Text_events = tuple('''text_event copyright_text_event
track_name instrument_name lyric marker cue_point text_event_08
text_event_09 text_event_0a text_event_0b text_event_0c
text_event_0d text_event_0e text_event_0f'''.split())

Nontext_meta_events = tuple('''end_track set_tempo
smpte_offset time_signature key_signature sequencer_specific
raw_meta_event sysex_f0 sysex_f7 song_position song_select
tune_request'''.split())
# unsupported: raw_data

# Actually, 'tune_request' is is F-series event, not strictly a meta-event...
Meta_events = Text_events + Nontext_meta_events
All_events  = MIDI_events + Meta_events

# And three dictionaries:
Number2patch = {   # General MIDI patch numbers:
0:'Acoustic Grand',
1:'Bright Acoustic',
2:'Electric Grand',
3:'Honky-Tonk',
4:'Electric Piano 1',
5:'Electric Piano 2',
6:'Harpsichord',
7:'Clav',
8:'Celesta',
9:'Glockenspiel',
10:'Music Box',
11:'Vibraphone',
12:'Marimba',
13:'Xylophone',
14:'Tubular Bells',
15:'Dulcimer',
16:'Drawbar Organ',
17:'Percussive Organ',
18:'Rock Organ',
19:'Church Organ',
20:'Reed Organ',
21:'Accordion',
22:'Harmonica',
23:'Tango Accordion',
24:'Acoustic Guitar(nylon)',
25:'Acoustic Guitar(steel)',
26:'Electric Guitar(jazz)',
27:'Electric Guitar(clean)',
28:'Electric Guitar(muted)',
29:'Overdriven Guitar',
30:'Distortion Guitar',
31:'Guitar Harmonics',
32:'Acoustic Bass',
33:'Electric Bass(finger)',
34:'Electric Bass(pick)',
35:'Fretless Bass',
36:'Slap Bass 1',
37:'Slap Bass 2',
38:'Synth Bass 1',
39:'Synth Bass 2',
40:'Violin',
41:'Viola',
42:'Cello',
43:'Contrabass',
44:'Tremolo Strings',
45:'Pizzicato Strings',
46:'Orchestral Harp',
47:'Timpani',
48:'String Ensemble 1',
49:'String Ensemble 2',
50:'SynthStrings 1',
51:'SynthStrings 2',
52:'Choir Aahs',
53:'Voice Oohs',
54:'Synth Voice',
55:'Orchestra Hit',
56:'Trumpet',
57:'Trombone',
58:'Tuba',
59:'Muted Trumpet',
60:'French Horn',
61:'Brass Section',
62:'SynthBrass 1',
63:'SynthBrass 2',
64:'Soprano Sax',
65:'Alto Sax',
66:'Tenor Sax',
67:'Baritone Sax',
68:'Oboe',
69:'English Horn',
70:'Bassoon',
71:'Clarinet',
72:'Piccolo',
73:'Flute',
74:'Recorder',
75:'Pan Flute',
76:'Blown Bottle',
77:'Skakuhachi',
78:'Whistle',
79:'Ocarina',
80:'Lead 1 (square)',
81:'Lead 2 (sawtooth)',
82:'Lead 3 (calliope)',
83:'Lead 4 (chiff)',
84:'Lead 5 (charang)',
85:'Lead 6 (voice)',
86:'Lead 7 (fifths)',
87:'Lead 8 (bass+lead)',
88:'Pad 1 (new age)',
89:'Pad 2 (warm)',
90:'Pad 3 (polysynth)',
91:'Pad 4 (choir)',
92:'Pad 5 (bowed)',
93:'Pad 6 (metallic)',
94:'Pad 7 (halo)',
95:'Pad 8 (sweep)',
96:'FX 1 (rain)',
97:'FX 2 (soundtrack)',
98:'FX 3 (crystal)',
99:'FX 4 (atmosphere)',
100:'FX 5 (brightness)',
101:'FX 6 (goblins)',
102:'FX 7 (echoes)',
103:'FX 8 (sci-fi)',
104:'Sitar',
105:'Banjo',
106:'Shamisen',
107:'Koto',
108:'Kalimba',
109:'Bagpipe',
110:'Fiddle',
111:'Shanai',
112:'Tinkle Bell',
113:'Agogo',
114:'Steel Drums',
115:'Woodblock',
116:'Taiko Drum',
117:'Melodic Tom',
118:'Synth Drum',
119:'Reverse Cymbal',
120:'Guitar Fret Noise',
121:'Breath Noise',
122:'Seashore',
123:'Bird Tweet',
124:'Telephone Ring',
125:'Helicopter',
126:'Applause',
127:'Gunshot',
}
Notenum2percussion = {   # General MIDI Percussion (on Channel 9):
35:'Acoustic Bass Drum',
36:'Bass Drum 1',
37:'Side Stick',
38:'Acoustic Snare',
39:'Hand Clap',
40:'Electric Snare',
41:'Low Floor Tom',
42:'Closed Hi-Hat',
43:'High Floor Tom',
44:'Pedal Hi-Hat',
45:'Low Tom',
46:'Open Hi-Hat',
47:'Low-Mid Tom',
48:'Hi-Mid Tom',
49:'Crash Cymbal 1',
50:'High Tom',
51:'Ride Cymbal 1',
52:'Chinese Cymbal',
53:'Ride Bell',
54:'Tambourine',
55:'Splash Cymbal',
56:'Cowbell',
57:'Crash Cymbal 2',
58:'Vibraslap',
59:'Ride Cymbal 2',
60:'Hi Bongo',
61:'Low Bongo',
62:'Mute Hi Conga',
63:'Open Hi Conga',
64:'Low Conga',
65:'High Timbale',
66:'Low Timbale',
67:'High Agogo',
68:'Low Agogo',
69:'Cabasa',
70:'Maracas',
71:'Short Whistle',
72:'Long Whistle',
73:'Short Guiro',
74:'Long Guiro',
75:'Claves',
76:'Hi Wood Block',
77:'Low Wood Block',
78:'Mute Cuica',
79:'Open Cuica',
80:'Mute Triangle',
81:'Open Triangle',
}

Event2channelindex = { 'note':3, 'note_off':2, 'note_on':2,
 'key_after_touch':2, 'control_change':2, 'patch_change':2,
 'channel_after_touch':2, 'pitch_wheel_change':2
}

################################################################
# The code below this line is full of frightening things, all to
# do with the actual encoding and decoding of binary MIDI data.

def _twobytes2int(byte_a):
    r'''decode a 16 bit quantity from two bytes,'''
    return (byte_a[1] | (byte_a[0] << 8))

def _int2twobytes(int_16bit):
    r'''encode a 16 bit quantity into two bytes,'''
    return bytes([(int_16bit>>8) & 0xFF, int_16bit & 0xFF])

def _read_14_bit(byte_a):
    r'''decode a 14 bit quantity from two bytes,'''
    return (byte_a[0] | (byte_a[1] << 7))

def _write_14_bit(int_14bit):
    r'''encode a 14 bit quantity into two bytes,'''
    return bytes([int_14bit & 0x7F, (int_14bit>>7) & 0x7F])

def _ber_compressed_int(integer):
    r'''BER compressed integer (not an ASN.1 BER, see perlpacktut for
details).  Its bytes represent an unsigned integer in base 128,
most significant digit first, with as few digits as possible.
Bit eight (the high bit) is set on each byte except the last.
'''
    ber = bytearray(b'')
    seven_bits = 0x7F & integer
    ber.insert(0, seven_bits)  # XXX surely should convert to a char ?
    integer >>= 7
    while integer > 0:
        seven_bits = 0x7F & integer
        ber.insert(0, 0x80|seven_bits)  # XXX surely should convert to a char ?
        integer >>= 7
    return ber

def _unshift_ber_int(ba):
    r'''Given a bytearray, returns a tuple of (the ber-integer at the
start, and the remainder of the bytearray).
'''
    if not len(ba):  # 6.7
        _warn('_unshift_ber_int: no integer found')
        return ((0, b""))
    byte = ba[0]
    ba = ba[1:]
    integer = 0
    while True:
        integer += (byte & 0x7F)
        if not (byte & 0x80):
            return ((integer, ba))
        if not len(ba):
            _warn('_unshift_ber_int: no end-of-integer found')
            return ((0, ba))
        byte = ba[0]
        ba = ba[1:]
        integer <<= 7


def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if _previous_times > 1:
        # E:1176, 0: invalid syntax (<string>, line 1176) (syntax-error) ???
        # print('  previous message repeated '+str(_previous_times)+' times', file=sys.stderr)
        # 6.7
        sys.stderr.write('  previous message repeated {0} times\n'.format(_previous_times))
    elif _previous_times > 0:
        sys.stderr.write('  previous message repeated\n')
    _previous_times = 0
    _previous_warning = ''


def _warn(s=''):
    if _no_warning:
        return
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s) + "\n")
        _previous_warning = s


def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    if str(type(text)).find("'str'") >= 0:  # 6.4 test for back-compatibility
        data = bytes(text, encoding=text_encoding)
    else:
        data = bytes(text)
    return b'\xFF' + bytes((which_kind,)) + _ber_compressed_int(len(data)) + data


def _consistentise_ticks(scores):  # 3.6
    # used by mix_scores, merge_scores, concatenate_scores
    if len(scores) == 1:
        return copy.deepcopy(scores)
    are_consistent = True
    ticks = scores[0][0]
    iscore = 1
    while iscore < len(scores):
        if scores[iscore][0] != ticks:
            are_consistent = False
            break
        iscore += 1
    if are_consistent:
        return copy.deepcopy(scores)
    new_scores = []
    iscore = 0
    while iscore < len(scores):
        score = scores[iscore]
        new_scores.append(opus2score(to_millisecs(score2opus(score))))
        iscore += 1
    return new_scores


###########################################################################
def _decode(trackdata=b'', exclude=None, include=None,
            event_callback=None, exclusive_event_callback=None, no_eot_magic=False):
    r'''Decodes MIDI track data into an opus-style list of events.
The options:
  'exclude' is a list of event types which will be ignored SHOULD BE A SET
  'include' (and no exclude), makes exclude a list
       of all possible events, /minus/ what include specifies
  'event_callback' is a coderef
  'exclusive_event_callback' is a coderef
'''
    trackdata = bytearray(trackdata)
    if exclude == None:
        exclude = []
    if include == None:
        include = []
    if include and not exclude:
        exclude = All_events
    include = set(include)
    exclude = set(exclude)

    # Pointer = 0;  not used here; we eat through the bytearray instead.
    event_code = -1;  # used for running status
    event_count = 0;
    events = []

    while (len(trackdata)):
        # loop while there's anything to analyze ...
        eot = False  # When True, the event registrar aborts this loop
        event_count += 1

        E = []
        # E for events - we'll feed it to the event registrar at the end.

        # Slice off the delta time code, and analyze it
        [time, trackdata] = _unshift_ber_int(trackdata)

        # Now let's see what we can make of the command
        first_byte = trackdata[0] & 0xFF
        trackdata = trackdata[1:]
        if (first_byte < 0xF0):  # It's a MIDI event
            if (first_byte & 0x80):
                event_code = first_byte
            else:
                # It wants running status; use last event_code value
                trackdata.insert(0, first_byte)
                if (event_code == -1):
                    _warn("Running status not set; Aborting track.")
                    return []

            command = event_code & 0xF0
            channel = event_code & 0x0F

            if (command == 0xF6):  # 0-byte argument
                pass
            elif (command == 0xC0 or command == 0xD0):  # 1-byte argument
                parameter = trackdata[0]  # could be B
                trackdata = trackdata[1:]
            else:  # 2-byte argument could be BB or 14-bit
                parameter = (trackdata[0], trackdata[1])
                trackdata = trackdata[2:]

            #################################################################
            # MIDI events

            if (command == 0x80):
                if 'note_off' in exclude:
                    continue
                E = ['note_off', time, channel, parameter[0], parameter[1]]
            elif (command == 0x90):
                if 'note_on' in exclude:
                    continue
                E = ['note_on', time, channel, parameter[0], parameter[1]]
            elif (command == 0xA0):
                if 'key_after_touch' in exclude:
                    continue
                E = ['key_after_touch', time, channel, parameter[0], parameter[1]]
            elif (command == 0xB0):
                if 'control_change' in exclude:
                    continue
                E = ['control_change', time, channel, parameter[0], parameter[1]]
            elif (command == 0xC0):
                if 'patch_change' in exclude:
                    continue
                E = ['patch_change', time, channel, parameter]
            elif (command == 0xD0):
                if 'channel_after_touch' in exclude:
                    continue
                E = ['channel_after_touch', time, channel, parameter]
            elif (command == 0xE0):
                if 'pitch_wheel_change' in exclude:
                    continue
                E = ['pitch_wheel_change', time, channel,
                     _read_14_bit(parameter) - 0x2000]
            else:
                _warn("Shouldn't get here; command=" + hex(command))

        elif (first_byte == 0xFF):  # It's a Meta-Event! ##################
            # [command, length, remainder] =
            #    unpack("xCwa*", substr(trackdata, $Pointer, 6));
            # Pointer += 6 - len(remainder);
            #    # Move past JUST the length-encoded.
            command = trackdata[0] & 0xFF
            trackdata = trackdata[1:]
            [length, trackdata] = _unshift_ber_int(trackdata)
            if (command == 0x00):
                if (length == 2):
                    E = ['set_sequence_number', time, _twobytes2int(trackdata)]
                else:
                    _warn('set_sequence_number: length must be 2, not ' + str(length))
                    E = ['set_sequence_number', time, 0]

            elif command >= 0x01 and command <= 0x0f:  # Text events
                # 6.2 take it in bytes; let the user get the right encoding.
                # text_str = trackdata[0:length].decode('ascii','ignore')
                # text_str = trackdata[0:length].decode('ISO-8859-1')
                # 6.4 take it in bytes; let the user get the right encoding.
                text_data = bytes(trackdata[0:length])  # 6.4
                # Defined text events
                if (command == 0x01):
                    E = ['text_event', time, text_data]
                elif (command == 0x02):
                    E = ['copyright_text_event', time, text_data]
                elif (command == 0x03):
                    E = ['track_name', time, text_data]
                elif (command == 0x04):
                    E = ['instrument_name', time, text_data]
                elif (command == 0x05):
                    E = ['lyric', time, text_data]
                elif (command == 0x06):
                    E = ['marker', time, text_data]
                elif (command == 0x07):
                    E = ['cue_point', time, text_data]
                # Reserved but apparently unassigned text events
                elif (command == 0x08):
                    E = ['text_event_08', time, text_data]
                elif (command == 0x09):
                    E = ['text_event_09', time, text_data]
                elif (command == 0x0a):
                    E = ['text_event_0a', time, text_data]
                elif (command == 0x0b):
                    E = ['text_event_0b', time, text_data]
                elif (command == 0x0c):
                    E = ['text_event_0c', time, text_data]
                elif (command == 0x0d):
                    E = ['text_event_0d', time, text_data]
                elif (command == 0x0e):
                    E = ['text_event_0e', time, text_data]
                elif (command == 0x0f):
                    E = ['text_event_0f', time, text_data]

            # Now the sticky events -------------------------------------
            elif (command == 0x2F):
                E = ['end_track', time]
                # The code for handling this, oddly, comes LATER,
                # in the event registrar.
            elif (command == 0x51):  # DTime, Microseconds/Crochet
                if length != 3:
                    _warn('set_tempo event, but length=' + str(length))
                E = ['set_tempo', time,
                     struct.unpack(">I", b'\x00' + trackdata[0:3])[0]]
            elif (command == 0x54):
                if length != 5:  # DTime, HR, MN, SE, FR, FF
                    _warn('smpte_offset event, but length=' + str(length))
                E = ['smpte_offset', time] + list(struct.unpack(">BBBBB", trackdata[0:5]))
            elif (command == 0x58):
                if length != 4:  # DTime, NN, DD, CC, BB
                    _warn('time_signature event, but length=' + str(length))
                E = ['time_signature', time] + list(trackdata[0:4])
            elif (command == 0x59):
                if length != 2:  # DTime, SF(signed), MI
                    _warn('key_signature event, but length=' + str(length))
                E = ['key_signature', time] + list(struct.unpack(">bB", trackdata[0:2]))
            elif (command == 0x7F):  # 6.4
                E = ['sequencer_specific', time, bytes(trackdata[0:length])]
            else:
                E = ['raw_meta_event', time, command,
                     bytes(trackdata[0:length])]  # 6.0
                # "[uninterpretable meta-event command of length length]"
                # DTime, Command, Binary Data
                # It's uninterpretable; record it as raw_data.

            # Pointer += length; #  Now move Pointer
            trackdata = trackdata[length:]

        ######################################################################
        elif (first_byte == 0xF0 or first_byte == 0xF7):
            # Note that sysexes in MIDI /files/ are different than sysexes
            # in MIDI transmissions!! The vast majority of system exclusive
            # messages will just use the F0 format. For instance, the
            # transmitted message F0 43 12 00 07 F7 would be stored in a
            # MIDI file as F0 05 43 12 00 07 F7. As mentioned above, it is
            # required to include the F7 at the end so that the reader of the
            # MIDI file knows that it has read the entire message. (But the F7
            # is omitted if this is a non-final block in a multiblock sysex;
            # but the F7 (if there) is counted in the message's declared
            # length, so we don't have to think about it anyway.)
            # command = trackdata.pop(0)
            [length, trackdata] = _unshift_ber_int(trackdata)
            if first_byte == 0xF0:
                # 20091008 added ISO-8859-1 to get an 8-bit str
                # 6.4 return bytes instead
                E = ['sysex_f0', time, bytes(trackdata[0:length])]
            else:
                E = ['sysex_f7', time, bytes(trackdata[0:length])]
            trackdata = trackdata[length:]

        ######################################################################
        # Now, the MIDI file spec says:
        #  <track data> = <MTrk event>+
        #  <MTrk event> = <delta-time> <event>
        #  <event> = <MIDI event> | <sysex event> | <meta-event>
        # I know that, on the wire, <MIDI event> can include note_on,
        # note_off, and all the other 8x to Ex events, AND Fx events
        # other than F0, F7, and FF -- namely, <song position msg>,
        # <song select msg>, and <tune request>.
        #
        # Whether these can occur in MIDI files is not clear specified
        # from the MIDI file spec.  So, I'm going to assume that
        # they CAN, in practice, occur.  I don't know whether it's
        # proper for you to actually emit these into a MIDI file.

        elif (first_byte == 0xF2):  # DTime, Beats
            #  <song position msg> ::=     F2 <data pair>
            E = ['song_position', time, _read_14_bit(trackdata[:2])]
            trackdata = trackdata[2:]

        elif (first_byte == 0xF3):  # <song select msg> ::= F3 <data singlet>
            # E = ['song_select', time, struct.unpack('>B',trackdata.pop(0))[0]]
            E = ['song_select', time, trackdata[0]]
            trackdata = trackdata[1:]
            # DTime, Thing (what?! song number?  whatever ...)

        elif (first_byte == 0xF6):  # DTime
            E = ['tune_request', time]
            # What would a tune request be doing in a MIDI /file/?

            #########################################################
            # ADD MORE META-EVENTS HERE.  TODO:
            # f1 -- MTC Quarter Frame Message. One data byte follows
            #     the Status; it's the time code value, from 0 to 127.
            # f8 -- MIDI clock.    no data.
            # fa -- MIDI start.    no data.
            # fb -- MIDI continue. no data.
            # fc -- MIDI stop.     no data.
            # fe -- Active sense.  no data.
            # f4 f5 f9 fd -- unallocated

            r'''
        elif (first_byte > 0xF0) { # Some unknown kinda F-series event ####
            # Here we only produce a one-byte piece of raw data.
            # But the encoder for 'raw_data' accepts any length of it.
            E = [ 'raw_data',
                         time, substr(trackdata,Pointer,1) ]
            # DTime and the Data (in this case, the one Event-byte)
            ++Pointer;  # itself

'''
        elif first_byte > 0xF0:  # Some unknown F-series event
            # Here we only produce a one-byte piece of raw data.
            # E = ['raw_data', time, bytest(trackdata[0])]   # 6.4
            E = ['raw_data', time, trackdata[0]]  # 6.4 6.7
            trackdata = trackdata[1:]
        else:  # Fallthru.
            _warn("Aborting track.  Command-byte first_byte=" + hex(first_byte))
            break
        # End of the big if-group

        ######################################################################
        #  THE EVENT REGISTRAR...
        if E and (E[0] == 'end_track'):
            # This is the code for exceptional handling of the EOT event.
            eot = True
            if not no_eot_magic:
                if E[1] > 0:  # a null text-event to carry the delta-time
                    E = ['text_event', E[1], '']
                else:
                    E = []  # EOT with a delta-time of 0; ignore it.

        if E and not (E[0] in exclude):
            # if ( $exclusive_event_callback ):
            #    &{ $exclusive_event_callback }( @E );
            # else:
            #    &{ $event_callback }( @E ) if $event_callback;
            events.append(E)
        if eot:
            break

    # End of the big "Event" while-block

    return events


###########################################################################
def _encode(events_lol, unknown_callback=None, never_add_eot=False,
  no_eot_magic=False, no_running_status=False, text_encoding='ISO-8859-1'):
    # encode an event structure, presumably for writing to a file
    # Calling format:
    #   $data_r = MIDI::Event::encode( \@event_lol, { options } );
    # Takes a REFERENCE to an event structure (a LoL)
    # Returns an (unblessed) REFERENCE to track data.

    # If you want to use this to encode a /single/ event,
    # you still have to do it as a reference to an event structure (a LoL)
    # that just happens to have just one event.  I.e.,
    #   encode( [ $event ] ) or encode( [ [ 'note_on', 100, 5, 42, 64] ] )
    # If you're doing this, consider the never_add_eot track option, as in
    #   print MIDI ${ encode( [ $event], { 'never_add_eot' => 1} ) };

    data = [] # what I'll store the chunks of byte-data in

    # This is so my end_track magic won't corrupt the original
    events = copy.deepcopy(events_lol)

    if not never_add_eot:
        # One way or another, tack on an 'end_track'
        if events:
            last = events[-1]
            if not (last[0] == 'end_track'):  # no end_track already
                if (last[0] == 'text_event' and len(last[2]) == 0):
                    # 0-length text event at track-end.
                    if no_eot_magic:
                        # Exceptional case: don't mess with track-final
                        # 0-length text_events; just peg on an end_track
                        events.append(['end_track', 0])
                    else:
                        # NORMAL CASE: replace with an end_track, leaving DTime
                        last[0] = 'end_track'
                else:
                    # last event was neither 0-length text_event nor end_track
                    events.append(['end_track', 0])
        else:  # an eventless track!
            events = [['end_track', 0],]

    # maybe_running_status = not no_running_status # unused? 4.7
    last_status = -1

    for event_r in (events):
        E = copy.deepcopy(event_r)
        # otherwise the shifting'd corrupt the original
        if not E:
            continue

        event = E.pop(0)
        if not len(event):
            continue

        dtime = int(E.pop(0))
        # print('event='+str(event)+' dtime='+str(dtime))

        event_data = ''

        if (   # MIDI events -- eligible for running status
             event    == 'note_on'
             or event == 'note_off'
             or event == 'control_change'
             or event == 'key_after_touch'
             or event == 'patch_change'
             or event == 'channel_after_touch'
             or event == 'pitch_wheel_change'  ):

            # This block is where we spend most of the time.  Gotta be tight.
            if (event == 'note_off'):
                status = 0x80 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'note_on'):
                status = 0x90 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'key_after_touch'):
                status = 0xA0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0x7F, int(E[2])&0x7F)
            elif (event == 'control_change'):
                status = 0xB0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>BB', int(E[1])&0xFF, int(E[2])&0xFF)
            elif (event == 'patch_change'):
                status = 0xC0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>B', int(E[1]) & 0xFF)
            elif (event == 'channel_after_touch'):
                status = 0xD0 | (int(E[0]) & 0x0F)
                parameters = struct.pack('>B', int(E[1]) & 0xFF)
            elif (event == 'pitch_wheel_change'):
                status = 0xE0 | (int(E[0]) & 0x0F)
                parameters =  _write_14_bit(int(E[1]) + 0x2000)
            else:
                _warn("BADASS FREAKOUT ERROR 31415!")

            # And now the encoding
            # w = BER compressed integer (not ASN.1 BER, see perlpacktut for
            # details).  Its bytes represent an unsigned integer in base 128,
            # most significant digit first, with as few digits as possible.
            # Bit eight (the high bit) is set on each byte except the last.

            data.append(_ber_compressed_int(dtime))
            if (status != last_status) or no_running_status:
                data.append(struct.pack('>B', status))
            data.append(parameters)
 
            last_status = status
            continue
        else:
            # Not a MIDI event.
            # All the code in this block could be more efficient,
            # but this is not where the code needs to be tight.
            # print "zaz $event\n";
            last_status = -1

            if event == 'raw_meta_event':
                event_data = _some_text_event(int(E[0]), E[1], text_encoding)
            elif (event == 'set_sequence_number'):  # 3.9
                event_data = b'\xFF\x00\x02'+_int2twobytes(E[0])

            # Text meta-events...
            # a case for a dict, I think (pjb) ...
            elif (event == 'text_event'):
                event_data = _some_text_event(0x01, E[0], text_encoding)
            elif (event == 'copyright_text_event'):
                event_data = _some_text_event(0x02, E[0], text_encoding)
            elif (event == 'track_name'):
                event_data = _some_text_event(0x03, E[0], text_encoding)
            elif (event == 'instrument_name'):
                event_data = _some_text_event(0x04, E[0], text_encoding)
            elif (event == 'lyric'):
                event_data = _some_text_event(0x05, E[0], text_encoding)
            elif (event == 'marker'):
                event_data = _some_text_event(0x06, E[0], text_encoding)
            elif (event == 'cue_point'):
                event_data = _some_text_event(0x07, E[0], text_encoding)
            elif (event == 'text_event_08'):
                event_data = _some_text_event(0x08, E[0], text_encoding)
            elif (event == 'text_event_09'):
                event_data = _some_text_event(0x09, E[0], text_encoding)
            elif (event == 'text_event_0a'):
                event_data = _some_text_event(0x0A, E[0], text_encoding)
            elif (event == 'text_event_0b'):
                event_data = _some_text_event(0x0B, E[0], text_encoding)
            elif (event == 'text_event_0c'):
                event_data = _some_text_event(0x0C, E[0], text_encoding)
            elif (event == 'text_event_0d'):
                event_data = _some_text_event(0x0D, E[0], text_encoding)
            elif (event == 'text_event_0e'):
                event_data = _some_text_event(0x0E, E[0], text_encoding)
            elif (event == 'text_event_0f'):
                event_data = _some_text_event(0x0F, E[0], text_encoding)
            # End of text meta-events

            elif (event == 'end_track'):
                event_data = b"\xFF\x2F\x00"

            elif (event == 'set_tempo'):
                #event_data = struct.pack(">BBwa*", 0xFF, 0x51, 3,
                #              substr( struct.pack('>I', E[0]), 1, 3))
                event_data = b'\xFF\x51\x03'+struct.pack('>I',E[0])[1:]
            elif (event == 'smpte_offset'):
                # event_data = struct.pack(">BBwBBBBB", 0xFF, 0x54, 5, E[0:5] )
                event_data = struct.pack(">BBBbBBBB", 0xFF,0x54,0x05,E[0],E[1],E[2],E[3],E[4])
            elif (event == 'time_signature'):
                # event_data = struct.pack(">BBwBBBB",  0xFF, 0x58, 4, E[0:4] )
                event_data = struct.pack(">BBBbBBB", 0xFF, 0x58, 0x04, E[0],E[1],E[2],E[3])
            elif (event == 'key_signature'):
                event_data = struct.pack(">BBBbB", 0xFF, 0x59, 0x02, E[0],E[1])
            elif (event == 'sequencer_specific'):
                # event_data = struct.pack(">BBwa*", 0xFF,0x7F, len(E[0]), E[0])
                event_data = _some_text_event(0x7F, E[0], text_encoding)
            # End of Meta-events

            # Other Things...
            elif (event == 'sysex_f0'):
                 #event_data = struct.pack(">Bwa*", 0xF0, len(E[0]), E[0])
                 #B=bitstring w=BER-compressed-integer a=null-padded-ascii-str
                 event_data = bytearray(b'\xF0')+_ber_compressed_int(len(E[0]))+bytearray(E[0])
            elif (event == 'sysex_f7'):
                 #event_data = struct.pack(">Bwa*", 0xF7, len(E[0]), E[0])
                 event_data = bytearray(b'\xF7')+_ber_compressed_int(len(E[0]))+bytearray(E[0])

            elif (event == 'song_position'):
                 event_data = b"\xF2" + _write_14_bit( E[0] )
            elif (event == 'song_select'):
                 event_data = struct.pack('>BB', 0xF3, E[0] )
            elif (event == 'tune_request'):
                 event_data = b"\xF6"
            elif (event == 'raw_data'):
                _warn("_encode: raw_data event not supported")
                # event_data = E[0]
                continue
            # End of Other Stuff

            else:
                # The Big Fallthru
                if unknown_callback:
                    # push(@data, &{ $unknown_callback }( @$event_r ))
                    pass
                else:
                    _warn("Unknown event: "+str(event))
                    # To surpress complaint here, just set
                    #  'unknown_callback' => sub { return () }
                continue

            #print "Event $event encoded part 2\n"
            if str(type(event_data)).find("'str'") >= 0:
                event_data = bytearray(event_data.encode('Latin1', 'ignore'))
            if len(event_data): # how could $event_data be empty
                # data.append(struct.pack('>wa*', dtime, event_data))
                # print(' event_data='+str(event_data))
                data.append(_ber_compressed_int(dtime)+event_data)

    return b''.join(data)

###################################################################################
###################################################################################
###################################################################################
#
#	Tegridy MIDI X Module (TMIDI X / tee-midi eks)
#
#	Based upon and includes the amazing MIDI.py module v.6.7. by Peter Billam
#	pjb.com.au
#
#	Project Los Angeles
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
###################################################################################

import os

import datetime

from datetime import datetime

import secrets

import random

import pickle

import csv

import tqdm

import multiprocessing

from itertools import zip_longest
from itertools import groupby
from itertools import cycle
from itertools import product

from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from collections import deque

from operator import itemgetter

from abc import ABC, abstractmethod

from difflib import SequenceMatcher as SM

import statistics
import math
from math import gcd

from functools import reduce

import matplotlib.pyplot as plt

import psutil

import json

from pathlib import Path

import shutil

import hashlib

from array import array

from pathlib import Path
from fnmatch import fnmatch

###################################################################################
#
# Original TMIDI Tegridy helper functions
#
###################################################################################

def Tegridy_TXT_to_INT_Converter(input_TXT_string, line_by_line_INT_string=True, max_INT = 0):

    '''Tegridy TXT to Intergers Converter
     
    Input: Input TXT string in the TMIDI-TXT format

           Type of output TXT INT string: line-by-line or one long string

           Maximum absolute integer to process. Maximum is inclusive 
           Default = process all integers. This helps to remove outliers/unwanted ints

    Output: List of pure intergers
            String of intergers in the specified format: line-by-line or one long string
            Number of processed integers
            Number of skipped integers
    
    Project Los Angeles
    Tegridy Code 2021'''

    print('Tegridy TXT to Intergers Converter')

    output_INT_list = []

    npi = 0
    nsi = 0

    TXT_List = list(input_TXT_string)
    for char in TXT_List:
      if max_INT != 0:
        if abs(ord(char)) <= max_INT:
          output_INT_list.append(ord(char))
          npi += 1
        else:
          nsi += 1  
      else:
        output_INT_list.append(ord(char))
        npi += 1    
    
    if line_by_line_INT_string:
      output_INT_string = '\n'.join([str(elem) for elem in output_INT_list])
    else:
      output_INT_string = ' '.join([str(elem) for elem in output_INT_list])  

    print('Converted TXT to INTs:', npi, ' / ', nsi)

    return output_INT_list, output_INT_string, npi, nsi

###################################################################################

def Tegridy_INT_to_TXT_Converter(input_INT_list):

    '''Tegridy Intergers to TXT Converter
     
    Input: List of intergers in TMIDI-TXT-INT format
    Output: Decoded TXT string in TMIDI-TXT format
    Project Los Angeles
    Tegridy Code 2020'''

    output_TXT_string = ''

    for i in input_INT_list:
      output_TXT_string += chr(int(i))
    
    return output_TXT_string

###################################################################################

def Tegridy_INT_String_to_TXT_Converter(input_INT_String, line_by_line_input=True):

    '''Tegridy Intergers String to TXT Converter
     
    Input: List of intergers in TMIDI-TXT-INT-String format
    Output: Decoded TXT string in TMIDI-TXT format
    Project Los Angeles
    Tegridy Code 2020'''
    
    print('Tegridy Intergers String to TXT Converter')

    if line_by_line_input:
      input_string = input_INT_String.split('\n')
    else:
      input_string = input_INT_String.split(' ')  

    output_TXT_string = ''

    for i in input_string:
      try:
        output_TXT_string += chr(abs(int(i)))
      except:
        print('Bad note:', i)
        continue  
    
    print('Done!')

    return output_TXT_string

###################################################################################

def Tegridy_SONG_to_MIDI_Converter(SONG,
                                  output_signature = 'Tegridy TMIDI Module', 
                                  track_name = 'Composition Track',
                                  number_of_ticks_per_quarter = 425,
                                  list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                  output_file_name = 'TMIDI-Composition',
                                  text_encoding='ISO-8859-1',
                                  verbose=True):

    '''Tegridy SONG to MIDI Converter
     
    Input: Input SONG in TMIDI SONG/MIDI.py Score format
           Output MIDI Track 0 name / MIDI Signature
           Output MIDI Track 1 name / Composition track name
           Number of ticks per quarter for the output MIDI
           List of 16 MIDI patch numbers for output MIDI. Def. is MuseNet compatible patches.
           Output file name w/o .mid extension.
           Optional text encoding if you are working with text_events/lyrics. This is especially useful for Karaoke. Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.

    Output: MIDI File
            Detailed MIDI stats

    Project Los Angeles
    Tegridy Code 2020'''                                  
    
    if verbose:
        print('Converting to MIDI. Please stand-by...')
    
    output_header = [number_of_ticks_per_quarter, 
                    [['track_name', 0, bytes(output_signature, text_encoding)]]]                                                    

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]], 
                    ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                    ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                    ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                    ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                    ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                    ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                    ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                    ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                    ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                    ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                    ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                    ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                    ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                    ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                    ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                    ['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [patch_list + SONG]

    midi_data = score2midi(output, text_encoding)
    detailed_MIDI_stats = score2stats(output)

    with open(output_file_name + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    if verbose:    
        print('Done! Enjoy! :)')
    
    return detailed_MIDI_stats

###################################################################################

def Tegridy_ms_SONG_to_MIDI_Converter(ms_SONG,
                                      output_signature = 'Tegridy TMIDI Module', 
                                      track_name = 'Composition Track',
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                      output_file_name = 'TMIDI-Composition',
                                      text_encoding='ISO-8859-1',
                                      timings_multiplier=1,
                                      verbose=True
                                      ):

    '''Tegridy milisecond SONG to MIDI Converter
     
    Input: Input ms SONG in TMIDI ms SONG/MIDI.py ms Score format
           Output MIDI Track 0 name / MIDI Signature
           Output MIDI Track 1 name / Composition track name
           List of 16 MIDI patch numbers for output MIDI. Def. is MuseNet compatible patches.
           Output file name w/o .mid extension.
           Optional text encoding if you are working with text_events/lyrics. This is especially useful for Karaoke. Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.
           Optional timings multiplier
           Optional verbose output

    Output: MIDI File
            Detailed MIDI stats

    Project Los Angeles
    Tegridy Code 2024'''                                  
    
    if verbose:
        print('Converting to MIDI. Please stand-by...')

    output_header = [1000,
                    [['set_tempo', 0, 1000000],
                     ['time_signature', 0, 4, 2, 24, 8],
                     ['track_name', 0, bytes(output_signature, text_encoding)]]]

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]], 
                    ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                    ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                    ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                    ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                    ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                    ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                    ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                    ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                    ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                    ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                    ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                    ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                    ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                    ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                    ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                    ['track_name', 0, bytes(track_name, text_encoding)]]

    SONG = copy.deepcopy(ms_SONG)

    if timings_multiplier != 1:
      for S in SONG:
        S[1] = S[1] * timings_multiplier
        if S[0] == 'note':
          S[2] = S[2] * timings_multiplier

    output = output_header + [patch_list + SONG]

    midi_data = score2midi(output, text_encoding)
    detailed_MIDI_stats = score2stats(output)

    with open(output_file_name + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    if verbose:    
        print('Done! Enjoy! :)')
    
    return detailed_MIDI_stats

###################################################################################

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]

def generate_colors(n):
    return [hsv_to_rgb(i/n, 1, 1) for i in range(n)]

def add_arrays(a, b):
    return [sum(pair) for pair in zip(a, b)]

#-------------------------------------------------------------------------------

def plot_ms_SONG(ms_song,
                  preview_length_in_notes=0,
                  block_lines_times_list = None,
                  plot_title='ms Song',
                  max_num_colors=129, 
                  drums_color_num=128, 
                  plot_size=(11,4), 
                  note_height = 0.75,
                  show_grid_lines=False,
                  return_plt = False,
                  timings_multiplier=1,
                  save_plt='',
                  save_only_plt_image=True,
                  save_transparent=False
                  ):

  '''Tegridy ms SONG plotter/vizualizer'''

  notes = [s for s in ms_song if s[0] == 'note']

  if (len(max(notes, key=len)) != 7) and (len(min(notes, key=len)) != 7):
    print('The song notes do not have patches information')
    print('Ploease add patches to the notes in the song')

  else:

    start_times = [(s[1] * timings_multiplier) / 1000 for s in notes]
    durations = [(s[2]  * timings_multiplier) / 1000 for s in notes]
    pitches = [s[4] for s in notes]
    patches = [s[6] for s in notes]

    colors = generate_colors(max_num_colors)
    colors[drums_color_num] = (1, 1, 1)

    pbl = (notes[preview_length_in_notes][1] * timings_multiplier) / 1000

    fig, ax = plt.subplots(figsize=plot_size)
    #fig, ax = plt.subplots()

    # Create a rectangle for each note with color based on patch number
    for start, duration, pitch, patch in zip(start_times, durations, pitches, patches):
        rect = plt.Rectangle((start, pitch), duration, note_height, facecolor=colors[patch])
        ax.add_patch(rect)

    # Set the limits of the plot
    ax.set_xlim([min(start_times), max(add_arrays(start_times, durations))])
    ax.set_ylim([min(pitches)-1, max(pitches)+1])

    # Set the background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')

    if preview_length_in_notes > 0:
      ax.axvline(x=pbl, c='white')

    if block_lines_times_list:
      for bl in block_lines_times_list:
        ax.axvline(x=bl, c='white')
           
    if show_grid_lines:
      ax.grid(color='white')

    plt.xlabel('Time (s)', c='black')
    plt.ylabel('MIDI Pitch', c='black')

    plt.title(plot_title)

    if save_plt != '':
      if save_only_plt_image:
        plt.axis('off')
        plt.title('')
        plt.savefig(save_plt, transparent=save_transparent, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
      
      else:
        plt.savefig(save_plt)
        plt.close()

    if return_plt:
      plt.close(fig)
      return fig

    plt.show()
    plt.close()

###################################################################################

def Tegridy_SONG_to_Full_MIDI_Converter(SONG,
                                        output_signature = 'Tegridy TMIDI Module', 
                                        track_name = 'Composition Track',
                                        number_of_ticks_per_quarter = 1000,
                                        output_file_name = 'TMIDI-Composition',
                                        text_encoding='ISO-8859-1',
                                        verbose=True):

    '''Tegridy SONG to Full MIDI Converter
     
    Input: Input SONG in Full TMIDI SONG/MIDI.py Score format
           Output MIDI Track 0 name / MIDI Signature
           Output MIDI Track 1 name / Composition track name
           Number of ticks per quarter for the output MIDI
           Output file name w/o .mid extension.
           Optional text encoding if you are working with text_events/lyrics. This is especially useful for Karaoke. Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.

    Output: MIDI File
            Detailed MIDI stats

    Project Los Angeles
    Tegridy Code 2023'''                                  
    
    if verbose:
        print('Converting to MIDI. Please stand-by...')
    
    output_header = [number_of_ticks_per_quarter,
                    [['set_tempo', 0, 1000000],
                      ['track_name', 0, bytes(output_signature, text_encoding)]]]                                                    

    song_track = [['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [song_track + SONG]

    midi_data = score2midi(output, text_encoding)
    detailed_MIDI_stats = score2stats(output)

    with open(output_file_name + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    if verbose:    
        print('Done! Enjoy! :)')
    
    return detailed_MIDI_stats

###################################################################################

def Tegridy_File_Time_Stamp(input_file_name='File_Created_on_', ext = ''):

  '''Tegridy File Time Stamp
     
  Input: Full path and file name without extention
         File extension
          
  Output: File name string with time-stamp and extension (time-stamped file name)

  Project Los Angeles
  Tegridy Code 2021'''       

  print('Time-stamping output file...')

  now = ''
  now_n = str(datetime.now())
  now_n = now_n.replace(' ', '_')
  now_n = now_n.replace(':', '_')
  now = now_n.replace('.', '_')
      
  fname = input_file_name + str(now) + ext

  return(fname)

###################################################################################

def Tegridy_Any_Pickle_File_Writer(Data, input_file_name='TMIDI_Pickle_File'):

  '''Tegridy Pickle File Writer
     
  Input: Data to write (I.e. a list)
         Full path and file name without extention
         
  Output: Named Pickle file

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Writer')

  full_path_to_output_dataset_to = input_file_name + '.pickle'

  if os.path.exists(full_path_to_output_dataset_to):
    os.remove(full_path_to_output_dataset_to)
    print('Removing old Dataset...')
  else:
    print("Creating new Dataset file...")

  with open(full_path_to_output_dataset_to, 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(Data, filehandle, protocol=pickle.HIGHEST_PROTOCOL)

  print('Dataset was saved as:', full_path_to_output_dataset_to)
  print('Task complete. Enjoy! :)')

###################################################################################

def Tegridy_Any_Pickle_File_Reader(input_file_name='TMIDI_Pickle_File', ext='.pickle', verbose=True):

  '''Tegridy Pickle File Loader
     
  Input: Full path and file name with or without extention
         File extension if different from default .pickle
       
  Output: Standard Python 3 unpickled data object

  Project Los Angeles
  Tegridy Code 2021'''

  if verbose:
    print('Tegridy Pickle File Loader')
    print('Loading the pickle file. Please wait...')

  if os.path.basename(input_file_name).endswith(ext):
    fname = input_file_name
  
  else:
    fname = input_file_name + ext

  with open(fname, 'rb') as pickle_file:
    content = pickle.load(pickle_file)

  if verbose:
    print('Done!')

  return content

###################################################################################

# TMIDI X Code is below

###################################################################################

def Optimus_MIDI_TXT_Processor(MIDI_file, 
                              line_by_line_output=True, 
                              chordify_TXT=False,
                              dataset_MIDI_events_time_denominator=1,
                              output_velocity=True,
                              output_MIDI_channels = False, 
                              MIDI_channel=0, 
                              MIDI_patch=[0, 1], 
                              char_offset = 30000,
                              transpose_by = 0,
                              flip=False, 
                              melody_conditioned_encoding=False,
                              melody_pitch_baseline = 0,
                              number_of_notes_to_sample = -1,
                              sampling_offset_from_start = 0,
                              karaoke=False,
                              karaoke_language_encoding='utf-8',
                              song_name='Song',
                              perfect_timings=False,
                              musenet_encoding=False,
                              transform=0,
                              zero_token=False,
                              reset_timings=False):

    '''Project Los Angeles
       Tegridy Code 2021'''
  
###########

    debug = False

    ev = 0

    chords_list_final = []
    chords_list = []
    events_matrix = []
    melody = []
    melody1 = []

    itrack = 1

    min_note = 0
    max_note = 0
    ev = 0
    patch = 0

    score = []
    rec_event = []

    txt = ''
    txtc = ''
    chords = []
    melody_chords = []

    karaoke_events_matrix = []
    karaokez = []

    sample = 0
    start_sample = 0

    bass_melody = []

    INTS = []
    bints = 0

###########    

    def list_average(num):
      sum_num = 0
      for t in num:
          sum_num = sum_num + t           

      avg = sum_num / len(num)
      return avg

###########

    #print('Loading MIDI file...')
    midi_file = open(MIDI_file, 'rb')
    if debug: print('Processing File:', MIDI_file)
    
    try:
      opus = midi2opus(midi_file.read())
    
    except:
      print('Problematic MIDI. Skipping...')
      print('File name:', MIDI_file)
      midi_file.close()
      return txt, melody, chords
         
    midi_file.close()

    score1 = to_millisecs(opus)
    score2 = opus2score(score1)

    # score2 = opus2score(opus) # TODO Improve score timings when it will be possible.
    
    if MIDI_channel == 16: # Process all MIDI channels
      score = score2
    
    if MIDI_channel >= 0 and MIDI_channel <= 15: # Process only a selected single MIDI channel
      score = grep(score2, [MIDI_channel])
    
    if MIDI_channel == -1: # Process all channels except drums (except channel 9)
      score = grep(score2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15])   
    
    #print('Reading all MIDI events from the MIDI file...')
    while itrack < len(score):
      for event in score[itrack]:
        
        if perfect_timings:
          if event[0] == 'note':
            event[1] = round(event[1], -1)
            event[2] = round(event[2], -1)

        if event[0] == 'text_event' or event[0] == 'lyric' or event[0] == 'note':
          if perfect_timings:
            event[1] = round(event[1], -1)
          karaokez.append(event)
        
        if event[0] == 'text_event' or event[0] == 'lyric':
          if perfect_timings:
            event[1] = round(event[1], -1)
          try:
            event[2] = str(event[2].decode(karaoke_language_encoding, 'replace')).replace('/', '').replace(' ', '').replace('\\', '')
          except:
            event[2] = str(event[2]).replace('/', '').replace(' ', '').replace('\\', '')
            continue
          karaoke_events_matrix.append(event)

        if event[0] == 'patch_change':
          patch = event[3]

        if event[0] == 'note' and patch in MIDI_patch:
          if len(event) == 6: # Checking for bad notes...
              eve = copy.deepcopy(event)
              
              eve[1] = int(event[1] / dataset_MIDI_events_time_denominator)
              eve[2] = int(event[2] / dataset_MIDI_events_time_denominator)
              
              eve[4] = int(event[4] + transpose_by)
              
              if flip == True:
                eve[4] = int(127 - (event[4] + transpose_by)) 
              
              if number_of_notes_to_sample > -1:
                if sample <= number_of_notes_to_sample:
                  if start_sample >= sampling_offset_from_start:
                    events_matrix.append(eve)
                    sample += 1
                    ev += 1
                  else:
                    start_sample += 1

              else:
                events_matrix.append(eve)
                ev += 1
                start_sample += 1
                
      itrack +=1 # Going to next track...

    #print('Doing some heavy pythonic sorting...Please stand by...')

    fn = os.path.basename(MIDI_file)
    song_name = song_name.replace(' ', '_').replace('=', '_').replace('\'', '-')
    if song_name == 'Song':
      sng_name = fn.split('.')[0].replace(' ', '_').replace('=', '_').replace('\'', '-')
      song_name = sng_name

    # Zero token
    if zero_token:
      txt += chr(char_offset) + chr(char_offset)
      if output_MIDI_channels:
        txt += chr(char_offset)
      if output_velocity:
        txt += chr(char_offset) + chr(char_offset)     
      else:
        txt += chr(char_offset)

      txtc += chr(char_offset) + chr(char_offset)
      if output_MIDI_channels:
        txtc += chr(char_offset)
      if output_velocity:
        txtc += chr(char_offset) + chr(char_offset)      
      else:
        txtc += chr(char_offset)
      
      txt += '=' + song_name + '_with_' + str(len(events_matrix)-1) + '_notes'
      txtc += '=' + song_name + '_with_' + str(len(events_matrix)-1) + '_notes'
    
    else:
      # Song stamp
      txt += 'SONG=' + song_name + '_with_' + str(len(events_matrix)-1) + '_notes'
      txtc += 'SONG=' + song_name + '_with_' + str(len(events_matrix)-1) + '_notes'

    if line_by_line_output:
      txt += chr(10)
      txtc += chr(10)
    else:
      txt += chr(32)
      txtc += chr(32)

    #print('Sorting input by start time...')
    events_matrix.sort(key=lambda x: x[1]) # Sorting input by start time    
    
    #print('Timings converter')
    if reset_timings:
      ev_matrix = Tegridy_Timings_Converter(events_matrix)[0]
    else:
      ev_matrix = events_matrix
    
    chords.extend(ev_matrix)
    #print(chords)

    #print('Extracting melody...')
    melody_list = []

    #print('Grouping by start time. This will take a while...')
    values = set(map(lambda x:x[1], ev_matrix)) # Non-multithreaded function version just in case

    groups = [[y for y in ev_matrix if y[1]==x and len(y) == 6] for x in values] # Grouping notes into chords while discarting bad notes...
  
    #print('Sorting events...')
    for items in groups:
        
        items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch
        
        if melody_conditioned_encoding: items[0][3] = 0 # Melody should always bear MIDI Channel 0 for code to work
        
        melody_list.append(items[0]) # Creating final melody list
        melody_chords.append(items) # Creating final chords list
        bass_melody.append(items[-1]) # Creating final bass melody list
    
    # [WIP] Melody-conditioned chords list
    if melody_conditioned_encoding == True:
      if not karaoke:
   
        previous_event = copy.deepcopy(melody_chords[0][0])

        for ev in melody_chords:
          hp = True
          ev.sort(reverse=False, key=lambda x: x[4]) # Sorting chord events by pitch
          for event in ev:
          
            # Computing events details
            start_time = int(abs(event[1] - previous_event[1]))
            
            duration = int(previous_event[2])

            if hp == True:
              if int(previous_event[4]) >= melody_pitch_baseline:
                channel = int(0)
                hp = False
              else:
                channel = int(previous_event[3]+1)
                hp = False  
            else:
              channel = int(previous_event[3]+1)
              hp = False

            pitch = int(previous_event[4])

            velocity = int(previous_event[5])

            # Writing INTergerS...
            try:
              INTS.append([(start_time)+char_offset, (duration)+char_offset, channel+char_offset, pitch+char_offset, velocity+char_offset])
            except:
              bints += 1

            # Converting to TXT if possible...
            try:
              txtc += str(chr(start_time + char_offset))
              txtc += str(chr(duration + char_offset))
              txtc += str(chr(pitch + char_offset))
              if output_velocity:
                txtc += str(chr(velocity + char_offset))
              if output_MIDI_channels:
                txtc += str(chr(channel + char_offset))

              if line_by_line_output:
              

                txtc += chr(10)
              else:

                txtc += chr(32)

              previous_event = copy.deepcopy(event)
            
            except:
              # print('Problematic MIDI event! Skipping...')
              continue

        if not line_by_line_output:
          txtc += chr(10)

        txt = txtc
        chords = melody_chords
    
    # Default stuff (not melody-conditioned/not-karaoke)
    else:      
      if not karaoke:
        melody_chords.sort(reverse=False, key=lambda x: x[0][1])
        mel_chords = []
        for mc in melody_chords:
          mel_chords.extend(mc)

        if transform != 0: 
          chords = Tegridy_Transform(mel_chords, transform)
        else:
          chords = mel_chords

        # TXT Stuff
        previous_event = copy.deepcopy(chords[0])
        for event in chords:

          # Computing events details
          start_time = int(abs(event[1] - previous_event[1]))
          
          duration = int(previous_event[2])

          channel = int(previous_event[3])

          pitch = int(previous_event[4] + transpose_by)
          if flip == True:
            pitch = 127 - int(previous_event[4] + transpose_by)

          velocity = int(previous_event[5])

          # Writing INTergerS...
          try:
            INTS.append([(start_time)+char_offset, (duration)+char_offset, channel+char_offset, pitch+char_offset, velocity+char_offset])
          except:
            bints += 1

          # Converting to TXT if possible...
          try:
            txt += str(chr(start_time + char_offset))
            txt += str(chr(duration + char_offset))
            txt += str(chr(pitch + char_offset))
            if output_velocity:
              txt += str(chr(velocity + char_offset))
            if output_MIDI_channels:
              txt += str(chr(channel + char_offset))


            if chordify_TXT == True and int(event[1] - previous_event[1]) == 0:
              txt += ''      
            else:     
              if line_by_line_output:
                txt += chr(10)
              else:
                txt += chr(32) 
            
            previous_event = copy.deepcopy(event)
          
          except:
            # print('Problematic MIDI event. Skipping...')
            continue

        if not line_by_line_output:
          txt += chr(10)      

    # Karaoke stuff
    if karaoke:

      melody_chords.sort(reverse=False, key=lambda x: x[0][1])
      mel_chords = []
      for mc in melody_chords:
        mel_chords.extend(mc)

      if transform != 0: 
        chords = Tegridy_Transform(mel_chords, transform)
      else:
        chords = mel_chords

      previous_event = copy.deepcopy(chords[0])
      for event in chords:

        # Computing events details
        start_time = int(abs(event[1] - previous_event[1]))
        
        duration = int(previous_event[2])

        channel = int(previous_event[3])

        pitch = int(previous_event[4] + transpose_by)

        velocity = int(previous_event[5])

        # Converting to TXT
        txt += str(chr(start_time + char_offset))
        txt += str(chr(duration + char_offset))
        txt += str(chr(pitch + char_offset))

        txt += str(chr(velocity + char_offset))
        txt += str(chr(channel + char_offset))     

        if start_time > 0:
          for k in karaoke_events_matrix:
            if event[1] == k[1]:
              txt += str('=')
              txt += str(k[2])          
              break

        if line_by_line_output:
          txt += chr(10)
        else:
          txt += chr(32) 
        
        previous_event = copy.deepcopy(event)
      
      if not line_by_line_output:
        txt += chr(10)

    # Final processing code...
    # =======================================================================

    # Helper aux/backup function for Karaoke
    karaokez.sort(reverse=False, key=lambda x: x[1])  

    # MuseNet sorting
    if musenet_encoding and not melody_conditioned_encoding and not karaoke:
      chords.sort(key=lambda x: (x[1], x[3]))
    
    # Final melody sort
    melody_list.sort()

    # auxs for future use
    aux1 = [None]
    aux2 = [None]

    return txt, melody_list, chords, bass_melody, karaokez, INTS, aux1, aux2 # aux1 and aux2 are not used atm

###################################################################################

def Optimus_TXT_to_Notes_Converter(Optimus_TXT_String,
                                    line_by_line_dataset = True,
                                    has_velocities = True,
                                    has_MIDI_channels = True,
                                    dataset_MIDI_events_time_denominator = 1,
                                    char_encoding_offset = 30000,
                                    save_only_first_composition = True,
                                    simulate_velocity=True,
                                    karaoke=False,
                                    zero_token=False):

    '''Project Los Angeles
       Tegridy Code 2020'''

    print('Tegridy Optimus TXT to Notes Converter')
    print('Converting TXT to Notes list...Please wait...')

    song_name = ''

    if line_by_line_dataset:
      input_string = Optimus_TXT_String.split('\n')
    else:
      input_string = Optimus_TXT_String.split(' ')

    if line_by_line_dataset:
      name_string = Optimus_TXT_String.split('\n')[0].split('=')
    else:
      name_string = Optimus_TXT_String.split(' ')[0].split('=')

    # Zero token
    zt = ''

    zt += chr(char_encoding_offset) + chr(char_encoding_offset)
    
    if has_MIDI_channels:
      zt += chr(char_encoding_offset)
    
    if has_velocities:
      zt += chr(char_encoding_offset) + chr(char_encoding_offset)     
    
    else:
      zt += chr(char_encoding_offset)

    if zero_token:
      if name_string[0] == zt:
        song_name = name_string[1]
    
    else:
      if name_string[0] == 'SONG':
        song_name = name_string[1]

    output_list = []
    st = 0

    for i in range(2, len(input_string)-1):

      if save_only_first_composition:
        if zero_token:
          if input_string[i].split('=')[0] == zt:

            song_name = name_string[1]
            break
        
        else:
          if input_string[i].split('=')[0] == 'SONG':

            song_name = name_string[1]
            break
      try:
        istring = input_string[i]

        if has_MIDI_channels == False:
          step = 4          

        if has_MIDI_channels == True:
          step = 5

        if has_velocities == False:
          step -= 1

        st += int(ord(istring[0]) - char_encoding_offset) * dataset_MIDI_events_time_denominator

        if not karaoke:
          for s in range(0, len(istring), step):
              if has_MIDI_channels==True:
                if step > 3 and len(istring) > 2:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time

                      out.append(int(ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration

                      if has_velocities:
                        out.append(int(ord(istring[s+4]) - char_encoding_offset)) # Channel
                      else:
                        out.append(int(ord(istring[s+3]) - char_encoding_offset)) # Channel  

                      out.append(int(ord(istring[s+2]) - char_encoding_offset)) # Pitch

                      if simulate_velocity:
                        if s == 0:
                          sim_vel = int(ord(istring[s+2]) - char_encoding_offset)
                        out.append(sim_vel) # Simulated Velocity (= highest note's pitch)
                      else:                      
                        out.append(int(ord(istring[s+3]) - char_encoding_offset)) # Velocity

              if has_MIDI_channels==False:
                if step > 3 and len(istring) > 2:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time
                      out.append(int(ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(0) # Channel
                      out.append(int(ord(istring[s+2]) - char_encoding_offset)) # Pitch

                      if simulate_velocity:
                        if s == 0:
                          sim_vel = int(ord(istring[s+2]) - char_encoding_offset)
                        out.append(sim_vel) # Simulated Velocity (= highest note's pitch)
                      else:                      
                        out.append(int(ord(istring[s+3]) - char_encoding_offset)) # Velocity

              if step == 3 and len(istring) > 2:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time
                      out.append(int(ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(0) # Channel
                      out.append(int(ord(istring[s+2]) - char_encoding_offset)) # Pitch

                      out.append(int(ord(istring[s+2]) - char_encoding_offset)) # Velocity = Pitch

              output_list.append(out)

        if karaoke:
          try:
              out = []       
              out.append('note')

              out.append(st) # Start time
              out.append(int(ord(istring[1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
              out.append(int(ord(istring[4]) - char_encoding_offset)) # Channel
              out.append(int(ord(istring[2]) - char_encoding_offset)) # Pitch

              if simulate_velocity:
                if s == 0:
                  sim_vel = int(ord(istring[2]) - char_encoding_offset)
                out.append(sim_vel) # Simulated Velocity (= highest note's pitch)
              else:                      
                out.append(int(ord(istring[3]) - char_encoding_offset)) # Velocity
              output_list.append(out)
              out = []
              if istring.split('=')[1] != '':
                out.append('lyric')
                out.append(st)
                out.append(istring.split('=')[1])
                output_list.append(out)
          except:
            continue


      except:
        print('Bad note string:', istring)
        continue

    # Simple error control just in case
    S = []
    for x in output_list:
      if len(x) == 6 or len(x) == 3:
        S.append(x)

    output_list.clear()    
    output_list = copy.deepcopy(S)


    print('Task complete! Enjoy! :)')

    return output_list, song_name

###################################################################################

def Optimus_Data2TXT_Converter(data,
                              dataset_time_denominator=1,
                              transpose_by = 0,
                              char_offset = 33,
                              line_by_line_output = True,
                              output_velocity = False,
                              output_MIDI_channels = False):


  '''Input: data as a flat chords list of flat chords lists

  Output: TXT string
          INTs

  Project Los Angeles
  Tegridy Code 2021'''

  txt = ''
  TXT = ''

  quit = False
  counter = 0

  INTs = []
  INTs_f = []

  for d in tqdm.tqdm(sorted(data)):

    if quit == True:
      break

    txt = 'SONG=' + str(counter)
    counter += 1

    if line_by_line_output:
      txt += chr(10)
    else:
      txt += chr(32)
      
    INTs = []

    # TXT Stuff
    previous_event = copy.deepcopy(d[0])
    for event in sorted(d):

      # Computing events details
      start_time = int(abs(event[1] - previous_event[1]) / dataset_time_denominator)
      
      duration = int(previous_event[2] / dataset_time_denominator)

      channel = int(previous_event[3])

      pitch = int(previous_event[4] + transpose_by)

      velocity = int(previous_event[5])

      INTs.append([start_time, duration, pitch])

      # Converting to TXT if possible...
      try:
        txt += str(chr(start_time + char_offset))
        txt += str(chr(duration + char_offset))
        txt += str(chr(pitch + char_offset))
        if output_velocity:
          txt += str(chr(velocity + char_offset))
        if output_MIDI_channels:
          txt += str(chr(channel + char_offset))
    
        if line_by_line_output:
          txt += chr(10)
        else:
          txt += chr(32) 
        
        previous_event = copy.deepcopy(event)
      except KeyboardInterrupt:
        quit = True
        break
      except:
        print('Problematic MIDI data. Skipping...')
        continue

    if not line_by_line_output:
      txt += chr(10)
    
    TXT += txt
    INTs_f.extend(INTs)

  return TXT, INTs_f

###################################################################################

def Optimus_Squash(chords_list, simulate_velocity=True, mono_compression=False):

  '''Input: Flat chords list
            Simulate velocity or not
            Mono-compression enabled or disabled
            
            Default is almost lossless 25% compression, otherwise, lossy 50% compression (mono-compression)

     Output: Squashed chords list
             Resulting compression level

             Please note that if drums are passed through as is

     Project Los Angeles
     Tegridy Code 2021'''

  output = []
  ptime = 0
  vel = 0
  boost = 15
  stptc = []
  ocount = 0
  rcount = 0

  for c in chords_list:
    
    cc = copy.deepcopy(c)
    ocount += 1
    
    if [cc[1], cc[3], (cc[4] % 12) + 60] not in stptc:
      stptc.append([cc[1], cc[3], (cc[4] % 12) + 60])

      if cc[3] != 9:
        cc[4] = (c[4] % 12) + 60

      if simulate_velocity and c[1] != ptime:
        vel = c[4] + boost
      
      if cc[3] != 9:
        cc[5] = vel

      if mono_compression:
        if c[1] != ptime:
          output.append(cc)
          rcount += 1  
      else:
        output.append(cc)
        rcount += 1
      
      ptime = c[1]

  output.sort(key=lambda x: (x[1], x[4]))

  comp_level = 100 - int((rcount * 100) / ocount)

  return output, comp_level

###################################################################################

def Optimus_Signature(chords_list, calculate_full_signature=False):

    '''Optimus Signature

    ---In the name of the search for a perfect score slice signature---
     
    Input: Flat chords list to evaluate

    Output: Full Optimus Signature as a list
            Best/recommended Optimus Signature as a list

    Project Los Angeles
    Tegridy Code 2021'''
    
    # Pitches

    ## StDev
    if calculate_full_signature:
      psd = statistics.stdev([int(y[4]) for y in chords_list])
    else:
      psd = 0

    ## Median
    pmh = statistics.median_high([int(y[4]) for y in chords_list])
    pm = statistics.median([int(y[4]) for y in chords_list])
    pml = statistics.median_low([int(y[4]) for y in chords_list])
    
    ## Mean
    if calculate_full_signature:
      phm = statistics.harmonic_mean([int(y[4]) for y in chords_list])
    else:
      phm = 0

    # Durations
    dur = statistics.median([int(y[2]) for y in chords_list])

    # Velocities

    vel = statistics.median([int(y[5]) for y in chords_list])

    # Beats
    mtds = statistics.median([int(abs(chords_list[i-1][1]-chords_list[i][1])) for i in range(1, len(chords_list))])
    if calculate_full_signature:
      hmtds = statistics.harmonic_mean([int(abs(chords_list[i-1][1]-chords_list[i][1])) for i in range(1, len(chords_list))])
    else:
      hmtds = 0

    # Final Optimus signatures
    full_Optimus_signature = [round(psd), round(pmh), round(pm), round(pml), round(phm), round(dur), round(vel), round(mtds), round(hmtds)]
    ########################    PStDev     PMedianH    PMedian    PMedianL    PHarmoMe    Duration    Velocity      Beat       HarmoBeat

    best_Optimus_signature = [round(pmh), round(pm), round(pml), round(dur, -1), round(vel, -1), round(mtds, -1)]
    ########################   PMedianH    PMedian    PMedianL      Duration        Velocity          Beat
    
    # Return...
    return full_Optimus_signature, best_Optimus_signature
    

###################################################################################
#
# TMIDI 2.0 Helper functions
#
###################################################################################

def Tegridy_FastSearch(needle, haystack, randomize = False):

  '''

  Input: Needle iterable
         Haystack iterable
         Randomize search range (this prevents determinism)

  Output: Start index of the needle iterable in a haystack iterable
          If nothing found, -1 is returned

  Project Los Angeles
  Tegridy Code 2021'''

  need = copy.deepcopy(needle)

  try:
    if randomize:
      idx = haystack.index(need, secrets.randbelow(len(haystack)-len(need)))
    else:
      idx = haystack.index(need)

  except KeyboardInterrupt:
    return -1

  except:
    return -1
    
  return idx

###################################################################################

def Tegridy_Chord_Match(chord1, chord2, match_type=2):

    '''Tegridy Chord Match
     
    Input: Two chords to evaluate
           Match type: 2 = duration, channel, pitch, velocity
                       3 = channel, pitch, velocity
                       4 = pitch, velocity
                       5 = velocity

    Output: Match rating (0-100)
            NOTE: Match rating == -1 means identical source chords
            NOTE: Match rating == 100 means mutual shortest chord

    Project Los Angeles
    Tegridy Code 2021'''

    match_rating = 0

    if chord1 == []:
      return 0
    if chord2 == []:
      return 0

    if chord1 == chord2:
      return -1

    else:
      zipped_pairs = list(zip(chord1, chord2))
      zipped_diff = abs(len(chord1) - len(chord2))

      short_match = [False]
      for pair in zipped_pairs:
        cho1 = ' '.join([str(y) for y in pair[0][match_type:]])
        cho2 = ' '.join([str(y) for y in pair[1][match_type:]])
        if cho1 == cho2:
          short_match.append(True)
        else:
          short_match.append(False)
      
      if True in short_match:
        return 100

      pairs_ratings = []

      for pair in zipped_pairs:
        cho1 = ' '.join([str(y) for y in pair[0][match_type:]])
        cho2 = ' '.join([str(y) for y in pair[1][match_type:]])
        pairs_ratings.append(SM(None, cho1, cho2).ratio())

      match_rating = sum(pairs_ratings) / len(pairs_ratings) * 100

      return match_rating

###################################################################################

def Tegridy_Last_Chord_Finder(chords_list):

    '''Tegridy Last Chord Finder
     
    Input: Flat chords list

    Output: Last detected chord of the chords list
            Last chord start index in the original chords list
            First chord end index in the original chords list

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    ptime = 0

    i = 0

    pc_idx = 0
    fc_idx = 0

    chords_list.sort(reverse=False, key=lambda x: x[1])
    
    for cc in chords_list:

      if cc[1] == ptime:
        
        cho.append(cc)

        ptime = cc[1]

      else:
        if pc_idx == 0: 
          fc_idx = chords_list.index(cc)
        pc_idx = chords_list.index(cc)
        
        chords.append(cho)
        
        cho = []
      
        cho.append(cc)
        
        ptime = cc[1]
        
        i += 1
      
    if cho != []: 
      chords.append(cho)
      i += 1
     
    return chords_list[pc_idx:], pc_idx, fc_idx

###################################################################################

def Tegridy_Chords_Generator(chords_list, shuffle_pairs = True, remove_single_notes=False):

    '''Tegridy Score Chords Pairs Generator
     
    Input: Flat chords list
           Shuffle pairs (recommended)

    Output: List of chords
            
            Average time(ms) per chord
            Average time(ms) per pitch
            Average chords delta time

            Average duration
            Average channel
            Average pitch
            Average velocity

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    i = 0

    # Sort by start time
    chords_list.sort(reverse=False, key=lambda x: x[1])

    # Main loop
    pcho = chords_list[0]
    for cc in chords_list:
      if cc[1] == pcho[1]:
        
        cho.append(cc)
        pcho = copy.deepcopy(cc)

      else:
        if not remove_single_notes:
          chords.append(cho)
          cho = []
          cho.append(cc)
          pcho = copy.deepcopy(cc)
          
          i += 1
        else:
          if len(cho) > 1:
            chords.append(cho)
          cho = []
          cho.append(cc)
          pcho = copy.deepcopy(cc)
            
          i += 1  
    
    # Averages
    t0 = chords[0][0][1]
    t1 = chords[-1][-1][1]
    tdel = abs(t1 - t0)
    avg_ms_per_chord = int(tdel / i)
    avg_ms_per_pitch = int(tdel / len(chords_list))

    # Delta time
    tds = [int(abs(chords_list[i-1][1]-chords_list[i][1]) / 1) for i in range(1, len(chords_list))]
    if len(tds) != 0: avg_delta_time = int(sum(tds) / len(tds))

    # Chords list attributes
    p = int(sum([int(y[4]) for y in chords_list]) / len(chords_list))
    d = int(sum([int(y[2]) for y in chords_list]) / len(chords_list))
    c = int(sum([int(y[3]) for y in chords_list]) / len(chords_list))
    v = int(sum([int(y[5]) for y in chords_list]) / len(chords_list))

    # Final shuffle
    if shuffle_pairs:
      random.shuffle(chords)

    return chords, [avg_ms_per_chord, avg_ms_per_pitch, avg_delta_time], [d, c, p, v]

###################################################################################

def Tegridy_Chords_List_Music_Features(chords_list, st_dur_div = 1, pitch_div = 1, vel_div = 1):

    '''Tegridy Chords List Music Features
     
    Input: Flat chords list

    Output: A list of the extracted chords list's music features

    Project Los Angeles
    Tegridy Code 2021'''

    chords_list1 = [x for x in chords_list if x]
    chords_list1.sort(reverse=False, key=lambda x: x[1])
    
    # Features extraction code

    melody_list = []
    bass_melody = []
    melody_chords = []
    mel_avg_tds = []
    mel_chrd_avg_tds = []
    bass_melody_avg_tds = []

    #print('Grouping by start time. This will take a while...')
    values = set(map(lambda x:x[1], chords_list1)) # Non-multithreaded function version just in case

    groups = [[y for y in chords_list1 if y[1]==x and len(y) == 6] for x in values] # Grouping notes into chords while discarting bad notes...

    #print('Sorting events...')
    for items in groups:
        items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch
        melody_list.append(items[0]) # Creating final melody list
        melody_chords.append(items) # Creating final chords list
        bass_melody.append(items[-1]) # Creating final bass melody list

    #print('Final sorting by start time...')      
    melody_list.sort(reverse=False, key=lambda x: x[1]) # Sorting events by start time
    melody_chords.sort(reverse=False, key=lambda x: x[0][1]) # Sorting events by start time
    bass_melody.sort(reverse=False, key=lambda x: x[1]) # Sorting events by start time

    # Extracting music features from the chords list
    
    # Melody features
    mel_avg_pitch = int(sum([y[4] for y in melody_list]) / len(melody_list) / pitch_div)
    mel_avg_dur = int(sum([int(y[2] / st_dur_div) for y in melody_list]) / len(melody_list))
    mel_avg_vel = int(sum([int(y[5] / vel_div) for y in melody_list]) / len(melody_list))
    mel_avg_chan = int(sum([int(y[3]) for y in melody_list]) / len(melody_list))
    
    mel_tds = [int(abs(melody_list[i-1][1]-melody_list[i][1])) for i in range(1, len(melody_list))]
    if len(mel_tds) != 0: mel_avg_tds = int(sum(mel_tds) / len(mel_tds) / st_dur_div)
    
    melody_features = [mel_avg_tds, mel_avg_dur, mel_avg_chan, mel_avg_pitch, mel_avg_vel]

    # Chords list features
    mel_chrd_avg_pitch = int(sum([y[4] for y in chords_list1]) / len(chords_list1) / pitch_div)
    mel_chrd_avg_dur = int(sum([int(y[2] / st_dur_div) for y in chords_list1]) / len(chords_list1))
    mel_chrd_avg_vel = int(sum([int(y[5] / vel_div) for y in chords_list1]) / len(chords_list1))
    mel_chrd_avg_chan = int(sum([int(y[3]) for y in chords_list1]) / len(chords_list1))
    
    mel_chrd_tds = [int(abs(chords_list1[i-1][1]-chords_list1[i][1])) for i in range(1, len(chords_list1))]
    if len(mel_tds) != 0: mel_chrd_avg_tds = int(sum(mel_chrd_tds) / len(mel_chrd_tds) / st_dur_div)
    
    chords_list_features = [mel_chrd_avg_tds, mel_chrd_avg_dur, mel_chrd_avg_chan, mel_chrd_avg_pitch, mel_chrd_avg_vel]

    # Bass melody features
    bass_melody_avg_pitch = int(sum([y[4] for y in bass_melody]) / len(bass_melody) / pitch_div)
    bass_melody_avg_dur = int(sum([int(y[2] / st_dur_div) for y in bass_melody]) / len(bass_melody))
    bass_melody_avg_vel = int(sum([int(y[5] / vel_div) for y in bass_melody]) / len(bass_melody))
    bass_melody_avg_chan = int(sum([int(y[3]) for y in bass_melody]) / len(bass_melody))
    
    bass_melody_tds = [int(abs(bass_melody[i-1][1]-bass_melody[i][1])) for i in range(1, len(bass_melody))]
    if len(bass_melody_tds) != 0: bass_melody_avg_tds = int(sum(bass_melody_tds) / len(bass_melody_tds) / st_dur_div)
    
    bass_melody_features = [bass_melody_avg_tds, bass_melody_avg_dur, bass_melody_avg_chan, bass_melody_avg_pitch, bass_melody_avg_vel]
    
    # A list to return all features
    music_features = []

    music_features.extend([len(chords_list1)]) # Count of the original chords list notes
    
    music_features.extend(melody_features) # Extracted melody features
    music_features.extend(chords_list_features) # Extracted chords list features
    music_features.extend(bass_melody_features) # Extracted bass melody features
    music_features.extend([sum([y[4] for y in chords_list1])]) # Sum of all pitches in the original chords list

    return music_features

###################################################################################

def Tegridy_Transform(chords_list, to_pitch=60, to_velocity=-1):

    '''Tegridy Transform
     
    Input: Flat chords list
           Desired average pitch (-1 == no change)
           Desired average velocity (-1 == no change)

    Output: Transformed flat chords list

    Project Los Angeles
    Tegridy Code 2021'''

    transformed_chords_list = []

    chords_list.sort(reverse=False, key=lambda x: x[1])

    chords_list_features = Optimus_Signature(chords_list)[1]

    pitch_diff = int((chords_list_features[0] + chords_list_features[1] + chords_list_features[2]) / 3) - to_pitch
    velocity_diff = chords_list_features[4] - to_velocity

    for c in chords_list:
      cc = copy.deepcopy(c)
      if c[3] != 9: # Except the drums
        if to_pitch != -1: 
          cc[4] = c[4] - pitch_diff
        
        if to_velocity != -1: 
          cc[5] = c[5] - velocity_diff
      
      transformed_chords_list.append(cc)

    return transformed_chords_list

###################################################################################

def Tegridy_MIDI_Zip_Notes_Summarizer(chords_list, match_type = 4):

    '''Tegridy MIDI Zip Notes Summarizer
     
    Input: Flat chords list / SONG
           Match type according to 'note' event of MIDI.py

    Output: Summarized chords list
            Number of summarized notes
            Number of dicarted notes

    Project Los Angeles
    Tegridy Code 2021'''

    i = 0
    j = 0
    out1 = []
    pout = []
 

    for o in chords_list:

      # MIDI Zip

      if o[match_type:] not in pout:
        pout.append(o[match_type:])
        
        out1.append(o)
        j += 1
      
      else:
        i += 1

    return out1, i

###################################################################################

def Tegridy_Score_Chords_Pairs_Generator(chords_list, shuffle_pairs = True, remove_single_notes=False):

    '''Tegridy Score Chords Pairs Generator
     
    Input: Flat chords list
           Shuffle pairs (recommended)

    Output: Score chords pairs list
            Number of created pairs
            Number of detected chords

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    i = 0
    j = 0

    chords_list.sort(reverse=False, key=lambda x: x[1])
    pcho = chords_list[0]
    for cc in chords_list:
      if cc[1] == pcho[1]:
        
        cho.append(cc)
        pcho = copy.deepcopy(cc)

      else:
        if not remove_single_notes:
          chords.append(cho)
          cho = []
          cho.append(cc)
          pcho = copy.deepcopy(cc)
          
          i += 1
        else:
          if len(cho) > 1:
            chords.append(cho)
          cho = []
          cho.append(cc)
          pcho = copy.deepcopy(cc)
            
          i += 1  
    
    chords_pairs = []
    for i in range(len(chords)-1):
      chords_pairs.append([chords[i], chords[i+1]])
      j += 1
    if shuffle_pairs: random.shuffle(chords_pairs)

    return chords_pairs, j, i

###################################################################################

def Tegridy_Sliced_Score_Pairs_Generator(chords_list, number_of_miliseconds_per_slice=2000, shuffle_pairs = False):

    '''Tegridy Sliced Score Pairs Generator
     
    Input: Flat chords list
           Number of miliseconds per slice

    Output: Sliced score pairs list
            Number of created slices

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    time = number_of_miliseconds_per_slice 

    i = 0

    chords_list1 = [x for x in chords_list if x]
    chords_list1.sort(reverse=False, key=lambda x: x[1])
    pcho = chords_list1[0]
    for cc in chords_list1[1:]:

      if cc[1] <= time:
        
        cho.append(cc)

      else:
        if cho != [] and pcho != []: chords.append([pcho, cho])
        pcho = copy.deepcopy(cho)
        cho = []
        cho.append(cc)
        time += number_of_miliseconds_per_slice
        i += 1
      
    if cho != [] and pcho != []: 
      chords.append([pcho, cho])
      pcho = copy.deepcopy(cho)
      i += 1
    
    if shuffle_pairs: random.shuffle(chords)

    return chords, i

###################################################################################

def Tegridy_Timings_Converter(chords_list, 
                              max_delta_time = 1000, 
                              fixed_start_time = 250, 
                              start_time = 0,
                              start_time_multiplier = 1,
                              durations_multiplier = 1):

    '''Tegridy Timings Converter
     
    Input: Flat chords list
           Max delta time allowed between notes
           Fixed start note time for excessive gaps

    Output: Converted flat chords list

    Project Los Angeles
    Tegridy Code 2021'''

    song = chords_list

    song1 = []

    p = song[0]

    p[1] = start_time

    time = start_time

    delta = [0]

    for i in range(len(song)):
      if song[i][0] == 'note':
        ss = copy.deepcopy(song[i])
        if song[i][1] != p[1]:
          
          if abs(song[i][1] - p[1]) > max_delta_time:
            time += fixed_start_time
          else:
            time += abs(song[i][1] - p[1])
            delta.append(abs(song[i][1] - p[1]))

          ss[1] = int(round(time * start_time_multiplier, -1))
          ss[2] = int(round(song[i][2] * durations_multiplier, -1))
          song1.append(ss)
          
          p = copy.deepcopy(song[i])
        else:
          
          ss[1] = int(round(time * start_time_multiplier, -1))
          ss[2] = int(round(song[i][2] * durations_multiplier, -1))
          song1.append(ss)
          
          p = copy.deepcopy(song[i])
      
      else:
        ss = copy.deepcopy(song[i])
        ss[1] = time
        song1.append(ss)
        
    average_delta_st = int(sum(delta) / len(delta))
    average_duration = int(sum([y[2] for y in song1 if y[0] == 'note']) / len([y[2] for y in song1 if y[0] == 'note']))

    song1.sort(reverse=False, key=lambda x: x[1])

    return song1, time, average_delta_st, average_duration

###################################################################################

def Tegridy_Score_Slicer(chords_list, number_of_miliseconds_per_slice=2000, overlap_notes = 0, overlap_chords=False):

    '''Tegridy Score Slicer
     
    Input: Flat chords list
           Number of miliseconds per slice

    Output: Sliced chords list
            Number of created slices

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    time = number_of_miliseconds_per_slice
    ptime = 0

    i = 0

    pc_idx = 0

    chords_list.sort(reverse=False, key=lambda x: x[1])
    
    for cc in chords_list:

      if cc[1] <= time:
        
        cho.append(cc)

        if ptime != cc[1]:
          pc_idx = cho.index(cc)

        ptime = cc[1]


      else:

        if overlap_chords:
          chords.append(cho)
          cho.extend(chords[-1][pc_idx:])
        
        else:
          chords.append(cho[:pc_idx])
        
        cho = []
      
        cho.append(cc)
        
        time += number_of_miliseconds_per_slice
        ptime = cc[1]
        
        i += 1
      
    if cho != []: 
      chords.append(cho)
      i += 1
    
    return [x for x in chords if x], i

###################################################################################

def Tegridy_TXT_Tokenizer(input_TXT_string, line_by_line_TXT_string=True):

    '''Tegridy TXT Tokenizer
     
    Input: TXT String

    Output: Tokenized TXT string + forward and reverse dics
    
    Project Los Angeles
    Tegridy Code 2021'''

    print('Tegridy TXT Tokenizer')

    if line_by_line_TXT_string:
      T = input_TXT_string.split()
    else:
      T = input_TXT_string.split(' ')

    DIC = dict(zip(T, range(len(T))))
    RDIC = dict(zip(range(len(T)), T))

    TXTT = ''

    for t in T:
      try:
        TXTT += chr(DIC[t])
      except:
        print('Error. Could not finish.')
        return TXTT, DIC, RDIC
    
    print('Done!')
    
    return TXTT, DIC, RDIC

###################################################################################

def Tegridy_TXT_DeTokenizer(input_Tokenized_TXT_string, RDIC):

    '''Tegridy TXT Tokenizer
     
    Input: Tokenized TXT String
           

    Output: DeTokenized TXT string
    
    Project Los Angeles
    Tegridy Code 2021'''

    print('Tegridy TXT DeTokenizer')

    Q = list(input_Tokenized_TXT_string)
    c = 0
    RTXT = ''
    for q in Q:
      try:
        RTXT += RDIC[ord(q)] + chr(10)
      except:
        c+=1

    print('Number of errors:', c)

    print('Done!')

    return RTXT

###################################################################################

def Tegridy_List_Slicer(input_list, slices_length_in_notes=20):

  '''Input: List to slice
            Desired slices length in notes
     
     Output: Sliced list of lists
     
     Project Los Angeles
     Tegridy Code 2021'''

  for i in range(0, len(input_list), slices_length_in_notes):
     yield input_list[i:i + slices_length_in_notes]
    
###################################################################################    
    
def Tegridy_Split_List(list_to_split, split_value=0):
    
    # src courtesy of www.geeksforgeeks.org
  
    # using list comprehension + zip() + slicing + enumerate()
    # Split list into lists by particular value
    size = len(list_to_split)
    idx_list = [idx + 1 for idx, val in
                enumerate(list_to_split) if val == split_value]


    res = [list_to_split[i: j] for i, j in
            zip([0] + idx_list, idx_list + 
            ([size] if idx_list[-1] != size else []))]
  
    # print result
    # print("The list after splitting by a value : " + str(res))
    
    return res

###################################################################################

# Binary chords functions

def tones_chord_to_bits(chord, reverse=True):

    bits = [0] * 12

    for num in chord:
        bits[num] = 1
    
    if reverse:
      bits.reverse()
      return bits
    
    else:
      return bits

def bits_to_tones_chord(bits):
    return [i for i, bit in enumerate(bits) if bit == 1]

def shift_bits(bits, n):
    return bits[-n:] + bits[:-n]

def bits_to_int(bits, shift_bits_value=0):
    bits = shift_bits(bits, shift_bits_value)
    result = 0
    for bit in bits:
        result = (result << 1) | bit
    
    return result

def int_to_bits(n):
    bits = [0] * 12
    for i in range(12):
        bits[11 - i] = n % 2
        n //= 2
    
    return bits

def bad_chord(chord):
    bad = any(b - a == 1 for a, b in zip(chord, chord[1:]))
    if (0 in chord) and (11 in chord):
      bad = True
    
    return bad

def pitches_chord_to_int(pitches_chord, tones_transpose_value=0):

    pitches_chord = [x for x in pitches_chord if 0 < x < 128]

    if not (-12 < tones_transpose_value < 12):
      tones_transpose_value = 0

    tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))
    bits = tones_chord_to_bits(tones_chord)
    integer = bits_to_int(bits, shift_bits_value=tones_transpose_value)

    return integer

def int_to_pitches_chord(integer, chord_base_pitch=60): 
    if 0 < integer < 4096:
      bits = int_to_bits(integer)
      tones_chord = bits_to_tones_chord(bits)
      if not bad_chord(tones_chord):
        pitches_chord = [t+chord_base_pitch for t in tones_chord]
        return [pitches_chord, tones_chord]
      
      else:
        return 0 # Bad chord code
    
    else:
      return -1 # Bad integer code

###################################################################################

def bad_chord(chord):
    bad = any(b - a == 1 for a, b in zip(chord, chord[1:]))
    if (0 in chord) and (11 in chord):
      bad = True
    
    return bad

def validate_pitches_chord(pitches_chord, return_sorted = True):

    pitches_chord = sorted(list(set([x for x in pitches_chord if 0 < x < 128])))

    tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

    if not bad_chord(tones_chord):
      if return_sorted:
        pitches_chord.sort(reverse=True)
      return pitches_chord
    
    else:
      if 0 in tones_chord and 11 in tones_chord:
        tones_chord.remove(0)

      fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

      fixed_tones_chord = []
      for f in fixed_tones:
        fixed_tones_chord.extend(f)
      fixed_tones_chord = list(set(fixed_tones_chord))
      
      fixed_pitches_chord = []

      for p in pitches_chord:
        if (p % 12) in fixed_tones_chord:
          fixed_pitches_chord.append(p)

      if return_sorted:
        fixed_pitches_chord.sort(reverse=True)

    return fixed_pitches_chord

def validate_pitches(chord, channel_to_check = 0, return_sorted = True):

    pitches_chord = sorted(list(set([x[4] for x in chord if 0 < x[4] < 128 and x[3] == channel_to_check])))

    if pitches_chord:

      tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

      if not bad_chord(tones_chord):
        if return_sorted:
          chord.sort(key = lambda x: x[4], reverse=True)
        return chord
      
      else:
        if 0 in tones_chord and 11 in tones_chord:
          tones_chord.remove(0)

        fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

        fixed_tones_chord = []
        for f in fixed_tones:
          fixed_tones_chord.extend(f)
        fixed_tones_chord = list(set(fixed_tones_chord))
        
        fixed_chord = []

        for c in chord:
          if c[3] == channel_to_check:
            if (c[4] % 12) in fixed_tones_chord:
              fixed_chord.append(c)
          else:
            fixed_chord.append(c)

        if return_sorted:
          fixed_chord.sort(key = lambda x: x[4], reverse=True)
      
        return fixed_chord 

    else:
      chord.sort(key = lambda x: x[4], reverse=True)
      return chord

def adjust_score_velocities(score,
                            max_velocity,
                            adj_per_channel=False,
                            adj_in_place=True
                           ):
  
    if adj_in_place:
        buf = score
        
    else:
        buf = copy.deepcopy(score)

    notes = [evt for evt in buf if evt[0] == 'note']
    
    if not notes:
        return buf

    if adj_per_channel:
        channel_max = {}
        
        for _, _, _, ch, _, vel, _ in notes:
            channel_max[ch] = max(channel_max.get(ch, 0), vel)

        channel_factor = {
            ch: (max_velocity / vmax if vmax > 0 else 1.0)
            for ch, vmax in channel_max.items()
        }

        for evt in buf:
            if evt[0] == 'note':
                ch = evt[3]
                factor = channel_factor.get(ch, 1.0)
                new_vel = int(evt[5] * factor)
                evt[5] = max(1, min(127, new_vel))

    else:
        global_max = max(vel for _, _, _, _, _, vel, _ in notes)
        factor = max_velocity / global_max if global_max > 0 else 1.0

        for evt in buf:
            if evt[0] == 'note':
                new_vel = int(evt[5] * factor)
                evt[5] = max(1, min(127, new_vel))

    if not adj_in_place:
        return buf

def chordify_score(score,
                  return_choridfied_score=True,
                  return_detected_score_information=False
                  ):

    if score:
    
      num_tracks = 1
      single_track_score = []
      score_num_ticks = 0

      if type(score[0]) == int and len(score) > 1:

        score_type = 'MIDI_PY'
        score_num_ticks = score[0]

        while num_tracks < len(score):
            for event in score[num_tracks]:
              single_track_score.append(event)
            num_tracks += 1
      
      else:
        score_type = 'CUSTOM'
        single_track_score = score

      if single_track_score and single_track_score[0]:
        
        try:

          if type(single_track_score[0][0]) == str or single_track_score[0][0] == 'note':
            single_track_score.sort(key = lambda x: x[1])
            score_timings = [s[1] for s in single_track_score]
          else:
            score_timings = [s[0] for s in single_track_score]

          is_score_time_absolute = lambda sct: all(x <= y for x, y in zip(sct, sct[1:]))

          score_timings_type = ''

          if is_score_time_absolute(score_timings):
            score_timings_type = 'ABS'

            chords = []
            cho = []

            if score_type == 'MIDI_PY':
              pe = single_track_score[0]
            else:
              pe = single_track_score[0]

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
                ptime = pe[1]
              else:
                time = e[0]
                ptime = pe[0]

              if time == ptime:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

              pe = e

            if len(cho) > 0:
              chords.append(cho)

          else:
            score_timings_type = 'REL'
            
            chords = []
            cho = []

            for e in single_track_score:
              
              if score_type == 'MIDI_PY':
                time = e[1]
              else:
                time = e[0]

              if time == 0:
                cho.append(e)
              
              else:
                if len(cho) > 0:
                  chords.append(cho)
                cho = []
                cho.append(e)

            if len(cho) > 0:
              chords.append(cho)

          requested_data = []

          if return_detected_score_information:
            
            detected_score_information = []

            detected_score_information.append(['Score type', score_type])
            detected_score_information.append(['Score timings type', score_timings_type])
            detected_score_information.append(['Score tpq', score_num_ticks])
            detected_score_information.append(['Score number of tracks', num_tracks])
            
            requested_data.append(detected_score_information)

          if return_choridfied_score and return_detected_score_information:
            requested_data.append(chords)

          if return_choridfied_score and not return_detected_score_information:
            requested_data.extend(chords)

          return requested_data

        except Exception as e:
          print('Error!')
          print('Check score for consistency and compatibility!')
          print('Exception detected:', e)

      else:
        return None

    else:
      return None

def fix_monophonic_score_durations(monophonic_score,
                                   min_notes_gap=1,
                                   min_notes_dur=1,
                                   extend_durs=False
                                   ):
  
    fixed_score = []

    if monophonic_score[0][0] == 'note':

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][1]

        if note[1]+note[2] >= nmt:
          note_dur = max(1, nmt-note[1]-min_notes_gap)
        else:
            if extend_durs:
                note_dur = max(1, nmt-note[1]-min_notes_gap)

            else:
                note_dur = note[2]

        new_note = [note[0], note[1], note_dur] + note[3:]
        
        if new_note[2] >= min_notes_dur:
            fixed_score.append(new_note)
      
      if monophonic_score[-1][2] >= min_notes_dur:
          fixed_score.append(monophonic_score[-1])

    elif type(monophonic_score[0][0]) == int:

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][0]

        if note[0]+note[1] >= nmt:
            note_dur = max(1, nmt-note[0]-min_notes_gap)
        else:
            if extend_durs:
                note_dur = max(1, nmt-note[0]-min_notes_gap)

            else:
                note_dur = note[1]
          
        new_note = [note[0], note_dur] + note[2:]
        
        if new_note[1] >= min_notes_dur:
            fixed_score.append(new_note)
      
      if monophonic_score[-1][1] >= min_notes_dur:
          fixed_score.append(monophonic_score[-1]) 

    return fixed_score

###################################################################################

ALL_CHORDS = [[0], [7], [5], [9], [2], [4], [11], [10], [8], [6], [3], [1], [0, 9], [2, 5],
              [4, 7], [7, 10], [2, 11], [0, 3], [6, 9], [1, 4], [8, 11], [5, 8], [1, 10],
              [3, 6], [0, 4], [5, 9], [7, 11], [0, 7], [0, 5], [2, 10], [2, 7], [2, 9],
              [2, 6], [4, 11], [4, 9], [3, 7], [5, 10], [1, 9], [0, 8], [6, 11], [3, 11],
              [4, 8], [3, 10], [3, 8], [1, 5], [1, 8], [1, 6], [6, 10], [3, 9], [4, 10],
              [1, 7], [0, 6], [2, 8], [5, 11], [5, 7], [0, 10], [0, 2], [9, 11], [7, 9],
              [2, 4], [4, 6], [3, 5], [8, 10], [6, 8], [1, 3], [1, 11], [2, 7, 11],
              [0, 4, 7], [0, 5, 9], [2, 6, 9], [2, 5, 10], [1, 4, 9], [4, 8, 11], [3, 7, 10],
              [0, 3, 8], [3, 6, 11], [1, 5, 8], [1, 6, 10], [0, 4, 9], [2, 5, 9], [4, 7, 11],
              [2, 7, 10], [2, 6, 11], [0, 3, 7], [0, 5, 8], [1, 4, 8], [1, 6, 9], [3, 8, 11],
              [1, 5, 10], [3, 6, 10], [2, 5, 11], [4, 7, 10], [3, 6, 9], [0, 6, 9],
              [0, 3, 9], [2, 8, 11], [2, 5, 8], [1, 7, 10], [1, 4, 7], [0, 3, 6], [1, 4, 10],
              [5, 8, 11], [2, 5, 7], [0, 7, 10], [0, 2, 9], [0, 3, 5], [6, 9, 11], [4, 7, 9],
              [2, 4, 11], [5, 8, 10], [1, 3, 10], [1, 4, 6], [3, 6, 8], [1, 8, 11],
              [5, 7, 11], [0, 4, 10], [3, 5, 9], [0, 2, 6], [1, 7, 9], [0, 7, 9], [5, 7, 10],
              [2, 8, 10], [3, 9, 11], [0, 2, 5], [2, 4, 8], [2, 4, 7], [0, 2, 7], [2, 7, 9],
              [4, 9, 11], [4, 6, 9], [1, 3, 7], [2, 4, 9], [0, 5, 7], [0, 3, 10], [2, 9, 11],
              [0, 5, 10], [0, 6, 8], [4, 6, 10], [4, 6, 11], [1, 4, 11], [6, 8, 11],
              [1, 5, 11], [1, 6, 11], [1, 8, 10], [1, 6, 8], [3, 5, 8], [3, 8, 10],
              [1, 3, 8], [3, 5, 10], [1, 3, 6], [2, 5, 7, 10], [0, 3, 7, 10], [1, 4, 8, 11],
              [2, 4, 7, 11], [0, 4, 7, 9], [0, 2, 5, 9], [2, 6, 9, 11], [1, 5, 8, 10],
              [0, 3, 5, 8], [3, 6, 8, 11], [1, 3, 6, 10], [1, 4, 6, 9], [1, 5, 9], [0, 4, 8],
              [2, 6, 10], [3, 7, 11], [0, 3, 6, 9], [2, 5, 8, 11], [1, 4, 7, 10],
              [2, 5, 7, 11], [0, 2, 6, 9], [0, 4, 7, 10], [2, 4, 8, 11], [0, 3, 5, 9],
              [1, 4, 7, 9], [3, 6, 9, 11], [2, 5, 8, 10], [1, 4, 6, 10], [0, 3, 6, 8],
              [1, 3, 7, 10], [1, 5, 8, 11], [2, 4, 10], [5, 9, 11], [1, 5, 7], [0, 2, 8],
              [0, 4, 6], [1, 7, 11], [3, 7, 9], [1, 3, 9], [7, 9, 11], [5, 7, 9], [0, 6, 10],
              [0, 2, 10], [2, 6, 8], [0, 2, 4], [4, 8, 10], [1, 9, 11], [2, 4, 6],
              [3, 5, 11], [3, 5, 7], [0, 8, 10], [4, 6, 8], [1, 3, 11], [6, 8, 10],
              [1, 3, 5], [0, 2, 5, 10], [0, 5, 7, 9], [0, 3, 8, 10], [0, 2, 4, 7],
              [4, 6, 8, 11], [3, 5, 7, 10], [2, 7, 9, 11], [2, 4, 6, 9], [1, 6, 8, 10],
              [1, 4, 9, 11], [1, 3, 5, 8], [1, 3, 6, 11], [2, 5, 9, 11], [2, 4, 7, 10],
              [0, 2, 5, 8], [1, 5, 7, 10], [0, 4, 6, 9], [1, 3, 6, 9], [0, 3, 6, 10],
              [2, 6, 8, 11], [0, 2, 7, 9], [1, 4, 8, 10], [0, 3, 7, 9], [3, 5, 8, 11],
              [0, 5, 7, 10], [0, 2, 5, 7], [1, 4, 7, 11], [2, 4, 7, 9], [0, 3, 5, 10],
              [4, 6, 9, 11], [1, 4, 6, 11], [2, 4, 9, 11], [1, 6, 8, 11], [1, 3, 6, 8],
              [1, 3, 8, 10], [3, 5, 8, 10], [4, 7, 9, 11], [0, 2, 7, 10], [2, 5, 7, 9],
              [0, 2, 4, 9], [1, 6, 9, 11], [2, 4, 6, 11], [0, 3, 5, 7], [0, 5, 8, 10],
              [1, 4, 6, 8], [1, 3, 5, 10], [1, 3, 8, 11], [3, 6, 8, 10], [0, 2, 5, 7, 10],
              [0, 2, 4, 7, 9], [0, 2, 5, 7, 9], [1, 3, 7, 9], [1, 4, 6, 9, 11],
              [1, 3, 6, 8, 11], [3, 5, 9, 11], [1, 3, 6, 8, 10], [1, 4, 6, 8, 11],
              [1, 3, 5, 8, 10], [2, 4, 6, 9, 11], [2, 4, 8, 10], [2, 4, 7, 9, 11],
              [0, 3, 5, 7, 10], [1, 5, 7, 11], [0, 2, 6, 8], [0, 3, 5, 8, 10], [0, 4, 6, 10],
              [1, 3, 5, 9], [1, 5, 7, 9], [2, 6, 8, 10], [3, 7, 9, 11], [0, 2, 4, 8],
              [0, 4, 6, 8], [0, 4, 8, 10], [2, 4, 6, 10], [1, 3, 7, 11], [0, 2, 6, 10],
              [1, 5, 9, 11], [3, 5, 7, 11], [1, 7, 9, 11], [0, 2, 4, 6], [1, 3, 9, 11],
              [0, 2, 4, 10], [5, 7, 9, 11], [2, 4, 6, 8], [0, 2, 8, 10], [3, 5, 7, 9],
              [1, 3, 5, 7], [4, 6, 8, 10], [0, 6, 8, 10], [1, 3, 5, 11], [0, 3, 6, 8, 10],
              [0, 2, 4, 6, 9], [1, 4, 7, 9, 11], [2, 4, 6, 8, 11], [1, 3, 6, 9, 11],
              [1, 3, 5, 8, 11], [0, 2, 5, 8, 10], [1, 4, 6, 8, 10], [0, 3, 5, 7, 9],
              [2, 5, 7, 9, 11], [1, 3, 5, 7, 10], [0, 2, 4, 7, 10], [1, 3, 5, 7, 9],
              [1, 3, 5, 9, 11], [1, 5, 7, 9, 11], [1, 3, 7, 9, 11], [3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10], [0, 4, 6, 8, 10], [0, 2, 6, 8, 10], [1, 3, 5, 7, 11],
              [0, 2, 4, 8, 10], [0, 2, 4, 6, 8], [0, 2, 4, 6, 10], [0, 2, 4, 6, 8, 10],
              [1, 3, 5, 7, 9, 11]]

def find_exact_match_variable_length(list_of_lists, target_list, uncertain_indices):
    # Infer possible values for each uncertain index
    possible_values = {idx: set() for idx in uncertain_indices}
    for sublist in list_of_lists:
        for idx in uncertain_indices:
            if idx < len(sublist):
                possible_values[idx].add(sublist[idx])
    
    # Generate all possible combinations for the uncertain elements
    uncertain_combinations = product(*(possible_values[idx] for idx in uncertain_indices))
    
    for combination in uncertain_combinations:
        # Create a copy of the target list and update the uncertain elements
        test_list = target_list[:]
        for idx, value in zip(uncertain_indices, combination):
            test_list[idx] = value
        
        # Check if the modified target list is an exact match in the list of lists
        # Only consider sublists that are at least as long as the target list
        for sublist in list_of_lists:
            if len(sublist) >= len(test_list) and sublist[:len(test_list)] == test_list:
                return sublist  # Return the matching sublist
    
    return None  # No exact match found


def advanced_validate_chord_pitches(chord, channel_to_check = 0, return_sorted = True):

    pitches_chord = sorted(list(set([x[4] for x in chord if 0 < x[4] < 128 and x[3] == channel_to_check])))

    if pitches_chord:

      tones_chord = sorted(list(set([c % 12 for c in sorted(list(set(pitches_chord)))])))

      if not bad_chord(tones_chord):
        if return_sorted:
          chord.sort(key = lambda x: x[4], reverse=True)
        return chord

      else:
        bad_chord_indices = list(set([i for s in [[tones_chord.index(a), tones_chord.index(b)] for a, b in zip(tones_chord, tones_chord[1:]) if b-a == 1] for i in s]))
        
        good_tones_chord = find_exact_match_variable_length(ALL_CHORDS, tones_chord, bad_chord_indices)
        
        if good_tones_chord is not None:
        
          fixed_chord = []

          for c in chord:
            if c[3] == channel_to_check:
              if (c[4] % 12) in good_tones_chord:
                fixed_chord.append(c)
            else:
              fixed_chord.append(c)

          if return_sorted:
            fixed_chord.sort(key = lambda x: x[4], reverse=True)

        else:

          if 0 in tones_chord and 11 in tones_chord:
            tones_chord.remove(0)

          fixed_tones = [[a, b] for a, b in zip(tones_chord, tones_chord[1:]) if b-a != 1]

          fixed_tones_chord = []
          for f in fixed_tones:
            fixed_tones_chord.extend(f)
          fixed_tones_chord = list(set(fixed_tones_chord))
          
          fixed_chord = []

          for c in chord:
            if c[3] == channel_to_check:
              if (c[4] % 12) in fixed_tones_chord:
                fixed_chord.append(c)
            else:
              fixed_chord.append(c)

          if return_sorted:
            fixed_chord.sort(key = lambda x: x[4], reverse=True)     
      
      return fixed_chord 

    else:
      chord.sort(key = lambda x: x[4], reverse=True)
      return chord

###################################################################################

def analyze_score_pitches(score, channels_to_analyze=[0]):

  analysis = {}

  score_notes = [s for s in score if s[3] in channels_to_analyze]

  cscore = chordify_score(score_notes)

  chords_tones = []

  all_tones = []

  all_chords_good = True

  bad_chords = []

  for c in cscore:
    tones = sorted(list(set([t[4] % 12 for t in c])))
    chords_tones.append(tones)
    all_tones.extend(tones)

    if tones not in ALL_CHORDS:
      all_chords_good = False
      bad_chords.append(tones)

  analysis['Number of notes'] = len(score_notes)
  analysis['Number of chords'] = len(cscore)
  analysis['Score tones'] = sorted(list(set(all_tones)))
  analysis['Shortest chord'] = sorted(min(chords_tones, key=len))
  analysis['Longest chord'] = sorted(max(chords_tones, key=len))
  analysis['All chords good'] = all_chords_good
  analysis['Bad chords'] = bad_chords

  return analysis

###################################################################################

ALL_CHORDS_GROUPED = [[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]],
                      [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10],
                        [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11],
                        [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [3, 5],
                        [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [4, 6], [4, 7], [4, 8],
                        [4, 9], [4, 10], [4, 11], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [6, 8],
                        [6, 9], [6, 10], [6, 11], [7, 9], [7, 10], [7, 11], [8, 10], [8, 11],
                        [9, 11]],
                      [[0, 2, 4], [0, 2, 5], [0, 3, 5], [0, 2, 6], [0, 3, 6], [0, 4, 6], [0, 2, 7],
                        [0, 3, 7], [0, 4, 7], [0, 5, 7], [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8],
                        [0, 6, 8], [0, 2, 9], [0, 3, 9], [0, 4, 9], [0, 5, 9], [0, 6, 9], [0, 7, 9],
                        [0, 2, 10], [0, 3, 10], [0, 4, 10], [0, 5, 10], [0, 6, 10], [0, 7, 10],
                        [0, 8, 10], [1, 3, 5], [1, 3, 6], [1, 4, 6], [1, 3, 7], [1, 4, 7], [1, 5, 7],
                        [1, 3, 8], [1, 4, 8], [1, 5, 8], [1, 6, 8], [1, 3, 9], [1, 4, 9], [1, 5, 9],
                        [1, 6, 9], [1, 7, 9], [1, 3, 10], [1, 4, 10], [1, 5, 10], [1, 6, 10],
                        [1, 7, 10], [1, 8, 10], [1, 3, 11], [1, 4, 11], [1, 5, 11], [1, 6, 11],
                        [1, 7, 11], [1, 8, 11], [1, 9, 11], [2, 4, 6], [2, 4, 7], [2, 5, 7],
                        [2, 4, 8], [2, 5, 8], [2, 6, 8], [2, 4, 9], [2, 5, 9], [2, 6, 9], [2, 7, 9],
                        [2, 4, 10], [2, 5, 10], [2, 6, 10], [2, 7, 10], [2, 8, 10], [2, 4, 11],
                        [2, 5, 11], [2, 6, 11], [2, 7, 11], [2, 8, 11], [2, 9, 11], [3, 5, 7],
                        [3, 5, 8], [3, 6, 8], [3, 5, 9], [3, 6, 9], [3, 7, 9], [3, 5, 10], [3, 6, 10],
                        [3, 7, 10], [3, 8, 10], [3, 5, 11], [3, 6, 11], [3, 7, 11], [3, 8, 11],
                        [3, 9, 11], [4, 6, 8], [4, 6, 9], [4, 7, 9], [4, 6, 10], [4, 7, 10],
                        [4, 8, 10], [4, 6, 11], [4, 7, 11], [4, 8, 11], [4, 9, 11], [5, 7, 9],
                        [5, 7, 10], [5, 8, 10], [5, 7, 11], [5, 8, 11], [5, 9, 11], [6, 8, 10],
                        [6, 8, 11], [6, 9, 11], [7, 9, 11]],
                      [[0, 2, 4, 6], [0, 2, 4, 7], [0, 2, 5, 7], [0, 3, 5, 7], [0, 2, 4, 8],
                        [0, 2, 5, 8], [0, 2, 6, 8], [0, 3, 5, 8], [0, 3, 6, 8], [0, 4, 6, 8],
                        [0, 2, 4, 9], [0, 2, 5, 9], [0, 2, 6, 9], [0, 2, 7, 9], [0, 3, 5, 9],
                        [0, 3, 6, 9], [0, 3, 7, 9], [0, 4, 6, 9], [0, 4, 7, 9], [0, 5, 7, 9],
                        [0, 2, 4, 10], [0, 2, 5, 10], [0, 2, 6, 10], [0, 2, 7, 10], [0, 2, 8, 10],
                        [0, 3, 5, 10], [0, 3, 6, 10], [0, 3, 7, 10], [0, 3, 8, 10], [0, 4, 6, 10],
                        [0, 4, 7, 10], [0, 4, 8, 10], [0, 5, 7, 10], [0, 5, 8, 10], [0, 6, 8, 10],
                        [1, 3, 5, 7], [1, 3, 5, 8], [1, 3, 6, 8], [1, 4, 6, 8], [1, 3, 5, 9],
                        [1, 3, 6, 9], [1, 3, 7, 9], [1, 4, 6, 9], [1, 4, 7, 9], [1, 5, 7, 9],
                        [1, 3, 5, 10], [1, 3, 6, 10], [1, 3, 7, 10], [1, 3, 8, 10], [1, 4, 6, 10],
                        [1, 4, 7, 10], [1, 4, 8, 10], [1, 5, 7, 10], [1, 5, 8, 10], [1, 6, 8, 10],
                        [1, 3, 5, 11], [1, 3, 6, 11], [1, 3, 7, 11], [1, 3, 8, 11], [1, 3, 9, 11],
                        [1, 4, 6, 11], [1, 4, 7, 11], [1, 4, 8, 11], [1, 4, 9, 11], [1, 5, 7, 11],
                        [1, 5, 8, 11], [1, 5, 9, 11], [1, 6, 8, 11], [1, 6, 9, 11], [1, 7, 9, 11],
                        [2, 4, 6, 8], [2, 4, 6, 9], [2, 4, 7, 9], [2, 5, 7, 9], [2, 4, 6, 10],
                        [2, 4, 7, 10], [2, 4, 8, 10], [2, 5, 7, 10], [2, 5, 8, 10], [2, 6, 8, 10],
                        [2, 4, 6, 11], [2, 4, 7, 11], [2, 4, 8, 11], [2, 4, 9, 11], [2, 5, 7, 11],
                        [2, 5, 8, 11], [2, 5, 9, 11], [2, 6, 8, 11], [2, 6, 9, 11], [2, 7, 9, 11],
                        [3, 5, 7, 9], [3, 5, 7, 10], [3, 5, 8, 10], [3, 6, 8, 10], [3, 5, 7, 11],
                        [3, 5, 8, 11], [3, 5, 9, 11], [3, 6, 8, 11], [3, 6, 9, 11], [3, 7, 9, 11],
                        [4, 6, 8, 10], [4, 6, 8, 11], [4, 6, 9, 11], [4, 7, 9, 11], [5, 7, 9, 11]],
                      [[0, 2, 4, 6, 8], [0, 2, 4, 6, 9], [0, 2, 4, 7, 9], [0, 2, 5, 7, 9],
                        [0, 3, 5, 7, 9], [0, 2, 4, 6, 10], [0, 2, 4, 7, 10], [0, 2, 4, 8, 10],
                        [0, 2, 5, 7, 10], [0, 2, 5, 8, 10], [0, 2, 6, 8, 10], [0, 3, 5, 7, 10],
                        [0, 3, 5, 8, 10], [0, 3, 6, 8, 10], [0, 4, 6, 8, 10], [1, 3, 5, 7, 9],
                        [1, 3, 5, 7, 10], [1, 3, 5, 8, 10], [1, 3, 6, 8, 10], [1, 4, 6, 8, 10],
                        [1, 3, 5, 7, 11], [1, 3, 5, 8, 11], [1, 3, 5, 9, 11], [1, 3, 6, 8, 11],
                        [1, 3, 6, 9, 11], [1, 3, 7, 9, 11], [1, 4, 6, 8, 11], [1, 4, 6, 9, 11],
                        [1, 4, 7, 9, 11], [1, 5, 7, 9, 11], [2, 4, 6, 8, 10], [2, 4, 6, 8, 11],
                        [2, 4, 6, 9, 11], [2, 4, 7, 9, 11], [2, 5, 7, 9, 11], [3, 5, 7, 9, 11]],
                      [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]]]

def group_sublists_by_length(lst):
    unique_lengths = sorted(list(set(map(len, lst))), reverse=True)
    return [[x for x in lst if len(x) == i] for i in unique_lengths]

def pitches_to_tones_chord(pitches):
  return sorted(set([p % 12 for p in pitches]))

def tones_chord_to_pitches(tones_chord, base_pitch=60):
  return [t+base_pitch for t in tones_chord if 0 <= t < 12]

###################################################################################

def advanced_score_processor(raw_score, 
                             patches_to_analyze=list(range(129)), 
                             return_score_analysis=False,
                             return_enhanced_score=False,
                             return_enhanced_score_notes=False,
                             return_enhanced_monophonic_melody=False,
                             return_chordified_enhanced_score=False,
                             return_chordified_enhanced_score_with_lyrics=False,
                             return_score_tones_chords=False,
                             return_text_and_lyric_events=False,
                             apply_sustain=False  
                            ):

  '''TMIDIX Advanced Score Processor'''

  # Score data types detection

  if raw_score and type(raw_score) == list:

      num_ticks = 0
      num_tracks = 1

      basic_single_track_score = []

      if type(raw_score[0]) != int:
        if len(raw_score[0]) < 5 and type(raw_score[0][0]) != str:
          return ['Check score for errors and compatibility!']

        else:
          basic_single_track_score = copy.deepcopy(raw_score)
      
      else:
        num_ticks = raw_score[0]
        while num_tracks < len(raw_score):
            for event in raw_score[num_tracks]:
              ev = copy.deepcopy(event)
              basic_single_track_score.append(ev)
            num_tracks += 1

      for e in basic_single_track_score:

          if e[0] == 'note':
              e[3] = e[3] % 16
              e[4] = e[4] % 128
              e[5] = e[5] % 128

          if e[0] == 'patch_change':
              e[2] = e[2] % 16
              e[3] = e[3] % 128

      if apply_sustain:
          apply_sustain_to_ms_score([1000, basic_single_track_score])
          
      basic_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      basic_single_track_score.sort(key=lambda x: x[1])

      enhanced_single_track_score = []
      patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      all_score_patches = []
      num_patch_changes = 0

      for event in basic_single_track_score:
        if event[0] == 'patch_change':
              patches[event[2]] = event[3]
              enhanced_single_track_score.append(event)
              num_patch_changes += 1

        if event[0] == 'note':            
            if event[3] != 9:
              event.extend([patches[event[3]]])
              all_score_patches.extend([patches[event[3]]])
            else:
              event.extend([128])
              all_score_patches.extend([128])

            if enhanced_single_track_score:
                if (event[1] == enhanced_single_track_score[-1][1]):
                    if ([event[3], event[4]] != enhanced_single_track_score[-1][3:5]):
                        enhanced_single_track_score.append(event)
                else:
                    enhanced_single_track_score.append(event)

            else:
                enhanced_single_track_score.append(event)

        if event[0] not in ['note', 'patch_change']:
          enhanced_single_track_score.append(event)

      enhanced_single_track_score.sort(key=lambda x: x[6] if x[0] == 'note' else -1)
      enhanced_single_track_score.sort(key=lambda x: x[4] if x[0] == 'note' else 128, reverse=True)
      enhanced_single_track_score.sort(key=lambda x: x[1])

      # Analysis and chordification

      cscore = []
      cescore = []
      chords_tones = []
      tones_chords = []
      all_tones = []
      all_chords_good = True
      bad_chords = []
      bad_chords_count = 0
      score_notes = []
      score_pitches = []
      score_patches = []
      num_text_events = 0
      num_lyric_events = 0
      num_other_events = 0
      text_and_lyric_events = []
      text_and_lyric_events_latin = None

      analysis = {}

      score_notes = [s for s in enhanced_single_track_score if s[0] == 'note' and s[6] in patches_to_analyze]
      score_patches = [sn[6] for sn in score_notes]

      if return_text_and_lyric_events:
        text_and_lyric_events = [e for e in enhanced_single_track_score if e[0] in ['text_event', 'lyric']]
        
        if text_and_lyric_events:
          text_and_lyric_events_latin = True
          for e in text_and_lyric_events:
            try:
              tle = str(e[2].decode())
            except:
              tle = str(e[2])

            for c in tle:
              if not 0 <= ord(c) < 128:
                text_and_lyric_events_latin = False

      if (return_chordified_enhanced_score or return_score_analysis) and any(elem in patches_to_analyze for elem in score_patches):

        cescore = chordify_score([num_ticks, enhanced_single_track_score])

        if return_score_analysis:

          cscore = chordify_score(score_notes)
          
          score_pitches = [sn[4] for sn in score_notes]
          
          text_events = [e for e in enhanced_single_track_score if e[0] == 'text_event']
          num_text_events = len(text_events)

          lyric_events = [e for e in enhanced_single_track_score if e[0] == 'lyric']
          num_lyric_events = len(lyric_events)

          other_events = [e for e in enhanced_single_track_score if e[0] not in ['note', 'patch_change', 'text_event', 'lyric']]
          num_other_events = len(other_events)
          
          for c in cscore:
            tones = sorted(set([t[4] % 12 for t in c if t[3] != 9]))

            if tones:
              chords_tones.append(tones)
              all_tones.extend(tones)

              if tones not in ALL_CHORDS:
                all_chords_good = False
                bad_chords.append(tones)
                bad_chords_count += 1
          
          analysis['Number of ticks per quarter note'] = num_ticks
          analysis['Number of tracks'] = num_tracks
          analysis['Number of all events'] = len(enhanced_single_track_score)
          analysis['Number of patch change events'] = num_patch_changes
          analysis['Number of text events'] = num_text_events
          analysis['Number of lyric events'] = num_lyric_events
          analysis['All text and lyric events Latin'] = text_and_lyric_events_latin
          analysis['Number of other events'] = num_other_events
          analysis['Number of score notes'] = len(score_notes)
          analysis['Number of score chords'] = len(cscore)
          analysis['Score patches'] = sorted(set(score_patches))
          analysis['Score pitches'] = sorted(set(score_pitches))
          analysis['Score tones'] = sorted(set(all_tones))
          if chords_tones:
            analysis['Shortest chord'] = sorted(min(chords_tones, key=len))
            analysis['Longest chord'] = sorted(max(chords_tones, key=len))
          analysis['All chords good'] = all_chords_good
          analysis['Number of bad chords'] = bad_chords_count
          analysis['Bad chords'] = sorted([list(c) for c in set(tuple(bc) for bc in bad_chords)])

      else:
        analysis['Error'] = 'Provided score does not have specified patches to analyse'
        analysis['Provided patches to analyse'] = sorted(patches_to_analyze)
        analysis['Patches present in the score'] = sorted(set(all_score_patches))

      if return_enhanced_monophonic_melody:

        score_notes_copy = copy.deepcopy(score_notes)
        chordified_score_notes = chordify_score(score_notes_copy)

        melody = [c[0] for c in chordified_score_notes]

        fixed_melody = []

        for i in range(len(melody)-1):
          note = melody[i]
          nmt = melody[i+1][1]

          if note[1]+note[2] >= nmt:
            note_dur = nmt-note[1]-1
          else:
            note_dur = note[2]

          melody[i][2] = note_dur

          fixed_melody.append(melody[i])
        fixed_melody.append(melody[-1])

      if return_score_tones_chords:
        cscore = chordify_score(score_notes)
        for c in cscore:
          tones_chord = sorted(set([t[4] % 12 for t in c if t[3] != 9]))
          if tones_chord:
            tones_chords.append(tones_chord)

      if return_chordified_enhanced_score_with_lyrics:
        score_with_lyrics = [e for e in enhanced_single_track_score if e[0] in ['note', 'text_event', 'lyric']]
        chordified_enhanced_score_with_lyrics = chordify_score(score_with_lyrics)
      
      # Returned data

      requested_data = []

      if return_score_analysis and analysis:
        requested_data.append([[k, v] for k, v in analysis.items()])

      if return_enhanced_score and enhanced_single_track_score:
        requested_data.append([num_ticks, enhanced_single_track_score])

      if return_enhanced_score_notes and score_notes:
        requested_data.append(score_notes)

      if return_enhanced_monophonic_melody and fixed_melody:
        requested_data.append(fixed_melody)
        
      if return_chordified_enhanced_score and cescore:
        requested_data.append(cescore)

      if return_chordified_enhanced_score_with_lyrics and chordified_enhanced_score_with_lyrics:
        requested_data.append(chordified_enhanced_score_with_lyrics)

      if return_score_tones_chords and tones_chords:
        requested_data.append(tones_chords)

      if return_text_and_lyric_events and text_and_lyric_events:
        requested_data.append(text_and_lyric_events)

      return requested_data
  
  else:
    return ['Check score for errors and compatibility!']

###################################################################################

import random
import copy

###################################################################################

def replace_bad_tones_chord(bad_tones_chord):
  bad_chord_p = [0] * 12
  for b in bad_tones_chord:
    bad_chord_p[b] = 1

  match_ratios = []
  good_chords = []
  for c in ALL_CHORDS:
    good_chord_p = [0] * 12
    for cc in c:
      good_chord_p[cc] = 1

    good_chords.append(good_chord_p)
    match_ratios.append(sum(i == j for i, j in zip(good_chord_p, bad_chord_p)) / len(good_chord_p))

  best_good_chord = good_chords[match_ratios.index(max(match_ratios))]

  replaced_chord = []
  for i in range(len(best_good_chord)):
    if best_good_chord[i] == 1:
     replaced_chord.append(i)

  return [replaced_chord, max(match_ratios)]

###################################################################################

def check_and_fix_chord(chord, 
                        channel_index=3,
                        pitch_index=4
                        ):

    tones_chord = sorted(set([t[pitch_index] % 12 for t in chord if t[channel_index] != 9]))

    notes_events = [t for t in chord if t[channel_index] != 9]
    notes_events.sort(key=lambda x: x[pitch_index], reverse=True)

    drums_events = [t for t in chord if t[channel_index] == 9]

    checked_and_fixed_chord = []

    if tones_chord:
        
        new_tones_chord = advanced_check_and_fix_tones_chord(tones_chord, high_pitch=notes_events[0][pitch_index])

        if new_tones_chord != tones_chord:

          if len(notes_events) > 1:
              checked_and_fixed_chord.extend([notes_events[0]])
              for cc in notes_events[1:]:
                  if cc[channel_index] != 9:
                      if (cc[pitch_index] % 12) in new_tones_chord:
                          checked_and_fixed_chord.extend([cc])
              checked_and_fixed_chord.extend(drums_events)
          else:
              checked_and_fixed_chord.extend([notes_events[0]])
        else:
          checked_and_fixed_chord.extend(chord)
    else:
        checked_and_fixed_chord.extend(chord)

    checked_and_fixed_chord.sort(key=lambda x: x[pitch_index], reverse=True)

    return checked_and_fixed_chord

###################################################################################

def find_similar_tones_chord(tones_chord, 
                             max_match_threshold=1, 
                             randomize_chords_matches=False, 
                             custom_chords_list=[]):
  chord_p = [0] * 12
  for b in tones_chord:
    chord_p[b] = 1

  match_ratios = []
  good_chords = []

  if custom_chords_list:
    CHORDS = copy.deepcopy([list(x) for x in set(tuple(t) for t in custom_chords_list)])
  else:
    CHORDS = copy.deepcopy(ALL_CHORDS)

  if randomize_chords_matches:
    random.shuffle(CHORDS)

  for c in CHORDS:
    good_chord_p = [0] * 12
    for cc in c:
      good_chord_p[cc] = 1

    good_chords.append(good_chord_p)
    match_ratio = sum(i == j for i, j in zip(good_chord_p, chord_p)) / len(good_chord_p)
    if match_ratio < max_match_threshold:
      match_ratios.append(match_ratio)
    else:
      match_ratios.append(0)

  best_good_chord = good_chords[match_ratios.index(max(match_ratios))]

  similar_chord = []
  for i in range(len(best_good_chord)):
    if best_good_chord[i] == 1:
     similar_chord.append(i)

  return [similar_chord, max(match_ratios)]

###################################################################################

def generate_tones_chords_progression(number_of_chords_to_generate=100, 
                                      start_tones_chord=[], 
                                      custom_chords_list=[]):

  if start_tones_chord:
    start_chord = start_tones_chord
  else:
    start_chord = random.choice(ALL_CHORDS)

  chord = []

  chords_progression = [start_chord]

  for i in range(number_of_chords_to_generate):
    if not chord:
      chord = start_chord

    if custom_chords_list:
      chord = find_similar_tones_chord(chord, randomize_chords_matches=True, custom_chords_list=custom_chords_list)[0]
    else:
      chord = find_similar_tones_chord(chord, randomize_chords_matches=True)[0]
    
    chords_progression.append(chord)

  return chords_progression

###################################################################################

def ascii_texts_search(texts = ['text1', 'text2', 'text3'],
                       search_query = 'Once upon a time...',
                       deterministic_matching = False
                       ):

    texts_copy = texts

    if not deterministic_matching:
      texts_copy = copy.deepcopy(texts)
      random.shuffle(texts_copy)

    clean_texts = []

    for t in texts_copy:
      text_words_list = [at.split(chr(32)) for at in t.split(chr(10))]
      
      clean_text_words_list = []
      for twl in text_words_list:
        for w in twl:
          clean_text_words_list.append(''.join(filter(str.isalpha, w.lower())))
          
      clean_texts.append(clean_text_words_list)

    text_search_query = [at.split(chr(32)) for at in search_query.split(chr(10))]
    clean_text_search_query = []
    for w in text_search_query:
      for ww in w:
        clean_text_search_query.append(''.join(filter(str.isalpha, ww.lower())))

    if clean_texts[0] and clean_text_search_query:
      texts_match_ratios = []
      words_match_indexes = []
      for t in clean_texts:
        word_match_count = 0
        wmis = []

        for c in clean_text_search_query:
          if c in t:
            word_match_count += 1
            wmis.append(t.index(c))
          else:
            wmis.append(-1)

        words_match_indexes.append(wmis)
        words_match_indexes_consequtive = all(abs(b) - abs(a) == 1 for a, b in zip(wmis, wmis[1:]))
        words_match_indexes_consequtive_ratio = sum([abs(b) - abs(a) == 1 for a, b in zip(wmis, wmis[1:])]) / len(wmis)

        if words_match_indexes_consequtive:
          texts_match_ratios.append(word_match_count / len(clean_text_search_query))
        else:
          texts_match_ratios.append(((word_match_count / len(clean_text_search_query)) + words_match_indexes_consequtive_ratio) / 2)

      if texts_match_ratios:
        max_text_match_ratio = max(texts_match_ratios)
        max_match_ratio_text = texts_copy[texts_match_ratios.index(max_text_match_ratio)]
        max_text_words_match_indexes = words_match_indexes[texts_match_ratios.index(max_text_match_ratio)]

      return [max_match_ratio_text, max_text_match_ratio, max_text_words_match_indexes]
    
    else:
      return None

###################################################################################

def ascii_text_words_counter(ascii_text):

    text_words_list = [at.split(chr(32)) for at in ascii_text.split(chr(10))]

    clean_text_words_list = []
    for twl in text_words_list:
      for w in twl:
        wo = ''
        for ww in w.lower():
          if 96 < ord(ww) < 123:
            wo += ww
        if wo != '':
          clean_text_words_list.append(wo)

    words = {}
    for i in clean_text_words_list:
        words[i] = words.get(i, 0) + 1

    words_sorted = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))

    return len(clean_text_words_list), words_sorted, clean_text_words_list
    
###################################################################################

def check_and_fix_tones_chord(tones_chord, use_full_chords=True):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  else:
    CHORDS = ALL_CHORDS_SORTED

  for c in tones_chord_combs:
    if c in CHORDS:
      checked_tones_chord = c
      break

  return sorted(checked_tones_chord)

###################################################################################

def find_closest_tone(tones, tone):
  return min(tones, key=lambda x:abs(x-tone))

###################################################################################

def advanced_check_and_fix_tones_chord(tones_chord, high_pitch=0, use_full_chords=True):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  else:
    CHORDS = ALL_CHORDS_SORTED

  for c in tones_chord_combs:
    if c in CHORDS:
      tchord = c

  if 0 < high_pitch < 128 and len(tchord) == 1:
    tchord = [high_pitch % 12]

  return tchord

###################################################################################

def create_similarity_matrix(list_of_values, matrix_length=0):

    counts = Counter(list_of_values).items()

    if matrix_length > 0:
      sim_matrix = [0] * max(matrix_length, len(list_of_values))
    else:
      sim_matrix = [0] * len(counts)

    for c in counts:
      sim_matrix[c[0]] = c[1]

    similarity_matrix = [[0] * len(sim_matrix) for _ in range(len(sim_matrix))]

    for i in range(len(sim_matrix)):
      for j in range(len(sim_matrix)):
        if max(sim_matrix[i], sim_matrix[j]) != 0:
          similarity_matrix[i][j] = min(sim_matrix[i], sim_matrix[j]) / max(sim_matrix[i], sim_matrix[j])

    return similarity_matrix, sim_matrix

###################################################################################

def ceil_with_precision(value, decimal_places):
    factor = 10 ** decimal_places
    return math.ceil(value * factor) / factor

###################################################################################

def augment_enhanced_score_notes(enhanced_score_notes,
                                  timings_divider=16,
                                  full_sorting=True,
                                  timings_shift=0,
                                  pitch_shift=0,
                                  ceil_timings=False,
                                  round_timings=False,
                                  legacy_timings=True,
                                  sort_drums_last=False,
                                  even_timings=False
                                ):

    esn = copy.deepcopy(enhanced_score_notes)

    pe = enhanced_score_notes[0]

    abs_time = max(0, int(enhanced_score_notes[0][1] / timings_divider))

    for i, e in enumerate(esn):
      
      dtime = (e[1] / timings_divider) - (pe[1] / timings_divider)

      if round_timings:
        dtime = round(dtime)
      
      else:
        if ceil_timings:
          dtime = math.ceil(dtime)
        
        else:
          dtime = int(dtime)

      if legacy_timings:
        abs_time = int(e[1] / timings_divider) + timings_shift

      else:
        abs_time += dtime

      e[1] = max(0, abs_time + timings_shift)

      if round_timings:
        e[2] = max(1, round(e[2] / timings_divider)) + timings_shift
      
      else:
        if ceil_timings:
          e[2] = max(1, math.ceil(e[2] / timings_divider)) + timings_shift
        else:
          e[2] = max(1, int(e[2] / timings_divider)) + timings_shift
      
      e[4] = max(1, min(127, e[4] + pitch_shift))

      pe = enhanced_score_notes[i]
      
      
    if even_timings:
        
      for e in esn:
          if e[1] % 2 != 0:
              e[1] += 1
          
          if e[2] % 2 != 0:
              e[2] += 1

    if full_sorting:

      # Sorting by patch, reverse pitch and start-time
      esn.sort(key=lambda x: x[6])
      esn.sort(key=lambda x: x[4], reverse=True)
      esn.sort(key=lambda x: x[1])
      
    if sort_drums_last:
        esn.sort(key=lambda x: (x[1], -x[4], x[6]) if x[6] != 128 else (x[1], x[6], -x[4]))

    return esn

###################################################################################

def stack_list(lst, base=12):
    return sum(j * base**i for i, j in enumerate(lst[::-1]))

def destack_list(num, base=12):
    lst = []
    while num:
        lst.append(num % base)
        num //= base
    return lst[::-1]

###################################################################################

def extract_melody(chordified_enhanced_score, 
                    melody_range=[48, 84], 
                    melody_channel=0,
                    melody_patch=0,
                    melody_velocity=0,
                    stacked_melody=False,
                    stacked_melody_base_pitch=60
                  ):

    if stacked_melody:

      
      all_pitches_chords = []
      for e in chordified_enhanced_score:
        all_pitches_chords.append(sorted(set([p[4] for p in e]), reverse=True))
      
      melody_score = []
      for i, chord in enumerate(chordified_enhanced_score):

        if melody_velocity > 0:
          vel = melody_velocity
        else:
          vel = chord[0][5]

        melody_score.append(['note', chord[0][1], chord[0][2], melody_channel, stacked_melody_base_pitch+(stack_list([p % 12 for p in all_pitches_chords[i]]) % 12), vel, melody_patch])
  
    else:

      melody_score = copy.deepcopy([c[0] for c in chordified_enhanced_score if c[0][3] != 9])
      
      for e in melody_score:
        
          e[3] = melody_channel

          if melody_velocity > 0:
            e[5] = melody_velocity

          e[6] = melody_patch

          if e[4] < melody_range[0]:
              e[4] = (e[4] % 12) + melody_range[0]
              
          if e[4] >= melody_range[1]:
              e[4] = (e[4] % 12) + (melody_range[1]-12)

    return fix_monophonic_score_durations(melody_score)

###################################################################################

def flip_enhanced_score_notes(enhanced_score_notes):

    min_pitch = min([e[4] for e in enhanced_score_notes if e[3] != 9])

    fliped_score_pitches = [127 - e[4]for e in enhanced_score_notes if e[3] != 9]

    delta_min_pitch = min_pitch - min([p for p in fliped_score_pitches])

    output_score = copy.deepcopy(enhanced_score_notes)

    for e in output_score:
        if e[3] != 9:
            e[4] = (127 - e[4]) + delta_min_pitch

    return output_score

###################################################################################

ALL_CHORDS_SORTED = [[0], [0, 2], [0, 3], [0, 4], [0, 2, 4], [0, 5], [0, 2, 5], [0, 3, 5], [0, 6],
                    [0, 2, 6], [0, 3, 6], [0, 4, 6], [0, 2, 4, 6], [0, 7], [0, 2, 7], [0, 3, 7],
                    [0, 4, 7], [0, 5, 7], [0, 2, 4, 7], [0, 2, 5, 7], [0, 3, 5, 7], [0, 8],
                    [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8], [0, 6, 8], [0, 2, 4, 8],
                    [0, 2, 5, 8], [0, 2, 6, 8], [0, 3, 5, 8], [0, 3, 6, 8], [0, 4, 6, 8],
                    [0, 2, 4, 6, 8], [0, 9], [0, 2, 9], [0, 3, 9], [0, 4, 9], [0, 5, 9], [0, 6, 9],
                    [0, 7, 9], [0, 2, 4, 9], [0, 2, 5, 9], [0, 2, 6, 9], [0, 2, 7, 9],
                    [0, 3, 5, 9], [0, 3, 6, 9], [0, 3, 7, 9], [0, 4, 6, 9], [0, 4, 7, 9],
                    [0, 5, 7, 9], [0, 2, 4, 6, 9], [0, 2, 4, 7, 9], [0, 2, 5, 7, 9],
                    [0, 3, 5, 7, 9], [0, 10], [0, 2, 10], [0, 3, 10], [0, 4, 10], [0, 5, 10],
                    [0, 6, 10], [0, 7, 10], [0, 8, 10], [0, 2, 4, 10], [0, 2, 5, 10],
                    [0, 2, 6, 10], [0, 2, 7, 10], [0, 2, 8, 10], [0, 3, 5, 10], [0, 3, 6, 10],
                    [0, 3, 7, 10], [0, 3, 8, 10], [0, 4, 6, 10], [0, 4, 7, 10], [0, 4, 8, 10],
                    [0, 5, 7, 10], [0, 5, 8, 10], [0, 6, 8, 10], [0, 2, 4, 6, 10],
                    [0, 2, 4, 7, 10], [0, 2, 4, 8, 10], [0, 2, 5, 7, 10], [0, 2, 5, 8, 10],
                    [0, 2, 6, 8, 10], [0, 3, 5, 7, 10], [0, 3, 5, 8, 10], [0, 3, 6, 8, 10],
                    [0, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10], [1], [1, 3], [1, 4], [1, 5], [1, 3, 5],
                    [1, 6], [1, 3, 6], [1, 4, 6], [1, 7], [1, 3, 7], [1, 4, 7], [1, 5, 7],
                    [1, 3, 5, 7], [1, 8], [1, 3, 8], [1, 4, 8], [1, 5, 8], [1, 6, 8], [1, 3, 5, 8],
                    [1, 3, 6, 8], [1, 4, 6, 8], [1, 9], [1, 3, 9], [1, 4, 9], [1, 5, 9], [1, 6, 9],
                    [1, 7, 9], [1, 3, 5, 9], [1, 3, 6, 9], [1, 3, 7, 9], [1, 4, 6, 9],
                    [1, 4, 7, 9], [1, 5, 7, 9], [1, 3, 5, 7, 9], [1, 10], [1, 3, 10], [1, 4, 10],
                    [1, 5, 10], [1, 6, 10], [1, 7, 10], [1, 8, 10], [1, 3, 5, 10], [1, 3, 6, 10],
                    [1, 3, 7, 10], [1, 3, 8, 10], [1, 4, 6, 10], [1, 4, 7, 10], [1, 4, 8, 10],
                    [1, 5, 7, 10], [1, 5, 8, 10], [1, 6, 8, 10], [1, 3, 5, 7, 10],
                    [1, 3, 5, 8, 10], [1, 3, 6, 8, 10], [1, 4, 6, 8, 10], [1, 11], [1, 3, 11],
                    [1, 4, 11], [1, 5, 11], [1, 6, 11], [1, 7, 11], [1, 8, 11], [1, 9, 11],
                    [1, 3, 5, 11], [1, 3, 6, 11], [1, 3, 7, 11], [1, 3, 8, 11], [1, 3, 9, 11],
                    [1, 4, 6, 11], [1, 4, 7, 11], [1, 4, 8, 11], [1, 4, 9, 11], [1, 5, 7, 11],
                    [1, 5, 8, 11], [1, 5, 9, 11], [1, 6, 8, 11], [1, 6, 9, 11], [1, 7, 9, 11],
                    [1, 3, 5, 7, 11], [1, 3, 5, 8, 11], [1, 3, 5, 9, 11], [1, 3, 6, 8, 11],
                    [1, 3, 6, 9, 11], [1, 3, 7, 9, 11], [1, 4, 6, 8, 11], [1, 4, 6, 9, 11],
                    [1, 4, 7, 9, 11], [1, 5, 7, 9, 11], [1, 3, 5, 7, 9, 11], [2], [2, 4], [2, 5],
                    [2, 6], [2, 4, 6], [2, 7], [2, 4, 7], [2, 5, 7], [2, 8], [2, 4, 8], [2, 5, 8],
                    [2, 6, 8], [2, 4, 6, 8], [2, 9], [2, 4, 9], [2, 5, 9], [2, 6, 9], [2, 7, 9],
                    [2, 4, 6, 9], [2, 4, 7, 9], [2, 5, 7, 9], [2, 10], [2, 4, 10], [2, 5, 10],
                    [2, 6, 10], [2, 7, 10], [2, 8, 10], [2, 4, 6, 10], [2, 4, 7, 10],
                    [2, 4, 8, 10], [2, 5, 7, 10], [2, 5, 8, 10], [2, 6, 8, 10], [2, 4, 6, 8, 10],
                    [2, 11], [2, 4, 11], [2, 5, 11], [2, 6, 11], [2, 7, 11], [2, 8, 11],
                    [2, 9, 11], [2, 4, 6, 11], [2, 4, 7, 11], [2, 4, 8, 11], [2, 4, 9, 11],
                    [2, 5, 7, 11], [2, 5, 8, 11], [2, 5, 9, 11], [2, 6, 8, 11], [2, 6, 9, 11],
                    [2, 7, 9, 11], [2, 4, 6, 8, 11], [2, 4, 6, 9, 11], [2, 4, 7, 9, 11],
                    [2, 5, 7, 9, 11], [3], [3, 5], [3, 6], [3, 7], [3, 5, 7], [3, 8], [3, 5, 8],
                    [3, 6, 8], [3, 9], [3, 5, 9], [3, 6, 9], [3, 7, 9], [3, 5, 7, 9], [3, 10],
                    [3, 5, 10], [3, 6, 10], [3, 7, 10], [3, 8, 10], [3, 5, 7, 10], [3, 5, 8, 10],
                    [3, 6, 8, 10], [3, 11], [3, 5, 11], [3, 6, 11], [3, 7, 11], [3, 8, 11],
                    [3, 9, 11], [3, 5, 7, 11], [3, 5, 8, 11], [3, 5, 9, 11], [3, 6, 8, 11],
                    [3, 6, 9, 11], [3, 7, 9, 11], [3, 5, 7, 9, 11], [4], [4, 6], [4, 7], [4, 8],
                    [4, 6, 8], [4, 9], [4, 6, 9], [4, 7, 9], [4, 10], [4, 6, 10], [4, 7, 10],
                    [4, 8, 10], [4, 6, 8, 10], [4, 11], [4, 6, 11], [4, 7, 11], [4, 8, 11],
                    [4, 9, 11], [4, 6, 8, 11], [4, 6, 9, 11], [4, 7, 9, 11], [5], [5, 7], [5, 8],
                    [5, 9], [5, 7, 9], [5, 10], [5, 7, 10], [5, 8, 10], [5, 11], [5, 7, 11],
                    [5, 8, 11], [5, 9, 11], [5, 7, 9, 11], [6], [6, 8], [6, 9], [6, 10],
                    [6, 8, 10], [6, 11], [6, 8, 11], [6, 9, 11], [7], [7, 9], [7, 10], [7, 11],
                    [7, 9, 11], [8], [8, 10], [8, 11], [9], [9, 11], [10], [11]]

###################################################################################

MIDI_Instruments_Families = {
                            0: 'Piano Family',
                            1: 'Chromatic Percussion Family',
                            2: 'Organ Family',
                            3: 'Guitar Family',
                            4: 'Bass Family',
                            5: 'Strings Family',
                            6: 'Ensemble Family',
                            7: 'Brass Family',
                            8: 'Reed Family',
                            9: 'Pipe Family',
                            10: 'Synth Lead Family',
                            11: 'Synth Pad Family',
                            12: 'Synth Effects Family',
                            13: 'Ethnic Family',
                            14: 'Percussive Family',
                            15: 'Sound Effects Family',
                            16: 'Drums Family',
                            -1: 'Unknown Family',
                            }

###################################################################################

def patch_to_instrument_family(MIDI_patch, drums_patch=128):

  if 0 <= MIDI_patch < 128:
    return MIDI_patch // 8, MIDI_Instruments_Families[MIDI_patch // 8]

  elif MIDI_patch == drums_patch:
    return MIDI_patch // 8, MIDI_Instruments_Families[16]

  else:
    return -1, MIDI_Instruments_Families[-1]

###################################################################################

def patch_list_from_enhanced_score_notes(enhanced_score_notes, 
                                         default_patch=0, 
                                         drums_patch=9,
                                         verbose=False
                                         ):

  patches = [-1] * 16

  for idx, e in enumerate(enhanced_score_notes):
    if e[0] == 'note':
      if e[3] != 9:
          if patches[e[3]] == -1:
              patches[e[3]] = e[6]
          else:
              if patches[e[3]] != e[6]:
                if e[6] in patches:
                  e[3] = patches.index(e[6])
                else:
                  if -1 in patches:
                      patches[patches.index(-1)] = e[6]
                  else:
                    patches[-1] = e[6]

                    if verbose:
                      print('=' * 70)
                      print('WARNING! Composition has more than 15 patches!')
                      print('Conflict note number:', idx)
                      print('Conflict channel number:', e[3])
                      print('Conflict patch number:', e[6])

  patches = [p if p != -1 else default_patch for p in patches]

  patches[9] = drums_patch

  if verbose:
    print('=' * 70)
    print('Composition patches')
    print('=' * 70)
    for c, p in enumerate(patches):
      print('Cha', str(c).zfill(2), '---', str(p).zfill(3), Number2patch[p])
    print('=' * 70)

  return patches

###################################################################################

def patch_enhanced_score_notes(escore_notes,
                               default_patch=0,
                               reserved_patch=-1,
                               reserved_patch_channel=-1,
                               drums_patch=9,
                               verbose=False
                              ):
  
    #===========================================================================

    enhanced_score_notes = copy.deepcopy(escore_notes)

    #===========================================================================
  
    enhanced_score_notes_with_patch_changes = []

    patches = [-1] * 16

    if -1 < reserved_patch < 128 and -1 < reserved_patch_channel < 128:
        patches[reserved_patch_channel] = reserved_patch

    overflow_idx = -1

    for idx, e in enumerate(enhanced_score_notes):
        if e[0] == 'note':
            if e[3] != 9:
                if -1 < reserved_patch < 128 and -1 < reserved_patch_channel < 128:
                    if e[6] == reserved_patch:
                        e[3] = reserved_patch_channel
                    
                if patches[e[3]] == -1:
                    patches[e[3]] = e[6]
                
                else:
                    if patches[e[3]] != e[6]:
                        if e[6] in patches:
                            e[3] = patches.index(e[6])
                        
                        else:
                            if -1 in patches:
                                patches[patches.index(-1)] = e[6]
                            
                            else:
                                overflow_idx = idx
                                break

        enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    overflow_patches = []
    overflow_channels = [-1] * 16
    overflow_channels[9] = drums_patch

    if -1 < reserved_patch < 128 and -1 < reserved_patch_channel < 128:
        overflow_channels[reserved_patch_channel] = reserved_patch

    if overflow_idx != -1:
        for idx, e in enumerate(enhanced_score_notes[overflow_idx:]):
            if e[0] == 'note':
                if e[3] != 9:
                    if e[6] not in overflow_channels:
                        
                        if -1 in overflow_channels:
                            free_chan = overflow_channels.index(-1)
                            overflow_channels[free_chan] = e[6]
                            e[3] = free_chan

                            enhanced_score_notes_with_patch_changes.append(['patch_change', e[1], e[3], e[6]])
                            
                            overflow_patches.append(e[6])
                
                        else:
                            overflow_channels = [-1] * 16
                            overflow_channels[9] = drums_patch
                            
                            if -1 < reserved_patch < 128 and -1 < reserved_patch_channel < 128:
                                overflow_channels[reserved_patch_channel] = reserved_patch
                                e[3] = reserved_patch_channel

                            if e[6] != reserved_patch:

                                free_chan = overflow_channels.index(-1)
                                e[3] = free_chan
                                    
                            overflow_channels[e[3]] = e[6]                            

                            enhanced_score_notes_with_patch_changes.append(['patch_change', e[1], e[3], e[6]])
                            
                            overflow_patches.append(e[6])

                    else:
                        e[3] = overflow_channels.index(e[6])

            enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    patches = [p if p != -1 else default_patch for p in patches]

    patches[9] = drums_patch

    #===========================================================================

    overflow_patches = ordered_set(overflow_patches)

    #===========================================================================

    if verbose:
      print('=' * 70)
      print('Main composition patches')
      print('=' * 70)
      for c, p in enumerate(patches):
        print('Cha', str(c).zfill(2), '---', str(p).zfill(3), Number2patch[p])
      print('=' * 70)

      if overflow_patches:
        print('Extra composition patches')
        print('=' * 70)
        for c, p in enumerate(overflow_patches):
          print(str(p).zfill(3), Number2patch[p])
        print('=' * 70)

    #===========================================================================

    return enhanced_score_notes_with_patch_changes, patches, overflow_patches

###################################################################################

def create_enhanced_monophonic_melody(monophonic_melody):

    enhanced_monophonic_melody = []

    for i, note in enumerate(monophonic_melody[:-1]):

      enhanced_monophonic_melody.append(note)

      if note[1]+note[2] < monophonic_melody[i+1][1]:
        
        delta_time = monophonic_melody[i+1][1] - (note[1]+note[2])
        enhanced_monophonic_melody.append(['silence', note[1]+note[2], delta_time, note[3], 0, 0, note[6]])
        
    enhanced_monophonic_melody.append(monophonic_melody[-1])

    return enhanced_monophonic_melody

###################################################################################

def frame_monophonic_melody(monophonic_melody, min_frame_time_threshold=10):

    mzip = list(zip(monophonic_melody[:-1], monophonic_melody[1:]))

    times_counts = Counter([(b[1]-a[1]) for a, b in mzip]).most_common()

    mc_time = next((item for item, count in times_counts if item >= min_frame_time_threshold), min_frame_time_threshold)

    times = [(b[1]-a[1]) // mc_time for a, b in mzip] + [monophonic_melody[-1][2] // mc_time]

    framed_melody = []

    for i, note in enumerate(monophonic_melody):
      
      stime = note[1]
      count = times[i]
      
      if count != 0:
        for j in range(count):

          new_note = copy.deepcopy(note)
          new_note[1] = stime + (j * mc_time)
          new_note[2] = mc_time
          framed_melody.append(new_note)
      
      else:
        framed_melody.append(note)

    return [framed_melody, mc_time]

###################################################################################

def delta_score_notes(score_notes, 
                      timings_clip_value=255, 
                      even_timings=False,
                      compress_timings=False
                      ):

  delta_score = []

  pe = score_notes[0]

  for n in score_notes:

    note = copy.deepcopy(n)

    time =  n[1] - pe[1]
    dur = n[2]

    if even_timings:
      if time != 0 and time % 2 != 0:
        time += 1
      if dur % 2 != 0:
        dur += 1

    time = max(0, min(timings_clip_value, time))
    dur = max(0, min(timings_clip_value, dur))

    if compress_timings:
      time /= 2
      dur /= 2

    note[1] = int(time)
    note[2] = int(dur)

    delta_score.append(note)

    pe = n

  return delta_score

###################################################################################

def check_and_fix_chords_in_chordified_score(chordified_score,
                                             channels_index=3,
                                             pitches_index=4
                                             ):
  fixed_chordified_score = []

  bad_chords_counter = 0

  for c in chordified_score:

    tones_chord = sorted(set([t[pitches_index] % 12 for t in c if t[channels_index] != 9]))

    if tones_chord:

        if tones_chord not in ALL_CHORDS_SORTED:
          bad_chords_counter += 1

        while tones_chord not in ALL_CHORDS_SORTED:
          tones_chord.pop(0)

    new_chord = []

    c.sort(key = lambda x: x[pitches_index], reverse=True)

    for e in c:
      if e[channels_index] != 9:
        if e[pitches_index] % 12 in tones_chord:
          new_chord.append(e)

      else:
        new_chord.append(e)

    fixed_chordified_score.append(new_chord)

  return fixed_chordified_score, bad_chords_counter

###################################################################################

from itertools import combinations, groupby

###################################################################################

def advanced_check_and_fix_chords_in_chordified_score(chordified_score,
                                                      channels_index=3,
                                                      pitches_index=4,
                                                      patches_index=6,
                                                      use_filtered_chords=False,
                                                      use_full_chords=False,
                                                      remove_duplicate_pitches=True,
                                                      fix_bad_tones_chords=False,
                                                      fix_bad_pitches=False,
                                                      skip_drums=False
                                                      ):
  fixed_chordified_score = []

  bad_chords_counter = 0
  duplicate_pitches_counter = 0

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  for c in chordified_score:

    chord = copy.deepcopy(c)

    if remove_duplicate_pitches:

      chord.sort(key = lambda x: x[pitches_index], reverse=True)

      seen = set()
      ddchord = []

      for cc in chord:
        if cc[channels_index] != 9:

          if tuple([cc[pitches_index], cc[patches_index]]) not in seen:
            ddchord.append(cc)
            seen.add(tuple([cc[pitches_index], cc[patches_index]]))
          else:
            duplicate_pitches_counter += 1
        
        else:
          ddchord.append(cc)
      
      chord = copy.deepcopy(ddchord)
      
    tones_chord = sorted(set([t[pitches_index] % 12 for t in chord if t[channels_index] != 9]))

    if tones_chord:

        if tones_chord not in CHORDS:
          
          pitches_chord = sorted(set([p[pitches_index] for p in c if p[channels_index] != 9]), reverse=True)
          
          if len(tones_chord) == 2:
            tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

            if tones_counts[0][1] > 1:
              good_tone = tones_counts[0][0]
              bad_tone = tones_counts[1][0]
            
            elif tones_counts[1][1] > 1:
              good_tone = tones_counts[1][0]
              bad_tone = tones_counts[0][0]
            
            else:
              good_tone = pitches_chord[0] % 12
              bad_tone = [t for t in tones_chord if t != good_tone][0]

            tones_chord = [good_tone]

            if fix_bad_tones_chords:

              if good_tone > bad_tone:

                if sorted([good_tone, (12+(bad_tone+1)) % 12]) in CHORDS:
                  tones_chord = sorted([good_tone, (12+(bad_tone-1)) % 12])

                elif sorted([good_tone, (12+(bad_tone-1)) % 12]) in CHORDS:
                  tones_chord = sorted([good_tone, (12+(bad_tone+1)) % 12])

              else:

                if sorted([good_tone, (12+(bad_tone-1)) % 12]) in CHORDS:
                  tones_chord = sorted([good_tone, (12+(bad_tone-1)) % 12])

                elif sorted([good_tone, (12+(bad_tone+1)) % 12]) in CHORDS:
                  tones_chord = sorted([good_tone, (12+(bad_tone+1)) % 12])          

          if len(tones_chord) > 2:
            tones_chord_combs = [list(comb) for i in range(len(tones_chord)-1, 0, -1) for comb in combinations(tones_chord, i)]

            for co in tones_chord_combs:
              if co in CHORDS:
                break

            if fix_bad_tones_chords:

              dt_chord = list(set(co) ^ set(tones_chord))

              for t in dt_chord:
                tones_chord.append((12+(t+1)) % 12)
                tones_chord.append((12+(t-1)) % 12)

              ex_tones_chord = sorted(set(tones_chord))

              tones_chord_combs = [list(comb) for i in range(4, 0, -2) for comb in combinations(ex_tones_chord, i) if all(t in list(comb) for t in co)]
              
              for eco in tones_chord_combs:
                if eco in CHORDS:
                  tones_chord = eco
                  break              

            else:
              tones_chord = co

          if len(tones_chord) == 1:
            tones_chord = [pitches_chord[0] % 12]
            
          bad_chords_counter += 1

    chord.sort(key = lambda x: x[pitches_index], reverse=True)

    new_chord = set()
    pipa = []

    for e in chord:
      if e[channels_index] != 9:
        if e[pitches_index] % 12 in tones_chord:
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

        elif (e[pitches_index]+1) % 12 in tones_chord:
          e[pitches_index] += 1
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

        elif (e[pitches_index]-1) % 12 in tones_chord:
          e[pitches_index] -= 1
          new_chord.add(tuple(e))
          pipa.append([e[pitches_index], e[patches_index]])

    if fix_bad_pitches:

      bad_chord = set()

      for e in chord:
        if e[channels_index] != 9:
          
          if e[pitches_index] % 12 not in tones_chord:
            bad_chord.add(tuple(e))
          
          elif (e[pitches_index]+1) % 12 not in tones_chord:
            bad_chord.add(tuple(e))
          
          elif (e[pitches_index]-1) % 12 not in tones_chord:
            bad_chord.add(tuple(e))
      
      for bc in bad_chord:

        bc = list(bc)

        tone = find_closest_tone(tones_chord, bc[pitches_index] % 12)

        new_pitch =  ((bc[pitches_index] // 12) * 12) + tone

        if [new_pitch, bc[patches_index]] not in pipa:
          bc[pitches_index] = new_pitch
          new_chord.add(tuple(bc))
          pipa.append([[new_pitch], bc[patches_index]])

    if not skip_drums:
      for e in c:
        if e[channels_index] == 9:
          new_chord.add(tuple(e))

    new_chord = [list(e) for e in new_chord]

    new_chord.sort(key = lambda x: (-x[pitches_index], x[patches_index]))

    fixed_chordified_score.append(new_chord)

  return fixed_chordified_score, bad_chords_counter, duplicate_pitches_counter

###################################################################################

def score_chord_to_tones_chord(chord,
                               transpose_value=0,
                               channels_index=3,
                               pitches_index=4):

  return sorted(set([(p[4]+transpose_value) % 12 for p in chord if p[channels_index] != 9]))

###################################################################################

def grouped_set(seq):
  return [k for k, v in groupby(seq)]

###################################################################################

def ordered_set(seq):
  dic = {}
  return [k for k, v in dic.fromkeys(seq).items()]

###################################################################################

def add_melody_to_enhanced_score_notes(enhanced_score_notes,
                                      melody_start_time=0,
                                      melody_start_chord=0,
                                      melody_notes_min_duration=-1,
                                      melody_notes_max_duration=255,
                                      melody_duration_overlap_tolerance=4,
                                      melody_avg_duration_divider=2,
                                      melody_base_octave=5,
                                      melody_channel=3,
                                      melody_patch=40,
                                      melody_max_velocity=110,
                                      acc_max_velocity=90,
                                      pass_drums=True,
                                      return_melody=False
                                      ):
  
    if pass_drums:
      score = copy.deepcopy(enhanced_score_notes)
    else:
      score = [e for e in copy.deepcopy(enhanced_score_notes) if e[3] !=9]

    if melody_notes_min_duration > 0:
      min_duration = melody_notes_min_duration
    else:
      durs = [d[2] for d in score]
      min_duration = Counter(durs).most_common()[0][0]

    adjust_score_velocities(score, acc_max_velocity)

    cscore = chordify_score([1000, score])

    melody_score = []
    acc_score = []

    pt = melody_start_time

    for c in cscore[:melody_start_chord]:
      acc_score.extend(c)

    for c in cscore[melody_start_chord:]:

      durs = [d[2] if d[3] != 9 else -1 for d in c]

      if not all(d == -1 for d in durs):
        ndurs = [d for d in durs if d != -1]
        avg_dur = (sum(ndurs) / len(ndurs)) / melody_avg_duration_divider
        best_dur = min(durs, key=lambda x:abs(x-avg_dur))
        pidx = durs.index(best_dur)

        cc = copy.deepcopy(c[pidx])

        if c[0][1] >= pt - melody_duration_overlap_tolerance and best_dur >= min_duration:

          cc[3] = melody_channel
          cc[4] = (c[pidx][4] % 24)
          cc[5] = 100 + ((c[pidx][4] % 12) * 2)
          cc[6] = melody_patch

          melody_score.append(cc)
          acc_score.extend(c)

          pt = c[0][1]+c[pidx][2]

        else:
          acc_score.extend(c)

      else:
        acc_score.extend(c)

    values = [e[4] % 24 for e in melody_score]
    smoothed = [values[0]]
    for i in range(1, len(values)):
        if abs(smoothed[-1] - values[i]) >= 12:
            if smoothed[-1] < values[i]:
                smoothed.append(values[i] - 12)
            else:
                smoothed.append(values[i] + 12)
        else:
            smoothed.append(values[i])

    smoothed_melody = copy.deepcopy(melody_score)

    for i, e in enumerate(smoothed_melody):
      e[4] = (melody_base_octave * 12) + smoothed[i]

    for i, m in enumerate(smoothed_melody[1:]):
      if m[1] - smoothed_melody[i][1] < melody_notes_max_duration:
        smoothed_melody[i][2] = m[1] - smoothed_melody[i][1]

    adjust_score_velocities(smoothed_melody, melody_max_velocity)

    if return_melody:
      final_score = sorted(smoothed_melody, key=lambda x: (x[1], -x[4]))

    else:
      final_score = sorted(smoothed_melody + acc_score, key=lambda x: (x[1], -x[4]))

    return final_score
    
###################################################################################

def find_paths(list_of_lists, path=[]):
    if not list_of_lists:
        return [path]
    return [p for sublist in list_of_lists[0] for p in find_paths(list_of_lists[1:], path+[sublist])]

###################################################################################

def recalculate_score_timings(score, 
                              start_time=0, 
                              timings_index=1
                              ):

  rscore = copy.deepcopy(score)

  pe = rscore[0]

  abs_time = start_time

  for e in rscore:

    dtime = e[timings_index] - pe[timings_index]
    pe = copy.deepcopy(e)
    abs_time += dtime
    e[timings_index] = abs_time
    
  return rscore

###################################################################################

WHITE_NOTES = [0, 2, 4, 5, 7, 9, 11]
BLACK_NOTES = [1, 3, 6, 8, 10]

###################################################################################

ALL_CHORDS_FILTERED = [[0], [0, 3], [0, 3, 5], [0, 3, 5, 8], [0, 3, 5, 9], [0, 3, 5, 10], [0, 3, 7],
                      [0, 3, 7, 10], [0, 3, 8], [0, 3, 9], [0, 3, 10], [0, 4], [0, 4, 6],
                      [0, 4, 6, 9], [0, 4, 6, 10], [0, 4, 7], [0, 4, 7, 10], [0, 4, 8], [0, 4, 9],
                      [0, 4, 10], [0, 5], [0, 5, 8], [0, 5, 9], [0, 5, 10], [0, 6], [0, 6, 9],
                      [0, 6, 10], [0, 7], [0, 7, 10], [0, 8], [0, 9], [0, 10], [1], [1, 4],
                      [1, 4, 6], [1, 4, 6, 9], [1, 4, 6, 10], [1, 4, 6, 11], [1, 4, 7],
                      [1, 4, 7, 10], [1, 4, 7, 11], [1, 4, 8], [1, 4, 8, 11], [1, 4, 9], [1, 4, 10],
                      [1, 4, 11], [1, 5], [1, 5, 8], [1, 5, 8, 11], [1, 5, 9], [1, 5, 10],
                      [1, 5, 11], [1, 6], [1, 6, 9], [1, 6, 10], [1, 6, 11], [1, 7], [1, 7, 10],
                      [1, 7, 11], [1, 8], [1, 8, 11], [1, 9], [1, 10], [1, 11], [2], [2, 5],
                      [2, 5, 8], [2, 5, 8, 11], [2, 5, 9], [2, 5, 10], [2, 5, 11], [2, 6], [2, 6, 9],
                      [2, 6, 10], [2, 6, 11], [2, 7], [2, 7, 10], [2, 7, 11], [2, 8], [2, 8, 11],
                      [2, 9], [2, 10], [2, 11], [3], [3, 5], [3, 5, 8], [3, 5, 8, 11], [3, 5, 9],
                      [3, 5, 10], [3, 5, 11], [3, 7], [3, 7, 10], [3, 7, 11], [3, 8], [3, 8, 11],
                      [3, 9], [3, 10], [3, 11], [4], [4, 6], [4, 6, 9], [4, 6, 10], [4, 6, 11],
                      [4, 7], [4, 7, 10], [4, 7, 11], [4, 8], [4, 8, 11], [4, 9], [4, 10], [4, 11],
                      [5], [5, 8], [5, 8, 11], [5, 9], [5, 10], [5, 11], [6], [6, 9], [6, 10],
                      [6, 11], [7], [7, 10], [7, 11], [8], [8, 11], [9], [10], [11]]

###################################################################################

def harmonize_enhanced_melody_score_notes(enhanced_melody_score_notes):
  
  mel_tones = [e[4] % 12 for e in enhanced_melody_score_notes]

  cur_chord = []

  song = []

  for i, m in enumerate(mel_tones):
    cur_chord.append(m)
    cc = sorted(set(cur_chord))

    if cc in ALL_CHORDS_FULL:
      song.append(cc)

    else:
      while sorted(set(cur_chord)) not in ALL_CHORDS_FULL:
        cur_chord.pop(0)
      cc = sorted(set(cur_chord))
      song.append(cc)

  return song

###################################################################################

def split_melody(enhanced_melody_score_notes, 
                 split_time=-1, 
                 max_score_time=255
                 ):

  mel_chunks = []

  if split_time == -1:

    durs = [max(0, min(max_score_time, e[2])) for e in enhanced_melody_score_notes]
    stime = max(durs)
    
  else:
    stime = split_time

  pe = enhanced_melody_score_notes[0]
  chu = []
  
  for e in enhanced_melody_score_notes:
    dtime = max(0, min(max_score_time, e[1]-pe[1]))

    if dtime > stime:
      if chu:
        mel_chunks.append(chu)
      chu = []
      chu.append(e)
    else:
      chu.append(e)

    pe = e

  if chu:
    mel_chunks.append(chu)

  return mel_chunks, [[m[0][1], m[-1][1]] for m in mel_chunks], len(mel_chunks)

###################################################################################

def flatten(list_of_lists):
  return [x for y in list_of_lists for x in y]

###################################################################################

def enhanced_delta_score_notes(enhanced_score_notes,
                               start_time=0,
                               max_score_time=255
                               ):

  delta_score = []

  pe = ['note', max(0, enhanced_score_notes[0][1]-start_time)]

  for e in enhanced_score_notes:

    dtime = max(0, min(max_score_time, e[1]-pe[1]))
    dur = max(1, min(max_score_time, e[2]))
    cha = max(0, min(15, e[3]))
    ptc = max(1, min(127, e[4]))
    vel = max(1, min(127, e[5]))
    pat = max(0, min(128, e[6]))

    delta_score.append([dtime, dur, cha, ptc, vel, pat])

    pe = e

  return delta_score

###################################################################################

def basic_enhanced_delta_score_notes_tokenizer(enhanced_delta_score_notes,
                                              tokenize_start_times=True,
                                              tokenize_durations=True,
                                              tokenize_channels=True,
                                              tokenize_pitches=True,
                                              tokenize_velocities=True,
                                              tokenize_patches=True,
                                              score_timings_range=256,
                                              max_seq_len=-1,
                                              seq_pad_value=-1
                                              ):
  
  
  
  score_tokens_ints_seq = []

  tokens_shifts = [-1] * 7

  for d in enhanced_delta_score_notes:

    seq = []
    shift = 0

    if tokenize_start_times:
      seq.append(d[0])
      tokens_shifts[0] = shift
      shift += score_timings_range

    if tokenize_durations:
      seq.append(d[1]+shift)
      tokens_shifts[1] = shift
      shift += score_timings_range

    if tokenize_channels:
      tokens_shifts[2] = shift
      seq.append(d[2]+shift)
      shift += 16
    
    if tokenize_pitches:
      tokens_shifts[3] = shift
      seq.append(d[3]+shift)
      shift += 128
    
    if tokenize_velocities:
      tokens_shifts[4] = shift
      seq.append(d[4]+shift)
      shift += 128

    if tokenize_patches:
      tokens_shifts[5] = shift
      seq.append(d[5]+shift)
      shift += 129

    tokens_shifts[6] = shift
    score_tokens_ints_seq.append(seq)

  final_score_tokens_ints_seq = flatten(score_tokens_ints_seq)

  if max_seq_len > -1:
    final_score_tokens_ints_seq = final_score_tokens_ints_seq[:max_seq_len]

  if seq_pad_value > -1:
    final_score_tokens_ints_seq += [seq_pad_value] * (max_seq_len - len(final_score_tokens_ints_seq))

  return [score_tokens_ints_seq,
          final_score_tokens_ints_seq, 
          tokens_shifts,
          seq_pad_value, 
          max_seq_len,
          len(score_tokens_ints_seq),
          len(final_score_tokens_ints_seq)
          ]

###################################################################################

def basic_enhanced_delta_score_notes_detokenizer(tokenized_seq, 
                                                 tokens_shifts, 
                                                 timings_multiplier=16
                                                 ):

  song_f = []

  time = 0
  dur = 16
  channel = 0
  pitch = 60
  vel = 90
  pat = 0

  note_seq_len = len([t for t in tokens_shifts if t > -1])-1
  tok_shifts_idxs = [i for i in range(len(tokens_shifts[:-1])) if tokens_shifts[i] > - 1]

  song = []

  for i in range(0, len(tokenized_seq), note_seq_len):
    note = tokenized_seq[i:i+note_seq_len]
    song.append(note)

  for note in song:
    for i, idx in enumerate(tok_shifts_idxs):
      if idx == 0:
        time += (note[i]-tokens_shifts[0]) * timings_multiplier
      elif idx == 1:
        dur = (note[i]-tokens_shifts[1]) * timings_multiplier
      elif idx == 2:
        channel = (note[i]-tokens_shifts[2])
      elif idx == 3:
        pitch = (note[i]-tokens_shifts[3])
      elif idx == 4:
        vel = (note[i]-tokens_shifts[4])
      elif idx == 5:
        pat = (note[i]-tokens_shifts[5])

    song_f.append(['note', time, dur, channel, pitch, vel, pat ])

  return song_f

###################################################################################

def enhanced_chord_to_chord_token(enhanced_chord, 
                                  channels_index=3, 
                                  pitches_index=4, 
                                  use_filtered_chords=False,
                                  use_full_chords=True
                                  ):
  
  bad_chords_counter = 0
  duplicate_pitches_counter = 0

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(set([t[pitches_index] % 12 for t in enhanced_chord if t[channels_index] != 9]))

  original_tones_chord = copy.deepcopy(tones_chord)

  if tones_chord:

      if tones_chord not in CHORDS:
        
        pitches_chord = sorted(set([p[pitches_index] for p in enhanced_chord if p[channels_index] != 9]), reverse=True)
        
        if len(tones_chord) == 2:
          tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

          if tones_counts[0][1] > 1:
            tones_chord = [tones_counts[0][0]]
          elif tones_counts[1][1] > 1:
            tones_chord = [tones_counts[1][0]]
          else:
            tones_chord = [pitches_chord[0] % 12]

        else:
          tones_chord_combs = [list(comb) for i in range(len(tones_chord)-2, 0, -1) for comb in combinations(tones_chord, i+1)]

          for co in tones_chord_combs:
            if co in CHORDS:
              tones_chord = co
              break

  if use_filtered_chords:
    chord_token = ALL_CHORDS_FILTERED.index(tones_chord)
  else:
    chord_token = ALL_CHORDS_SORTED.index(tones_chord)

  return [chord_token, tones_chord, original_tones_chord, sorted(set(original_tones_chord) ^ set(tones_chord))]

###################################################################################

def enhanced_chord_to_tones_chord(enhanced_chord):
  return sorted(set([t[4] % 12 for t in enhanced_chord if t[3] != 9]))

###################################################################################

import hashlib

###################################################################################

def md5_hash(file_path_or_data=None, original_md5_hash=None):

  if type(file_path_or_data) == str:

    with open(file_path_or_data, 'rb') as file_to_check:
      data = file_to_check.read()
      
      if data:
        md5 = hashlib.md5(data).hexdigest()

  else:
    if file_path_or_data:
      md5 = hashlib.md5(file_path_or_data).hexdigest()

  if md5:

    if original_md5_hash:

      if md5 == original_md5_hash:
        check = True
      else:
        check = False
        
    else:
      check = None

    return [md5, check]

  else:

    md5 = None
    check = None

    return [md5, check]

###################################################################################

ALL_PITCHES_CHORDS_FILTERED = [[67], [64], [62], [69], [60], [65], [59], [70], [66], [63], [68], [61],
                              [64, 60], [67, 64], [65, 62], [62, 59], [69, 65], [60, 57], [66, 62], [59, 55],
                              [62, 57], [67, 62], [64, 59], [64, 60, 55], [60, 55], [65, 60], [64, 61],
                              [69, 64], [66, 62, 57], [69, 66], [62, 59, 55], [64, 60, 57], [62, 58],
                              [65, 60, 57], [70, 67], [67, 63], [64, 61, 57], [61, 57], [63, 60], [68, 64],
                              [65, 62, 58], [65, 62, 57], [59, 56], [63, 58], [68, 65], [59, 54, 47, 35],
                              [70, 65], [66, 61], [64, 59, 56], [65, 61], [64, 59, 55], [63, 59], [61, 58],
                              [68, 63], [60, 56], [67, 63, 60], [67, 63, 58], [66, 62, 59], [61, 56],
                              [70, 66], [67, 62, 58], [63, 60, 56], [65, 61, 56], [66, 61, 58], [66, 61, 57],
                              [65, 60, 56], [65, 61, 58], [65, 59], [68, 64, 61], [66, 60], [64, 58],
                              [62, 56], [63, 57], [61, 55], [66, 64], [60, 58], [65, 63], [63, 59, 56],
                              [65, 62, 59], [61, 59], [66, 60, 57], [64, 61, 55], [64, 58, 55], [62, 59, 56],
                              [64, 60, 58], [63, 60, 57], [64, 60, 58, 55], [65, 62, 56], [64, 61, 58],
                              [66, 64, 59], [60, 58, 55], [65, 63, 60], [63, 57, 53], [65, 63, 60, 57],
                              [65, 59, 56], [63, 60, 58, 55], [67, 61, 58], [64, 61, 57, 54], [64, 61, 59],
                              [70, 65, 60], [68, 65, 63, 60], [63, 60, 58], [65, 63, 58], [69, 66, 64],
                              [64, 60, 54], [64, 60, 57, 54], [66, 64, 61], [66, 61, 59], [67, 63, 59],
                              [65, 61, 57], [68, 65, 63], [64, 61, 59, 56], [65, 61, 59], [66, 64, 61, 58],
                              [64, 61, 58, 55], [64, 60, 56], [65, 61, 59, 56], [66, 62, 58], [61, 59, 56],
                              [64, 58, 54], [63, 59, 53], [65, 62, 59, 56], [61, 59, 55], [64, 61, 59, 55],
                              [68, 65, 63, 59], [70, 66, 60], [65, 63, 60, 58], [64, 61, 59, 54],
                              [70, 64, 60, 54]]

###################################################################################

ALL_PITCHES_CHORDS_SORTED = [[60], [62, 60], [63, 60], [64, 60], [64, 62, 60], [65, 60], [65, 62, 60],
                            [65, 63, 60], [66, 60], [66, 62, 60], [66, 63, 60], [64, 60, 54],
                            [64, 60, 54, 50], [60, 55], [67, 62, 60], [67, 63, 60], [64, 60, 55],
                            [65, 60, 55], [64, 62, 60, 55], [67, 65, 62, 60], [67, 65, 63, 60], [60, 56],
                            [62, 60, 56], [63, 60, 56], [64, 60, 56], [65, 60, 56], [66, 60, 56],
                            [72, 68, 64, 62], [65, 62, 60, 56], [66, 62, 60, 56], [68, 65, 63, 60],
                            [68, 66, 63, 60], [60, 44, 42, 40], [88, 80, 74, 66, 60, 56], [60, 57],
                            [62, 60, 57], [63, 60, 57], [64, 60, 57], [65, 60, 57], [66, 60, 57],
                            [67, 60, 57], [64, 62, 60, 57], [65, 62, 60, 57], [69, 66, 62, 60],
                            [67, 62, 60, 57], [65, 63, 60, 57], [66, 63, 60, 57], [67, 63, 60, 57],
                            [64, 60, 57, 54], [67, 64, 60, 57], [67, 65, 60, 57], [69, 64, 60, 54, 38],
                            [67, 64, 62, 60, 57], [67, 65, 62, 60, 57], [67, 65, 63, 60, 57], [60, 58],
                            [62, 60, 58], [63, 60, 58], [64, 60, 58], [70, 65, 60], [70, 66, 60],
                            [60, 58, 55], [70, 60, 56], [74, 64, 60, 58], [65, 62, 60, 58],
                            [70, 66, 62, 60], [62, 60, 58, 55], [72, 68, 62, 58], [65, 63, 60, 58],
                            [70, 66, 63, 60], [63, 60, 58, 55], [70, 63, 60, 56], [70, 64, 60, 54],
                            [64, 60, 58, 55], [68, 64, 60, 58], [65, 60, 58, 55], [70, 65, 60, 56],
                            [70, 66, 60, 56], [78, 76, 74, 72, 70, 66], [67, 64, 62, 58, 36],
                            [74, 68, 64, 58, 48], [65, 62, 58, 55, 36], [65, 62, 60, 56, 46],
                            [72, 66, 62, 56, 46], [79, 65, 63, 58, 53, 36], [65, 60, 56, 51, 46, 41],
                            [70, 66, 63, 60, 44], [68, 66, 64, 58, 56, 48],
                            [94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58,
                              56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24],
                            [61], [63, 61], [64, 61], [65, 61], [65, 63, 61], [66, 61], [66, 63, 61],
                            [66, 64, 61], [61, 55], [67, 63, 61], [64, 61, 55], [65, 61, 55],
                            [65, 61, 55, 39], [61, 56], [63, 61, 56], [68, 64, 61], [65, 61, 56],
                            [66, 61, 56], [68, 65, 63, 61], [54, 49, 44, 39], [68, 64, 61, 42], [61, 57],
                            [63, 61, 57], [64, 61, 57], [65, 61, 57], [66, 61, 57], [67, 61, 57],
                            [69, 65, 63, 61], [66, 63, 61, 57], [67, 63, 61, 57], [64, 61, 57, 54],
                            [67, 64, 61, 57], [65, 61, 55, 45], [67, 65, 63, 61, 57], [61, 58],
                            [63, 61, 58], [64, 61, 58], [65, 61, 58], [66, 61, 58], [67, 61, 58],
                            [61, 58, 56], [65, 63, 61, 58], [66, 63, 61, 58], [67, 63, 61, 58],
                            [63, 61, 58, 56], [66, 64, 61, 58], [64, 61, 58, 55], [68, 64, 61, 58],
                            [65, 61, 58, 55], [65, 61, 58, 56], [58, 54, 49, 44], [70, 65, 61, 55, 39],
                            [80, 68, 65, 63, 61, 58], [63, 58, 54, 49, 44, 39], [73, 68, 64, 58, 54],
                            [61, 59], [63, 61, 59], [64, 61, 59], [65, 61, 59], [66, 61, 59], [61, 59, 55],
                            [61, 59, 56], [61, 59, 57], [63, 59, 53, 49], [66, 63, 61, 59],
                            [71, 67, 63, 61], [63, 61, 59, 56], [61, 57, 51, 47], [64, 61, 59, 54],
                            [64, 61, 59, 55], [64, 61, 59, 56], [64, 61, 59, 57], [65, 61, 59, 55],
                            [65, 61, 59, 56], [69, 65, 61, 59], [66, 61, 59, 56], [71, 66, 61, 57],
                            [71, 67, 61, 57], [67, 63, 59, 53, 49], [68, 65, 63, 59, 37],
                            [65, 63, 61, 59, 57], [66, 63, 61, 59, 56], [73, 69, 66, 63, 59],
                            [79, 75, 73, 61, 59, 33], [61, 56, 52, 47, 42, 35], [76, 73, 69, 66, 35],
                            [71, 67, 64, 61, 57], [73, 71, 69, 67, 65],
                            [95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59,
                              57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25],
                            [62], [64, 62], [65, 62], [66, 62], [66, 64, 62], [67, 62], [67, 64, 62],
                            [67, 65, 62], [62, 56], [68, 64, 62], [65, 62, 56], [66, 62, 56],
                            [66, 62, 56, 52], [62, 57], [50, 45, 40], [65, 62, 57], [66, 62, 57],
                            [55, 50, 45], [66, 64, 62, 57], [55, 50, 45, 40], [69, 67, 65, 62], [62, 58],
                            [64, 62, 58], [65, 62, 58], [66, 62, 58], [67, 62, 58], [62, 58, 56],
                            [66, 64, 62, 58], [67, 64, 62, 58], [64, 62, 58, 56], [65, 62, 58, 55],
                            [65, 62, 58, 56], [66, 62, 58, 56], [66, 64, 58, 44, 38], [62, 59],
                            [64, 62, 59], [65, 62, 59], [66, 62, 59], [62, 59, 55], [62, 59, 56],
                            [62, 59, 57], [66, 64, 62, 59], [67, 64, 62, 59], [64, 62, 59, 56],
                            [64, 62, 59, 57], [67, 65, 62, 59], [65, 62, 59, 56], [69, 65, 62, 59],
                            [66, 62, 59, 56], [69, 66, 62, 59], [59, 55, 50, 45], [64, 62, 59, 56, 54],
                            [69, 66, 62, 59, 40], [64, 59, 55, 50, 45, 40], [69, 65, 62, 59, 55], [63],
                            [65, 63], [66, 63], [67, 63], [67, 65, 63], [68, 63], [68, 65, 63],
                            [68, 66, 63], [63, 57], [63, 57, 53], [66, 63, 57], [67, 63, 57],
                            [67, 63, 57, 53], [63, 58], [65, 63, 58], [66, 63, 58], [67, 63, 58],
                            [68, 63, 58], [67, 65, 63, 58], [63, 58, 56, 53], [70, 68, 66, 63], [63, 59],
                            [63, 59, 53], [66, 63, 59], [67, 63, 59], [63, 59, 56], [63, 59, 57],
                            [63, 59, 55, 53], [68, 65, 63, 59], [69, 65, 63, 59], [66, 63, 59, 56],
                            [66, 63, 59, 57], [67, 63, 59, 57], [67, 63, 59, 57, 41], [64], [66, 64],
                            [67, 64], [68, 64], [68, 66, 64], [69, 64], [69, 66, 64], [69, 67, 64],
                            [64, 58], [64, 58, 54], [64, 58, 55], [68, 64, 58], [68, 64, 58, 42], [64, 59],
                            [66, 64, 59], [64, 59, 55], [64, 59, 56], [64, 59, 57], [64, 59, 56, 54],
                            [64, 59, 57, 54], [69, 64, 59, 55], [65], [67, 65], [68, 65], [69, 65],
                            [69, 67, 65], [70, 65], [65, 58, 55], [70, 68, 65], [65, 59], [65, 59, 55],
                            [65, 59, 56], [59, 57, 53], [69, 65, 59, 55], [66], [68, 66], [69, 66],
                            [70, 66], [80, 70, 54], [59, 54, 47, 35], [66, 59, 56], [71, 69, 66], [67],
                            [69, 67], [70, 67], [59, 55], [71, 69, 67], [68], [70, 68], [59, 56], [69],
                            [71, 69], [70], [59]]

###################################################################################

def sort_list_by_other(list1, list2):
    return sorted(list1, key=lambda x: list2.index(x) if x in list2 else len(list2))

###################################################################################

ALL_CHORDS_PAIRS_SORTED = [[[0], [0, 4, 7]], [[0, 2], [0, 4, 7]], [[0, 3], [0, 3, 7]],
                          [[0, 4], [0, 4, 7]], [[0, 2, 4], [0, 2, 4, 7]], [[0, 5], [0, 5, 9]],
                          [[0, 2, 5], [0, 2, 5, 9]], [[0, 3, 5], [0, 3, 5, 9]], [[0, 6], [0, 2, 6, 9]],
                          [[0, 2, 6], [0, 2, 6, 9]], [[0, 3, 6], [0, 3, 6, 8]],
                          [[0, 4, 6], [0, 4, 6, 9]], [[0, 2, 4, 6], [0, 2, 4, 6, 9]],
                          [[0, 7], [0, 4, 7]], [[0, 2, 7], [0, 2, 4, 7]], [[0, 3, 7], [0, 3, 7, 10]],
                          [[0, 4, 7], [0, 4, 7, 9]], [[0, 5, 7], [0, 5, 7, 9]],
                          [[0, 2, 4, 7], [0, 2, 4, 7, 9]], [[0, 2, 5, 7], [0, 2, 5, 7, 9]],
                          [[0, 3, 5, 7], [0, 3, 5, 7, 10]], [[0, 8], [0, 3, 8]],
                          [[0, 2, 8], [0, 2, 5, 8]], [[0, 3, 8], [0, 3, 5, 8]],
                          [[0, 4, 8], [2, 4, 8, 11]], [[0, 5, 8], [0, 3, 5, 8]],
                          [[0, 6, 8], [0, 3, 6, 8]], [[0, 2, 4, 8], [0, 2, 4, 6, 8]],
                          [[0, 2, 5, 8], [0, 2, 5, 8, 10]], [[0, 2, 6, 8], [0, 2, 6, 8, 10]],
                          [[0, 3, 5, 8], [0, 3, 5, 8, 10]], [[0, 3, 6, 8], [0, 3, 6, 8, 10]],
                          [[0, 4, 6, 8], [2, 4, 6, 8, 11]], [[0, 2, 4, 6, 8], [2, 4, 6, 8, 11]],
                          [[0, 9], [0, 4, 9]], [[0, 2, 9], [0, 2, 6, 9]], [[0, 3, 9], [0, 3, 5, 9]],
                          [[0, 4, 9], [0, 4, 7, 9]], [[0, 5, 9], [0, 2, 5, 9]],
                          [[0, 6, 9], [0, 2, 6, 9]], [[0, 7, 9], [0, 4, 7, 9]],
                          [[0, 2, 4, 9], [0, 2, 4, 7, 9]], [[0, 2, 5, 9], [0, 2, 5, 7, 9]],
                          [[0, 2, 6, 9], [0, 2, 4, 6, 9]], [[0, 2, 7, 9], [0, 2, 4, 7, 9]],
                          [[0, 3, 5, 9], [0, 3, 5, 7, 9]], [[0, 3, 6, 9], [0, 2, 4, 6, 9]],
                          [[0, 3, 7, 9], [0, 3, 5, 7, 9]], [[0, 4, 6, 9], [0, 2, 4, 6, 9]],
                          [[0, 4, 7, 9], [0, 2, 4, 7, 9]], [[0, 5, 7, 9], [0, 2, 5, 7, 9]],
                          [[0, 2, 4, 6, 9], [2, 4, 6, 9, 11]], [[0, 2, 4, 7, 9], [2, 4, 7, 9, 11]],
                          [[0, 2, 5, 7, 9], [2, 5, 7, 9, 11]], [[0, 3, 5, 7, 9], [2, 4, 6, 8, 11]],
                          [[0, 10], [2, 5, 10]], [[0, 2, 10], [0, 2, 5, 10]],
                          [[0, 3, 10], [0, 3, 7, 10]], [[0, 4, 10], [0, 4, 7, 10]],
                          [[0, 5, 10], [0, 2, 5, 10]], [[0, 6, 10], [0, 3, 6, 10]],
                          [[0, 7, 10], [0, 4, 7, 10]], [[0, 8, 10], [0, 3, 8, 10]],
                          [[0, 2, 4, 10], [0, 2, 4, 7, 10]], [[0, 2, 5, 10], [0, 2, 5, 7, 10]],
                          [[0, 2, 6, 10], [0, 2, 6, 8, 10]], [[0, 2, 7, 10], [0, 2, 5, 7, 10]],
                          [[0, 2, 8, 10], [0, 2, 5, 8, 10]], [[0, 3, 5, 10], [0, 3, 5, 7, 10]],
                          [[0, 3, 6, 10], [0, 3, 6, 8, 10]], [[0, 3, 7, 10], [0, 3, 5, 7, 10]],
                          [[0, 3, 8, 10], [0, 3, 5, 8, 10]], [[0, 4, 6, 10], [0, 2, 4, 6, 10]],
                          [[0, 4, 7, 10], [0, 2, 4, 7, 10]], [[0, 4, 8, 10], [0, 2, 4, 8, 10]],
                          [[0, 5, 7, 10], [0, 3, 5, 7, 10]], [[0, 5, 8, 10], [0, 3, 5, 8, 10]],
                          [[0, 6, 8, 10], [0, 3, 6, 8, 10]], [[0, 2, 4, 6, 10], [0, 2, 4, 8, 10]],
                          [[0, 2, 4, 7, 10], [1, 3, 6, 9, 11]], [[0, 2, 4, 8, 10], [1, 3, 7, 9, 11]],
                          [[0, 2, 5, 7, 10], [0, 3, 5, 7, 10]], [[0, 2, 5, 8, 10], [1, 4, 7, 9, 11]],
                          [[0, 2, 6, 8, 10], [2, 4, 6, 8, 10]], [[0, 3, 5, 7, 10], [0, 2, 5, 7, 10]],
                          [[0, 3, 5, 8, 10], [1, 3, 5, 8, 10]], [[0, 3, 6, 8, 10], [1, 3, 6, 8, 10]],
                          [[0, 4, 6, 8, 10], [0, 2, 4, 6, 9]],
                          [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], [[1], [1, 8]], [[1, 3], [1, 5, 8]],
                          [[1, 4], [1, 4, 9]], [[1, 5], [1, 5, 8]], [[1, 3, 5], [1, 3, 5, 10]],
                          [[1, 6], [1, 6, 10]], [[1, 3, 6], [1, 3, 6, 10]], [[1, 4, 6], [1, 4, 6, 9]],
                          [[1, 7], [1, 4, 7]], [[1, 3, 7], [1, 3, 7, 10]], [[1, 4, 7], [1, 4, 7, 9]],
                          [[1, 5, 7], [1, 5, 7, 10]], [[1, 3, 5, 7], [1, 3, 5, 7, 10]],
                          [[1, 8], [1, 5, 8]], [[1, 3, 8], [1, 3, 5, 8]], [[1, 4, 8], [1, 4, 8, 11]],
                          [[1, 5, 8], [1, 5, 8, 10]], [[1, 6, 8], [1, 3, 6, 8]],
                          [[1, 3, 5, 8], [1, 3, 5, 8, 10]], [[1, 3, 6, 8], [1, 3, 6, 8, 10]],
                          [[1, 4, 6, 8], [1, 4, 6, 8, 11]], [[1, 9], [1, 4, 9]],
                          [[1, 3, 9], [1, 3, 6, 9]], [[1, 4, 9], [1, 4, 6, 9]],
                          [[1, 5, 9], [0, 3, 5, 9]], [[1, 6, 9], [1, 4, 6, 9]],
                          [[1, 7, 9], [1, 4, 7, 9]], [[1, 3, 5, 9], [0, 3, 5, 7, 9]],
                          [[1, 3, 6, 9], [1, 3, 6, 9, 11]], [[1, 3, 7, 9], [1, 3, 5, 7, 9]],
                          [[1, 4, 6, 9], [1, 4, 6, 9, 11]], [[1, 4, 7, 9], [1, 4, 7, 9, 11]],
                          [[1, 5, 7, 9], [1, 3, 7, 9, 11]], [[1, 3, 5, 7, 9], [2, 4, 6, 8, 11]],
                          [[1, 10], [1, 5, 10]], [[1, 3, 10], [1, 3, 7, 10]],
                          [[1, 4, 10], [1, 4, 6, 10]], [[1, 5, 10], [1, 5, 8, 10]],
                          [[1, 6, 10], [1, 4, 6, 10]], [[1, 7, 10], [1, 3, 7, 10]],
                          [[1, 8, 10], [1, 5, 8, 10]], [[1, 3, 5, 10], [1, 3, 5, 8, 10]],
                          [[1, 3, 6, 10], [1, 3, 6, 8, 10]], [[1, 3, 7, 10], [1, 3, 5, 7, 10]],
                          [[1, 3, 8, 10], [1, 3, 5, 8, 10]], [[1, 4, 6, 10], [1, 4, 6, 8, 10]],
                          [[1, 4, 7, 10], [0, 2, 4, 7, 10]], [[1, 4, 8, 10], [1, 4, 6, 8, 10]],
                          [[1, 5, 7, 10], [1, 3, 5, 7, 10]], [[1, 5, 8, 10], [1, 3, 5, 8, 10]],
                          [[1, 6, 8, 10], [1, 3, 6, 8, 10]], [[1, 3, 5, 7, 10], [2, 4, 6, 8, 11]],
                          [[1, 3, 5, 8, 10], [0, 3, 5, 8, 10]], [[1, 3, 6, 8, 10], [0, 3, 6, 8, 10]],
                          [[1, 4, 6, 8, 10], [0, 3, 5, 7, 9]], [[1, 11], [2, 6, 11]],
                          [[1, 3, 11], [1, 3, 6, 11]], [[1, 4, 11], [1, 4, 8, 11]],
                          [[1, 5, 11], [1, 5, 8, 11]], [[1, 6, 11], [1, 4, 6, 11]],
                          [[1, 7, 11], [1, 4, 7, 11]], [[1, 8, 11], [1, 4, 8, 11]],
                          [[1, 9, 11], [1, 4, 9, 11]], [[1, 3, 5, 11], [1, 3, 5, 8, 11]],
                          [[1, 3, 6, 11], [1, 3, 6, 8, 11]], [[1, 3, 7, 11], [1, 3, 7, 9, 11]],
                          [[1, 3, 8, 11], [1, 3, 6, 8, 11]], [[1, 3, 9, 11], [1, 3, 6, 9, 11]],
                          [[1, 4, 6, 11], [1, 4, 6, 9, 11]], [[1, 4, 7, 11], [1, 4, 7, 9, 11]],
                          [[1, 4, 8, 11], [1, 4, 6, 8, 11]], [[1, 4, 9, 11], [1, 4, 6, 9, 11]],
                          [[1, 5, 7, 11], [0, 4, 6, 8, 10]], [[1, 5, 8, 11], [1, 3, 5, 8, 11]],
                          [[1, 5, 9, 11], [1, 5, 7, 9, 11]], [[1, 6, 8, 11], [1, 3, 6, 8, 11]],
                          [[1, 6, 9, 11], [1, 4, 6, 9, 11]], [[1, 7, 9, 11], [1, 4, 7, 9, 11]],
                          [[1, 3, 5, 7, 11], [0, 2, 4, 6, 8]], [[1, 3, 5, 8, 11], [0, 2, 4, 7, 10]],
                          [[1, 3, 5, 9, 11], [1, 3, 7, 9, 11]], [[1, 3, 6, 8, 11], [1, 4, 6, 8, 11]],
                          [[1, 3, 6, 9, 11], [0, 2, 5, 8, 10]], [[1, 3, 7, 9, 11], [1, 3, 6, 9, 11]],
                          [[1, 4, 6, 8, 11], [1, 4, 6, 9, 11]], [[1, 4, 6, 9, 11], [2, 4, 6, 9, 11]],
                          [[1, 4, 7, 9, 11], [2, 4, 7, 9, 11]], [[1, 5, 7, 9, 11], [2, 4, 7, 9, 11]],
                          [[1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10]], [[2], [2, 9]], [[2, 4], [2, 6, 9]],
                          [[2, 5], [2, 5, 9]], [[2, 6], [2, 6, 9]], [[2, 4, 6], [2, 4, 6, 9]],
                          [[2, 7], [2, 7, 11]], [[2, 4, 7], [2, 4, 7, 11]], [[2, 5, 7], [2, 5, 7, 11]],
                          [[2, 8], [4, 8, 11]], [[2, 4, 8], [2, 4, 8, 11]], [[2, 5, 8], [2, 5, 8, 10]],
                          [[2, 6, 8], [2, 6, 8, 11]], [[2, 4, 6, 8], [2, 4, 6, 8, 11]],
                          [[2, 9], [2, 6, 9]], [[2, 4, 9], [2, 4, 6, 9]], [[2, 5, 9], [0, 2, 5, 9]],
                          [[2, 6, 9], [2, 6, 9, 11]], [[2, 7, 9], [2, 7, 9, 11]],
                          [[2, 4, 6, 9], [2, 4, 6, 9, 11]], [[2, 4, 7, 9], [2, 4, 7, 9, 11]],
                          [[2, 5, 7, 9], [0, 2, 5, 7, 9]], [[2, 10], [2, 5, 10]],
                          [[2, 4, 10], [2, 4, 7, 10]], [[2, 5, 10], [2, 5, 7, 10]],
                          [[2, 6, 10], [1, 4, 6, 10]], [[2, 7, 10], [2, 5, 7, 10]],
                          [[2, 8, 10], [2, 5, 8, 10]], [[2, 4, 6, 10], [0, 2, 4, 6, 10]],
                          [[2, 4, 7, 10], [0, 2, 4, 7, 10]], [[2, 4, 8, 10], [2, 4, 7, 9, 11]],
                          [[2, 5, 7, 10], [0, 2, 5, 7, 10]], [[2, 5, 8, 10], [0, 2, 5, 8, 10]],
                          [[2, 6, 8, 10], [1, 3, 5, 7, 10]], [[2, 4, 6, 8, 10], [0, 2, 6, 8, 10]],
                          [[2, 11], [2, 7, 11]], [[2, 4, 11], [2, 4, 8, 11]],
                          [[2, 5, 11], [2, 5, 7, 11]], [[2, 6, 11], [2, 6, 9, 11]],
                          [[2, 7, 11], [2, 4, 7, 11]], [[2, 8, 11], [2, 4, 8, 11]],
                          [[2, 9, 11], [2, 6, 9, 11]], [[2, 4, 6, 11], [2, 4, 6, 9, 11]],
                          [[2, 4, 7, 11], [2, 4, 7, 9, 11]], [[2, 4, 8, 11], [2, 4, 6, 8, 11]],
                          [[2, 4, 9, 11], [2, 4, 7, 9, 11]], [[2, 5, 7, 11], [2, 5, 7, 9, 11]],
                          [[2, 5, 8, 11], [1, 3, 5, 8, 11]], [[2, 5, 9, 11], [2, 5, 7, 9, 11]],
                          [[2, 6, 8, 11], [2, 4, 6, 8, 11]], [[2, 6, 9, 11], [2, 4, 6, 9, 11]],
                          [[2, 7, 9, 11], [2, 4, 7, 9, 11]], [[2, 4, 6, 8, 11], [2, 4, 6, 9, 11]],
                          [[2, 4, 6, 9, 11], [2, 4, 7, 9, 11]], [[2, 4, 7, 9, 11], [0, 2, 4, 7, 9]],
                          [[2, 5, 7, 9, 11], [2, 4, 7, 9, 11]], [[3], [3, 10]], [[3, 5], [3, 7, 10]],
                          [[3, 6], [3, 6, 11]], [[3, 7], [3, 7, 10]], [[3, 5, 7], [3, 5, 7, 10]],
                          [[3, 8], [0, 3, 8]], [[3, 5, 8], [0, 3, 5, 8]], [[3, 6, 8], [0, 3, 6, 8]],
                          [[3, 9], [0, 3, 9]], [[3, 5, 9], [0, 3, 5, 9]], [[3, 6, 9], [3, 6, 9, 11]],
                          [[3, 7, 9], [0, 3, 7, 9]], [[3, 5, 7, 9], [0, 3, 5, 7, 9]],
                          [[3, 10], [3, 7, 10]], [[3, 5, 10], [3, 5, 7, 10]],
                          [[3, 6, 10], [1, 3, 6, 10]], [[3, 7, 10], [0, 3, 7, 10]],
                          [[3, 8, 10], [0, 3, 8, 10]], [[3, 5, 7, 10], [0, 3, 5, 7, 10]],
                          [[3, 5, 8, 10], [0, 3, 5, 8, 10]], [[3, 6, 8, 10], [1, 3, 6, 8, 10]],
                          [[3, 11], [3, 6, 11]], [[3, 5, 11], [3, 5, 8, 11]],
                          [[3, 6, 11], [3, 6, 9, 11]], [[3, 7, 11], [2, 5, 7, 11]],
                          [[3, 8, 11], [3, 6, 8, 11]], [[3, 9, 11], [3, 6, 9, 11]],
                          [[3, 5, 7, 11], [3, 5, 7, 9, 11]], [[3, 5, 8, 11], [1, 3, 5, 8, 11]],
                          [[3, 5, 9, 11], [3, 5, 7, 9, 11]], [[3, 6, 8, 11], [1, 3, 6, 8, 11]],
                          [[3, 6, 9, 11], [1, 3, 6, 9, 11]], [[3, 7, 9, 11], [2, 4, 7, 9, 11]],
                          [[3, 5, 7, 9, 11], [2, 5, 7, 9, 11]], [[4], [4, 11]], [[4, 6], [4, 7, 11]],
                          [[4, 7], [0, 4, 7]], [[4, 8], [4, 8, 11]], [[4, 6, 8], [4, 6, 8, 11]],
                          [[4, 9], [1, 4, 9]], [[4, 6, 9], [1, 4, 6, 9]], [[4, 7, 9], [1, 4, 7, 9]],
                          [[4, 10], [4, 7, 10]], [[4, 6, 10], [1, 4, 6, 10]],
                          [[4, 7, 10], [0, 4, 7, 10]], [[4, 8, 10], [1, 4, 8, 10]],
                          [[4, 6, 8, 10], [1, 4, 6, 8, 10]], [[4, 11], [4, 8, 11]],
                          [[4, 6, 11], [4, 6, 8, 11]], [[4, 7, 11], [2, 4, 7, 11]],
                          [[4, 8, 11], [2, 4, 8, 11]], [[4, 9, 11], [2, 4, 9, 11]],
                          [[4, 6, 8, 11], [1, 4, 6, 8, 11]], [[4, 6, 9, 11], [2, 4, 6, 9, 11]],
                          [[4, 7, 9, 11], [2, 4, 7, 9, 11]], [[5], [0, 5, 9]], [[5, 7], [0, 4, 7]],
                          [[5, 8], [0, 5, 8]], [[5, 9], [0, 5, 9]], [[5, 7, 9], [0, 4, 7, 9]],
                          [[5, 10], [2, 5, 10]], [[5, 7, 10], [2, 5, 7, 10]],
                          [[5, 8, 10], [2, 5, 8, 10]], [[5, 11], [0, 5, 9]], [[5, 7, 11], [2, 5, 7, 11]],
                          [[5, 8, 11], [1, 5, 8, 11]], [[5, 9, 11], [2, 5, 9, 11]],
                          [[5, 7, 9, 11], [2, 5, 7, 9, 11]], [[6], [1, 6]], [[6, 8], [1, 5, 8]],
                          [[6, 9], [2, 6, 9]], [[6, 10], [1, 6, 10]], [[6, 8, 10], [1, 5, 8, 10]],
                          [[6, 11], [3, 6, 11]], [[6, 8, 11], [3, 6, 8, 11]],
                          [[6, 9, 11], [3, 6, 9, 11]], [[7], [2, 7, 11]], [[7, 9], [2, 6, 9]],
                          [[7, 10], [2, 7, 10]], [[7, 11], [2, 7, 11]], [[7, 9, 11], [2, 7, 9, 11]],
                          [[8], [3, 8]], [[8, 10], [3, 7, 10]], [[8, 11], [4, 8, 11]], [[9], [4, 9]],
                          [[9, 11], [4, 8, 11]], [[10], [2, 5, 10]], [[11], [6, 11]]]

###################################################################################

ALL_CHORDS_PAIRS_FILTERED = [[[0], [0, 4, 7]], [[0, 3], [0, 3, 7]], [[0, 3, 5], [0, 3, 5, 9]],
                            [[0, 3, 5, 8], [0, 3, 7, 10]], [[0, 3, 5, 9], [0, 3, 7, 10]],
                            [[0, 3, 5, 10], [0, 3, 5, 9]], [[0, 3, 7], [0, 3, 7, 10]],
                            [[0, 3, 7, 10], [0, 3, 5, 9]], [[0, 3, 8], [0, 3, 5, 8]],
                            [[0, 3, 9], [0, 3, 5, 9]], [[0, 3, 10], [0, 3, 7, 10]], [[0, 4], [0, 4, 7]],
                            [[0, 4, 6], [0, 4, 6, 9]], [[0, 4, 6, 9], [1, 4, 6, 9]],
                            [[0, 4, 6, 10], [0, 4, 7, 10]], [[0, 4, 7], [0, 4, 7, 10]],
                            [[0, 4, 7, 10], [1, 4, 7, 10]], [[0, 4, 8], [0, 4, 7, 10]],
                            [[0, 4, 9], [0, 4, 6, 9]], [[0, 4, 10], [0, 4, 7, 10]], [[0, 5], [0, 5, 9]],
                            [[0, 5, 8], [0, 3, 5, 8]], [[0, 5, 9], [0, 3, 5, 9]],
                            [[0, 5, 10], [0, 3, 5, 10]], [[0, 6], [0, 6, 9]], [[0, 6, 9], [0, 4, 6, 9]],
                            [[0, 6, 10], [0, 4, 7, 10]], [[0, 7], [0, 4, 7]], [[0, 7, 10], [0, 4, 7, 10]],
                            [[0, 8], [0, 3, 8]], [[0, 9], [0, 4, 9]], [[0, 10], [2, 5, 10]], [[1], [1, 8]],
                            [[1, 4], [1, 4, 9]], [[1, 4, 6], [1, 4, 6, 9]], [[1, 4, 6, 9], [1, 4, 8, 11]],
                            [[1, 4, 6, 10], [0, 3, 5, 9]], [[1, 4, 6, 11], [1, 4, 6, 9]],
                            [[1, 4, 7], [1, 4, 7, 10]], [[1, 4, 7, 10], [0, 4, 7, 10]],
                            [[1, 4, 7, 11], [1, 4, 6, 10]], [[1, 4, 8], [1, 4, 8, 11]],
                            [[1, 4, 8, 11], [1, 4, 6, 9]], [[1, 4, 9], [1, 4, 6, 9]],
                            [[1, 4, 10], [1, 4, 6, 10]], [[1, 4, 11], [1, 4, 8, 11]], [[1, 5], [1, 5, 8]],
                            [[1, 5, 8], [1, 5, 8, 11]], [[1, 5, 8, 11], [2, 5, 8, 11]],
                            [[1, 5, 9], [0, 3, 5, 9]], [[1, 5, 10], [0, 4, 7, 10]],
                            [[1, 5, 11], [1, 5, 8, 11]], [[1, 6], [1, 6, 10]], [[1, 6, 9], [1, 4, 6, 9]],
                            [[1, 6, 10], [1, 4, 6, 10]], [[1, 6, 11], [1, 4, 6, 11]], [[1, 7], [1, 4, 7]],
                            [[1, 7, 10], [1, 4, 7, 10]], [[1, 7, 11], [1, 4, 7, 11]], [[1, 8], [1, 5, 8]],
                            [[1, 8, 11], [1, 4, 8, 11]], [[1, 9], [1, 4, 9]], [[1, 10], [1, 5, 10]],
                            [[1, 11], [2, 6, 11]], [[2], [2, 9]], [[2, 5], [2, 5, 9]],
                            [[2, 5, 8], [2, 5, 8, 11]], [[2, 5, 8, 11], [1, 4, 7, 10]],
                            [[2, 5, 9], [0, 3, 5, 9]], [[2, 5, 10], [0, 3, 5, 9]],
                            [[2, 5, 11], [2, 5, 8, 11]], [[2, 6], [2, 6, 9]], [[2, 6, 9], [1, 4, 6, 9]],
                            [[2, 6, 10], [1, 4, 6, 10]], [[2, 6, 11], [1, 4, 6, 10]], [[2, 7], [2, 7, 11]],
                            [[2, 7, 10], [0, 4, 7, 10]], [[2, 7, 11], [1, 4, 6, 9]], [[2, 8], [4, 8, 11]],
                            [[2, 8, 11], [2, 5, 8, 11]], [[2, 9], [2, 6, 9]], [[2, 10], [2, 5, 10]],
                            [[2, 11], [2, 7, 11]], [[3], [3, 10]], [[3, 5], [3, 7, 10]],
                            [[3, 5, 8], [0, 3, 5, 8]], [[3, 5, 8, 11], [2, 5, 8, 11]],
                            [[3, 5, 9], [0, 3, 5, 9]], [[3, 5, 10], [0, 3, 5, 10]],
                            [[3, 5, 11], [3, 5, 8, 11]], [[3, 7], [3, 7, 10]], [[3, 7, 10], [0, 3, 7, 10]],
                            [[3, 7, 11], [0, 3, 7, 10]], [[3, 8], [0, 3, 8]], [[3, 8, 11], [3, 5, 8, 11]],
                            [[3, 9], [0, 3, 9]], [[3, 10], [3, 7, 10]], [[3, 11], [3, 8, 11]],
                            [[4], [4, 11]], [[4, 6], [4, 7, 11]], [[4, 6, 9], [1, 4, 6, 9]],
                            [[4, 6, 10], [1, 4, 6, 10]], [[4, 6, 11], [1, 4, 6, 11]], [[4, 7], [0, 4, 7]],
                            [[4, 7, 10], [0, 4, 7, 10]], [[4, 7, 11], [1, 4, 7, 11]], [[4, 8], [4, 8, 11]],
                            [[4, 8, 11], [1, 4, 8, 11]], [[4, 9], [1, 4, 9]], [[4, 10], [4, 7, 10]],
                            [[4, 11], [4, 8, 11]], [[5], [0, 5, 9]], [[5, 8], [0, 5, 8]],
                            [[5, 8, 11], [1, 5, 8, 11]], [[5, 9], [0, 5, 9]], [[5, 10], [2, 5, 10]],
                            [[5, 11], [0, 5, 9]], [[6], [1, 6]], [[6, 9], [2, 6, 9]],
                            [[6, 10], [1, 6, 10]], [[6, 11], [2, 6, 11]], [[7], [2, 7, 11]],
                            [[7, 10], [2, 7, 10]], [[7, 11], [2, 7, 11]], [[8], [3, 8]],
                            [[8, 11], [4, 8, 11]], [[9], [4, 9]], [[10], [2, 5, 10]], [[11], [6, 11]]]

###################################################################################

ALL_CHORDS_TRIPLETS_SORTED = [[[0], [0, 4, 7], [0]], [[0, 2], [0, 4, 7], [0]], [[0, 3], [0, 3, 7], [0]],
                              [[0, 4], [0, 4, 7], [0, 4]], [[0, 2, 4], [0, 2, 4, 7], [0]],
                              [[0, 5], [0, 5, 9], [0, 5]], [[0, 2, 5], [0, 2, 5, 9], [0, 2, 5]],
                              [[0, 3, 5], [0, 3, 5, 9], [0, 3, 5]], [[0, 6], [0, 2, 6, 9], [2]],
                              [[0, 2, 6], [0, 2, 6, 9], [0, 2, 6]], [[0, 3, 6], [0, 3, 6, 8], [0, 3, 6]],
                              [[0, 4, 6], [0, 4, 6, 9], [0, 4, 6]],
                              [[0, 2, 4, 6], [0, 2, 4, 6, 9], [0, 2, 4, 6]], [[0, 7], [0, 4, 7], [0, 7]],
                              [[0, 2, 7], [0, 2, 4, 7], [0, 2, 7]], [[0, 3, 7], [0, 3, 7, 10], [0, 3, 7]],
                              [[0, 4, 7], [0, 4, 7, 9], [0, 4, 7]], [[0, 5, 7], [0, 5, 7, 9], [0, 5, 7]],
                              [[0, 2, 4, 7], [0, 2, 4, 7, 9], [0, 2, 4, 7]],
                              [[0, 2, 5, 7], [0, 2, 5, 7, 9], [0, 2, 5, 7]],
                              [[0, 3, 5, 7], [0, 3, 5, 7, 10], [0, 3, 5, 7]], [[0, 8], [0, 3, 8], [8]],
                              [[0, 2, 8], [0, 2, 5, 8], [0, 2, 8]], [[0, 3, 8], [0, 3, 5, 8], [0, 3, 8]],
                              [[0, 4, 8], [2, 4, 8, 11], [0, 4, 9]], [[0, 5, 8], [0, 3, 5, 8], [0, 5, 8]],
                              [[0, 6, 8], [0, 3, 6, 8], [0, 6, 8]],
                              [[0, 2, 4, 8], [0, 2, 4, 6, 8], [0, 2, 4, 8]],
                              [[0, 2, 5, 8], [0, 2, 5, 8, 10], [0, 2, 5, 8]],
                              [[0, 2, 6, 8], [0, 2, 6, 8, 10], [0, 2, 6, 8]],
                              [[0, 3, 5, 8], [0, 3, 5, 8, 10], [0, 3, 5, 8]],
                              [[0, 3, 6, 8], [0, 3, 6, 8, 10], [0, 3, 6, 8]],
                              [[0, 4, 6, 8], [2, 4, 6, 8, 11], [2, 6, 8, 11]],
                              [[0, 2, 4, 6, 8], [2, 4, 6, 8, 11], [2, 6, 8, 11]], [[0, 9], [0, 4, 9], [9]],
                              [[0, 2, 9], [0, 2, 6, 9], [0, 2, 9]], [[0, 3, 9], [0, 3, 5, 9], [0, 3, 9]],
                              [[0, 4, 9], [0, 4, 7, 9], [0, 4, 9]], [[0, 5, 9], [0, 2, 5, 9], [0, 5, 9]],
                              [[0, 6, 9], [0, 2, 6, 9], [0, 6, 9]], [[0, 7, 9], [0, 4, 7, 9], [0, 7, 9]],
                              [[0, 2, 4, 9], [0, 2, 4, 7, 9], [0, 2, 4, 9]],
                              [[0, 2, 5, 9], [0, 2, 5, 7, 9], [0, 2, 5, 9]],
                              [[0, 2, 6, 9], [0, 2, 4, 6, 9], [0, 2, 6, 9]],
                              [[0, 2, 7, 9], [0, 2, 4, 7, 9], [0, 2, 7, 9]],
                              [[0, 3, 5, 9], [0, 3, 5, 7, 9], [0, 3, 5, 9]],
                              [[0, 3, 6, 9], [0, 2, 4, 6, 9], [4, 6, 9]],
                              [[0, 3, 7, 9], [0, 3, 5, 7, 9], [0, 3, 7, 9]],
                              [[0, 4, 6, 9], [0, 2, 4, 6, 9], [0, 4, 6, 9]],
                              [[0, 4, 7, 9], [0, 2, 4, 7, 9], [0, 4, 7, 9]],
                              [[0, 5, 7, 9], [0, 2, 5, 7, 9], [0, 5, 7, 9]],
                              [[0, 2, 4, 6, 9], [2, 4, 6, 9, 11], [0, 2, 4, 6, 9]],
                              [[0, 2, 4, 7, 9], [2, 4, 7, 9, 11], [0, 2, 4, 7, 9]],
                              [[0, 2, 5, 7, 9], [2, 5, 7, 9, 11], [7]],
                              [[0, 3, 5, 7, 9], [2, 4, 6, 8, 11], [1, 4, 6, 8, 10]],
                              [[0, 10], [2, 5, 10], [10]], [[0, 2, 10], [0, 2, 5, 10], [10]],
                              [[0, 3, 10], [0, 3, 7, 10], [0, 3, 10]],
                              [[0, 4, 10], [0, 4, 7, 10], [0, 4, 10]],
                              [[0, 5, 10], [0, 2, 5, 10], [0, 5, 10]],
                              [[0, 6, 10], [0, 3, 6, 10], [0, 6, 10]],
                              [[0, 7, 10], [0, 4, 7, 10], [0, 7, 10]], [[0, 8, 10], [0, 3, 8, 10], [8]],
                              [[0, 2, 4, 10], [0, 2, 4, 7, 10], [0, 4, 10]],
                              [[0, 2, 5, 10], [0, 2, 5, 7, 10], [0, 2, 5, 10]],
                              [[0, 2, 6, 10], [0, 2, 6, 8, 10], [8]],
                              [[0, 2, 7, 10], [0, 2, 5, 7, 10], [2, 7, 10]],
                              [[0, 2, 8, 10], [0, 2, 5, 8, 10], [8, 10]],
                              [[0, 3, 5, 10], [0, 3, 5, 7, 10], [0, 3, 5, 10]],
                              [[0, 3, 6, 10], [0, 3, 6, 8, 10], [0, 3, 6, 10]],
                              [[0, 3, 7, 10], [0, 3, 5, 7, 10], [0, 3, 7, 10]],
                              [[0, 3, 8, 10], [0, 3, 5, 8, 10], [0, 3, 8, 10]],
                              [[0, 4, 6, 10], [0, 2, 4, 6, 10], [2]],
                              [[0, 4, 7, 10], [0, 2, 4, 7, 10], [0, 4, 7, 10]],
                              [[0, 4, 8, 10], [0, 2, 4, 8, 10], [0, 4, 8, 10]],
                              [[0, 5, 7, 10], [0, 3, 5, 7, 10], [0, 5, 7, 10]],
                              [[0, 5, 8, 10], [0, 3, 5, 8, 10], [10]],
                              [[0, 6, 8, 10], [0, 3, 6, 8, 10], [6]],
                              [[0, 2, 4, 6, 10], [0, 2, 4, 8, 10], [0, 2, 6, 8, 10]],
                              [[0, 2, 4, 7, 10], [1, 3, 6, 9, 11], [0, 2, 5, 8, 10]],
                              [[0, 2, 4, 8, 10], [1, 3, 7, 9, 11], [0, 2, 6, 8, 10]],
                              [[0, 2, 5, 7, 10], [0, 3, 5, 7, 10], [5, 10]],
                              [[0, 2, 5, 8, 10], [1, 4, 7, 9, 11], [8]],
                              [[0, 2, 6, 8, 10], [2, 4, 6, 8, 10], [0, 2, 6, 8, 10]],
                              [[0, 3, 5, 7, 10], [0, 2, 5, 7, 10], [9]],
                              [[0, 3, 5, 8, 10], [1, 3, 5, 8, 10], [0, 3, 5, 8, 10]],
                              [[0, 3, 6, 8, 10], [1, 3, 6, 8, 10], [0, 3, 6, 8, 10]],
                              [[0, 4, 6, 8, 10], [0, 2, 4, 6, 9], [1, 3, 5, 8, 10]],
                              [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10]],
                              [[1], [1, 8], [1]], [[1, 3], [1, 5, 8], [1]], [[1, 4], [1, 4, 9], [9]],
                              [[1, 5], [1, 5, 8], [1, 5]], [[1, 3, 5], [1, 3, 5, 10], [1, 3, 5]],
                              [[1, 6], [1, 6, 10], [1, 6]], [[1, 3, 6], [1, 3, 6, 10], [1, 3, 6]],
                              [[1, 4, 6], [1, 4, 6, 9], [1, 4, 6]], [[1, 7], [1, 4, 7], [1, 7]],
                              [[1, 3, 7], [1, 3, 7, 10], [1, 3, 7]], [[1, 4, 7], [1, 4, 7, 9], [1, 4, 7]],
                              [[1, 5, 7], [1, 5, 7, 10], [1, 5, 7]], [[1, 3, 5, 7], [1, 3, 5, 7, 10], [7]],
                              [[1, 8], [1, 5, 8], [1, 8]], [[1, 3, 8], [1, 3, 5, 8], [1, 3, 8]],
                              [[1, 4, 8], [1, 4, 8, 11], [1, 4, 8]], [[1, 5, 8], [1, 5, 8, 10], [1, 5, 8]],
                              [[1, 6, 8], [1, 3, 6, 8], [1, 6, 8]],
                              [[1, 3, 5, 8], [1, 3, 5, 8, 10], [1, 3, 5, 8]],
                              [[1, 3, 6, 8], [1, 3, 6, 8, 10], [1, 3, 6, 8]],
                              [[1, 4, 6, 8], [1, 4, 6, 8, 11], [1, 4, 6, 8]], [[1, 9], [1, 4, 9], [9]],
                              [[1, 3, 9], [1, 3, 6, 9], [1, 3, 9]], [[1, 4, 9], [1, 4, 6, 9], [1, 4, 9]],
                              [[1, 5, 9], [0, 3, 5, 9], [0, 5, 9]], [[1, 6, 9], [1, 4, 6, 9], [1, 6, 9]],
                              [[1, 7, 9], [1, 4, 7, 9], [1, 7, 9]],
                              [[1, 3, 5, 9], [0, 3, 5, 7, 9], [1, 5, 9]],
                              [[1, 3, 6, 9], [1, 3, 6, 9, 11], [1, 3, 6, 9]],
                              [[1, 3, 7, 9], [1, 3, 5, 7, 9], [1, 7]],
                              [[1, 4, 6, 9], [1, 4, 6, 9, 11], [1, 4, 6, 9]],
                              [[1, 4, 7, 9], [1, 4, 7, 9, 11], [1, 4, 7, 9]],
                              [[1, 5, 7, 9], [1, 3, 7, 9, 11], [1, 5, 7, 9]],
                              [[1, 3, 5, 7, 9], [2, 4, 6, 8, 11], [9]], [[1, 10], [1, 5, 10], [10]],
                              [[1, 3, 10], [1, 3, 7, 10], [1, 3, 10]],
                              [[1, 4, 10], [1, 4, 6, 10], [1, 4, 10]],
                              [[1, 5, 10], [1, 5, 8, 10], [1, 5, 10]],
                              [[1, 6, 10], [1, 4, 6, 10], [1, 6, 10]],
                              [[1, 7, 10], [1, 3, 7, 10], [1, 7, 10]], [[1, 8, 10], [1, 5, 8, 10], [10]],
                              [[1, 3, 5, 10], [1, 3, 5, 8, 10], [1, 3, 5, 10]],
                              [[1, 3, 6, 10], [1, 3, 6, 8, 10], [1, 3, 6, 10]],
                              [[1, 3, 7, 10], [1, 3, 5, 7, 10], [1, 3, 7, 10]],
                              [[1, 3, 8, 10], [1, 3, 5, 8, 10], [1, 3, 8, 10]],
                              [[1, 4, 6, 10], [1, 4, 6, 8, 10], [1, 4, 6, 10]],
                              [[1, 4, 7, 10], [0, 2, 4, 7, 10], [0, 4, 7, 10]],
                              [[1, 4, 8, 10], [1, 4, 6, 8, 10], [1, 4, 8, 10]],
                              [[1, 5, 7, 10], [1, 3, 5, 7, 10], [1, 5, 7, 10]],
                              [[1, 5, 8, 10], [1, 3, 5, 8, 10], [1, 5, 8, 10]],
                              [[1, 6, 8, 10], [1, 3, 6, 8, 10], [1, 6, 8, 10]],
                              [[1, 3, 5, 7, 10], [2, 4, 6, 8, 11], [0, 3, 5, 7, 9]],
                              [[1, 3, 5, 8, 10], [0, 3, 5, 8, 10], [6, 8, 10]],
                              [[1, 3, 6, 8, 10], [0, 3, 6, 8, 10], [8]],
                              [[1, 4, 6, 8, 10], [0, 3, 5, 7, 9], [2, 4, 6, 8, 11]],
                              [[1, 11], [2, 6, 11], [11]], [[1, 3, 11], [1, 3, 6, 11], [11]],
                              [[1, 4, 11], [1, 4, 8, 11], [1]], [[1, 5, 11], [1, 5, 8, 11], [1, 5, 11]],
                              [[1, 6, 11], [1, 4, 6, 11], [1, 6, 11]],
                              [[1, 7, 11], [1, 4, 7, 11], [1, 7, 11]],
                              [[1, 8, 11], [1, 4, 8, 11], [1, 8, 11]], [[1, 9, 11], [1, 4, 9, 11], [9]],
                              [[1, 3, 5, 11], [1, 3, 5, 8, 11], [1, 3, 5, 11]],
                              [[1, 3, 6, 11], [1, 3, 6, 8, 11], [1, 3, 6, 11]],
                              [[1, 3, 7, 11], [1, 3, 7, 9, 11], [0]],
                              [[1, 3, 8, 11], [1, 3, 6, 8, 11], [1, 3, 8, 11]],
                              [[1, 3, 9, 11], [1, 3, 6, 9, 11], [1, 3, 9, 11]],
                              [[1, 4, 6, 11], [1, 4, 6, 9, 11], [1, 4, 6, 11]],
                              [[1, 4, 7, 11], [1, 4, 7, 9, 11], [1, 4, 7, 11]],
                              [[1, 4, 8, 11], [1, 4, 6, 8, 11], [1, 4, 8, 11]],
                              [[1, 4, 9, 11], [1, 4, 6, 9, 11], [1, 4, 9, 11]],
                              [[1, 5, 7, 11], [0, 4, 6, 8, 10], [5, 7, 9, 11]],
                              [[1, 5, 8, 11], [1, 3, 5, 8, 11], [1, 5, 8, 11]],
                              [[1, 5, 9, 11], [1, 5, 7, 9, 11], [9]],
                              [[1, 6, 8, 11], [1, 3, 6, 8, 11], [1, 6, 8, 11]],
                              [[1, 6, 9, 11], [1, 4, 6, 9, 11], [1, 6, 9, 11]],
                              [[1, 7, 9, 11], [1, 4, 7, 9, 11], [1, 7, 9, 11]],
                              [[1, 3, 5, 7, 11], [0, 2, 4, 6, 8], [7, 9]],
                              [[1, 3, 5, 8, 11], [0, 2, 4, 7, 10], [1, 3, 6, 9, 11]],
                              [[1, 3, 5, 9, 11], [1, 3, 7, 9, 11], [0, 2, 6, 8, 10]],
                              [[1, 3, 6, 8, 11], [1, 4, 6, 8, 11], [6, 8, 11]],
                              [[1, 3, 6, 9, 11], [0, 2, 5, 8, 10], [1, 4, 7, 9, 11]],
                              [[1, 3, 7, 9, 11], [1, 3, 6, 9, 11], [11]],
                              [[1, 4, 6, 8, 11], [1, 4, 6, 9, 11], [9, 11]],
                              [[1, 4, 6, 9, 11], [2, 4, 6, 9, 11], [1, 4, 6, 9, 11]],
                              [[1, 4, 7, 9, 11], [2, 4, 7, 9, 11], [7, 9, 11]],
                              [[1, 5, 7, 9, 11], [2, 4, 7, 9, 11], [5, 7, 9]],
                              [[1, 3, 5, 7, 9, 11], [0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
                              [[2], [2, 9], [2]], [[2, 4], [2, 6, 9], [2]], [[2, 5], [2, 5, 9], [2]],
                              [[2, 6], [2, 6, 9], [2]], [[2, 4, 6], [2, 4, 6, 9], [2, 4, 6]],
                              [[2, 7], [2, 7, 11], [2, 7]], [[2, 4, 7], [2, 4, 7, 11], [2, 4, 7]],
                              [[2, 5, 7], [2, 5, 7, 11], [2, 5, 7]], [[2, 8], [4, 8, 11], [4]],
                              [[2, 4, 8], [2, 4, 8, 11], [2, 4, 8]], [[2, 5, 8], [2, 5, 8, 10], [2, 5, 8]],
                              [[2, 6, 8], [2, 6, 8, 11], [2, 6, 8]],
                              [[2, 4, 6, 8], [2, 4, 6, 8, 11], [2, 4, 6, 8]], [[2, 9], [2, 6, 9], [2, 9]],
                              [[2, 4, 9], [2, 4, 6, 9], [2, 4, 9]], [[2, 5, 9], [0, 2, 5, 9], [2, 5, 9]],
                              [[2, 6, 9], [2, 6, 9, 11], [2, 6, 9]], [[2, 7, 9], [2, 7, 9, 11], [2, 7, 9]],
                              [[2, 4, 6, 9], [2, 4, 6, 9, 11], [2, 4, 6, 9]],
                              [[2, 4, 7, 9], [2, 4, 7, 9, 11], [2, 4, 7, 9]],
                              [[2, 5, 7, 9], [0, 2, 5, 7, 9], [2, 5, 7, 9]], [[2, 10], [2, 5, 10], [10]],
                              [[2, 4, 10], [2, 4, 7, 10], [2, 4, 10]],
                              [[2, 5, 10], [2, 5, 7, 10], [2, 5, 10]],
                              [[2, 6, 10], [1, 4, 6, 10], [1, 6, 10]],
                              [[2, 7, 10], [2, 5, 7, 10], [2, 7, 10]],
                              [[2, 8, 10], [2, 5, 8, 10], [2, 8, 10]],
                              [[2, 4, 6, 10], [0, 2, 4, 6, 10], [2, 4, 6, 10]],
                              [[2, 4, 7, 10], [0, 2, 4, 7, 10], [2, 4, 7, 10]],
                              [[2, 4, 8, 10], [2, 4, 7, 9, 11], [2, 4, 7, 11]],
                              [[2, 5, 7, 10], [0, 2, 5, 7, 10], [2, 5, 7, 10]],
                              [[2, 5, 8, 10], [0, 2, 5, 8, 10], [2, 5, 8, 10]],
                              [[2, 6, 8, 10], [1, 3, 5, 7, 10], [1, 7]],
                              [[2, 4, 6, 8, 10], [0, 2, 6, 8, 10], [2, 4, 6, 8, 10]],
                              [[2, 11], [2, 7, 11], [7]], [[2, 4, 11], [2, 4, 8, 11], [2, 4, 11]],
                              [[2, 5, 11], [2, 5, 7, 11], [2, 5, 11]],
                              [[2, 6, 11], [2, 6, 9, 11], [2, 6, 11]],
                              [[2, 7, 11], [2, 4, 7, 11], [2, 7, 11]],
                              [[2, 8, 11], [2, 4, 8, 11], [2, 8, 11]],
                              [[2, 9, 11], [2, 6, 9, 11], [2, 9, 11]],
                              [[2, 4, 6, 11], [2, 4, 6, 9, 11], [2, 4, 6, 11]],
                              [[2, 4, 7, 11], [2, 4, 7, 9, 11], [2, 4, 7, 11]],
                              [[2, 4, 8, 11], [2, 4, 6, 8, 11], [2, 4, 8, 11]],
                              [[2, 4, 9, 11], [2, 4, 7, 9, 11], [2, 4, 9, 11]],
                              [[2, 5, 7, 11], [2, 5, 7, 9, 11], [2, 5, 7, 11]],
                              [[2, 5, 8, 11], [1, 3, 5, 8, 11], [1, 5, 8, 11]],
                              [[2, 5, 9, 11], [2, 5, 7, 9, 11], [2, 5, 9, 11]],
                              [[2, 6, 8, 11], [2, 4, 6, 8, 11], [2, 6, 8, 11]],
                              [[2, 6, 9, 11], [2, 4, 6, 9, 11], [2, 6, 9, 11]],
                              [[2, 7, 9, 11], [2, 4, 7, 9, 11], [2, 7, 9, 11]],
                              [[2, 4, 6, 8, 11], [2, 4, 6, 9, 11], [2, 4, 6, 8, 11]],
                              [[2, 4, 6, 9, 11], [2, 4, 7, 9, 11], [2, 7, 9]],
                              [[2, 4, 7, 9, 11], [0, 2, 4, 7, 9], [11]],
                              [[2, 5, 7, 9, 11], [2, 4, 7, 9, 11], [2, 7, 9, 11]], [[3], [3, 10], [3]],
                              [[3, 5], [3, 7, 10], [3]], [[3, 6], [3, 6, 11], [11]],
                              [[3, 7], [3, 7, 10], [3]], [[3, 5, 7], [3, 5, 7, 10], [3, 5, 7]],
                              [[3, 8], [0, 3, 8], [3, 8]], [[3, 5, 8], [0, 3, 5, 8], [8]],
                              [[3, 6, 8], [0, 3, 6, 8], [3, 6, 8]], [[3, 9], [0, 3, 9], [3, 9]],
                              [[3, 5, 9], [0, 3, 5, 9], [3, 5, 9]], [[3, 6, 9], [3, 6, 9, 11], [3, 6, 9]],
                              [[3, 7, 9], [0, 3, 7, 9], [3, 7, 9]],
                              [[3, 5, 7, 9], [0, 3, 5, 7, 9], [0, 3, 5, 9]], [[3, 10], [3, 7, 10], [3, 10]],
                              [[3, 5, 10], [3, 5, 7, 10], [3, 5, 10]],
                              [[3, 6, 10], [1, 3, 6, 10], [3, 6, 10]],
                              [[3, 7, 10], [0, 3, 7, 10], [3, 7, 10]],
                              [[3, 8, 10], [0, 3, 8, 10], [3, 8, 10]],
                              [[3, 5, 7, 10], [0, 3, 5, 7, 10], [3, 5, 7, 10]],
                              [[3, 5, 8, 10], [0, 3, 5, 8, 10], [3, 5, 8, 10]],
                              [[3, 6, 8, 10], [1, 3, 6, 8, 10], [3, 6, 8, 10]], [[3, 11], [3, 6, 11], [11]],
                              [[3, 5, 11], [3, 5, 8, 11], [3, 5, 11]],
                              [[3, 6, 11], [3, 6, 9, 11], [3, 6, 11]],
                              [[3, 7, 11], [2, 5, 7, 11], [2, 7, 11]],
                              [[3, 8, 11], [3, 6, 8, 11], [3, 8, 11]],
                              [[3, 9, 11], [3, 6, 9, 11], [3, 9, 11]],
                              [[3, 5, 7, 11], [3, 5, 7, 9, 11], [3, 5, 7, 11]],
                              [[3, 5, 8, 11], [1, 3, 5, 8, 11], [3, 5, 8, 11]],
                              [[3, 5, 9, 11], [3, 5, 7, 9, 11], [5, 7, 9, 11]],
                              [[3, 6, 8, 11], [1, 3, 6, 8, 11], [3, 6, 8, 11]],
                              [[3, 6, 9, 11], [1, 3, 6, 9, 11], [3, 6, 9, 11]],
                              [[3, 7, 9, 11], [2, 4, 7, 9, 11], [7, 9, 11]],
                              [[3, 5, 7, 9, 11], [2, 5, 7, 9, 11], [2, 5, 7, 11]], [[4], [4, 11], [4]],
                              [[4, 6], [4, 7, 11], [4]], [[4, 7], [0, 4, 7], [0]], [[4, 8], [4, 8, 11], [4]],
                              [[4, 6, 8], [4, 6, 8, 11], [4]], [[4, 9], [1, 4, 9], [4, 9]],
                              [[4, 6, 9], [1, 4, 6, 9], [4, 6, 9]], [[4, 7, 9], [1, 4, 7, 9], [4, 7, 9]],
                              [[4, 10], [4, 7, 10], [4, 10]], [[4, 6, 10], [1, 4, 6, 10], [4, 6, 10]],
                              [[4, 7, 10], [0, 4, 7, 10], [4, 7, 10]], [[4, 8, 10], [1, 4, 8, 10], [1]],
                              [[4, 6, 8, 10], [1, 4, 6, 8, 10], [6]], [[4, 11], [4, 8, 11], [4, 11]],
                              [[4, 6, 11], [4, 6, 8, 11], [4, 6, 11]],
                              [[4, 7, 11], [2, 4, 7, 11], [4, 7, 11]],
                              [[4, 8, 11], [2, 4, 8, 11], [4, 8, 11]],
                              [[4, 9, 11], [2, 4, 9, 11], [4, 9, 11]],
                              [[4, 6, 8, 11], [1, 4, 6, 8, 11], [4, 6, 8, 11]],
                              [[4, 6, 9, 11], [2, 4, 6, 9, 11], [4, 6, 9, 11]],
                              [[4, 7, 9, 11], [2, 4, 7, 9, 11], [4, 7, 9, 11]], [[5], [0, 5, 9], [5]],
                              [[5, 7], [0, 4, 7], [0]], [[5, 8], [0, 5, 8], [5]], [[5, 9], [0, 5, 9], [5]],
                              [[5, 7, 9], [0, 4, 7, 9], [5]], [[5, 10], [2, 5, 10], [5, 10]],
                              [[5, 7, 10], [2, 5, 7, 10], [7]], [[5, 8, 10], [2, 5, 8, 10], [5, 8, 10]],
                              [[5, 11], [0, 5, 9], [5]], [[5, 7, 11], [2, 5, 7, 11], [5, 7, 11]],
                              [[5, 8, 11], [1, 5, 8, 11], [5, 8, 11]],
                              [[5, 9, 11], [2, 5, 9, 11], [5, 9, 11]],
                              [[5, 7, 9, 11], [2, 5, 7, 9, 11], [5, 7, 9]], [[6], [1, 6], [6]],
                              [[6, 8], [1, 5, 8], [8]], [[6, 9], [2, 6, 9], [2]], [[6, 10], [1, 6, 10], [6]],
                              [[6, 8, 10], [1, 5, 8, 10], [6, 8, 10]], [[6, 11], [3, 6, 11], [6, 11]],
                              [[6, 8, 11], [3, 6, 8, 11], [6, 8, 11]],
                              [[6, 9, 11], [3, 6, 9, 11], [6, 9, 11]], [[7], [2, 7, 11], [7]],
                              [[7, 9], [2, 6, 9], [2]], [[7, 10], [2, 7, 10], [7]],
                              [[7, 11], [2, 7, 11], [7]], [[7, 9, 11], [2, 7, 9, 11], [7, 9, 11]],
                              [[8], [3, 8], [8]], [[8, 10], [3, 7, 10], [3]], [[8, 11], [4, 8, 11], [4]],
                              [[9], [4, 9], [9]], [[9, 11], [4, 8, 11], [4]], [[10], [2, 5, 10], [10]],
                              [[11], [6, 11], [11]]]

###################################################################################

ALL_CHORDS_TRIPLETS_FILTERED = [[[0], [0, 4, 7], [7]], [[0, 3], [0, 3, 7], [0]],
                                [[0, 3, 5], [0, 3, 5, 9], [5]], [[0, 3, 5, 8], [0, 3, 7, 10], [0]],
                                [[0, 3, 5, 9], [0, 3, 7, 10], [10]], [[0, 3, 5, 10], [0, 3, 5, 9], [5]],
                                [[0, 3, 7], [0, 3, 7, 10], [0]], [[0, 3, 7, 10], [0, 3, 5, 9], [2, 5, 10]],
                                [[0, 3, 8], [0, 3, 5, 8], [8]], [[0, 3, 9], [0, 3, 5, 9], [5]],
                                [[0, 3, 10], [0, 3, 7, 10], [0]], [[0, 4], [0, 4, 7], [0]],
                                [[0, 4, 6], [0, 4, 6, 9], [4]], [[0, 4, 6, 9], [1, 4, 6, 9], [9]],
                                [[0, 4, 6, 10], [0, 4, 7, 10], [0, 4, 10]], [[0, 4, 7], [0, 4, 7, 10], [0]],
                                [[0, 4, 7, 10], [1, 4, 7, 10], [0]], [[0, 4, 8], [0, 4, 7, 10], [0, 5, 8]],
                                [[0, 4, 9], [0, 4, 6, 9], [9]], [[0, 4, 10], [0, 4, 7, 10], [0]],
                                [[0, 5], [0, 5, 9], [5]], [[0, 5, 8], [0, 3, 5, 8], [5]],
                                [[0, 5, 9], [0, 3, 5, 9], [5]], [[0, 5, 10], [0, 3, 5, 10], [10]],
                                [[0, 6], [0, 6, 9], [9]], [[0, 6, 9], [0, 4, 6, 9], [6]],
                                [[0, 6, 10], [0, 4, 7, 10], [10]], [[0, 7], [0, 4, 7], [0]],
                                [[0, 7, 10], [0, 4, 7, 10], [0]], [[0, 8], [0, 3, 8], [8]],
                                [[0, 9], [0, 4, 9], [9]], [[0, 10], [2, 5, 10], [10]], [[1], [1, 8], [8]],
                                [[1, 4], [1, 4, 9], [9]], [[1, 4, 6], [1, 4, 6, 9], [6]],
                                [[1, 4, 6, 9], [1, 4, 8, 11], [4]], [[1, 4, 6, 10], [0, 3, 5, 9], [5]],
                                [[1, 4, 6, 11], [1, 4, 6, 9], [6]], [[1, 4, 7], [1, 4, 7, 10], [10]],
                                [[1, 4, 7, 10], [0, 4, 7, 10], [0]],
                                [[1, 4, 7, 11], [1, 4, 6, 10], [1, 6, 10]], [[1, 4, 8], [1, 4, 8, 11], [1]],
                                [[1, 4, 8, 11], [1, 4, 6, 9], [1, 4, 9]], [[1, 4, 9], [1, 4, 6, 9], [9]],
                                [[1, 4, 10], [1, 4, 6, 10], [6]], [[1, 4, 11], [1, 4, 8, 11], [1]],
                                [[1, 5], [1, 5, 8], [1]], [[1, 5, 8], [1, 5, 8, 11], [1]],
                                [[1, 5, 8, 11], [2, 5, 8, 11], [1]], [[1, 5, 9], [0, 3, 5, 9], [0, 5, 9]],
                                [[1, 5, 10], [0, 4, 7, 10], [0]], [[1, 5, 11], [1, 5, 8, 11], [11]],
                                [[1, 6], [1, 6, 10], [6]], [[1, 6, 9], [1, 4, 6, 9], [6]],
                                [[1, 6, 10], [1, 4, 6, 10], [6]], [[1, 6, 11], [1, 4, 6, 11], [11]],
                                [[1, 7], [1, 4, 7], [4]], [[1, 7, 10], [1, 4, 7, 10], [4]],
                                [[1, 7, 11], [1, 4, 7, 11], [7]], [[1, 8], [1, 5, 8], [1]],
                                [[1, 8, 11], [1, 4, 8, 11], [1]], [[1, 9], [1, 4, 9], [9]],
                                [[1, 10], [1, 5, 10], [10]], [[1, 11], [2, 6, 11], [11]], [[2], [2, 9], [9]],
                                [[2, 5], [2, 5, 9], [2]], [[2, 5, 8], [2, 5, 8, 11], [2]],
                                [[2, 5, 8, 11], [1, 4, 7, 10], [0, 3, 8]],
                                [[2, 5, 9], [0, 3, 5, 9], [2, 5, 10]], [[2, 5, 10], [0, 3, 5, 9], [2, 10]],
                                [[2, 5, 11], [2, 5, 8, 11], [8]], [[2, 6], [2, 6, 9], [2]],
                                [[2, 6, 9], [1, 4, 6, 9], [1, 4, 9]], [[2, 6, 10], [1, 4, 6, 10], [1, 6, 10]],
                                [[2, 6, 11], [1, 4, 6, 10], [1, 6, 10]], [[2, 7], [2, 7, 11], [7]],
                                [[2, 7, 10], [0, 4, 7, 10], [0]], [[2, 7, 11], [1, 4, 6, 9], [1, 4, 9]],
                                [[2, 8], [4, 8, 11], [4]], [[2, 8, 11], [2, 5, 8, 11], [4]],
                                [[2, 9], [2, 6, 9], [2]], [[2, 10], [2, 5, 10], [10]],
                                [[2, 11], [2, 7, 11], [7]], [[3], [3, 10], [10]], [[3, 5], [3, 7, 10], [3]],
                                [[3, 5, 8], [0, 3, 5, 8], [8]], [[3, 5, 8, 11], [2, 5, 8, 11], [2]],
                                [[3, 5, 9], [0, 3, 5, 9], [5]], [[3, 5, 10], [0, 3, 5, 10], [5, 10]],
                                [[3, 5, 11], [3, 5, 8, 11], [5]], [[3, 7], [3, 7, 10], [3]],
                                [[3, 7, 10], [0, 3, 7, 10], [10]], [[3, 7, 11], [0, 3, 7, 10], [3, 7, 10]],
                                [[3, 8], [0, 3, 8], [8]], [[3, 8, 11], [3, 5, 8, 11], [11]],
                                [[3, 9], [0, 3, 9], [9]], [[3, 10], [3, 7, 10], [3]],
                                [[3, 11], [3, 8, 11], [8]], [[4], [4, 11], [11]], [[4, 6], [4, 7, 11], [4]],
                                [[4, 6, 9], [1, 4, 6, 9], [9]], [[4, 6, 10], [1, 4, 6, 10], [6]],
                                [[4, 6, 11], [1, 4, 6, 11], [11]], [[4, 7], [0, 4, 7], [0]],
                                [[4, 7, 10], [0, 4, 7, 10], [0]], [[4, 7, 11], [1, 4, 7, 11], [11]],
                                [[4, 8], [4, 8, 11], [4]], [[4, 8, 11], [1, 4, 8, 11], [4]],
                                [[4, 9], [1, 4, 9], [9]], [[4, 10], [4, 7, 10], [7]],
                                [[4, 11], [4, 8, 11], [4]], [[5], [0, 5, 9], [0]], [[5, 8], [0, 5, 8], [5]],
                                [[5, 8, 11], [1, 5, 8, 11], [1]], [[5, 9], [0, 5, 9], [5]],
                                [[5, 10], [2, 5, 10], [10]], [[5, 11], [0, 5, 9], [5]], [[6], [1, 6], [1]],
                                [[6, 9], [2, 6, 9], [2]], [[6, 10], [1, 6, 10], [6]],
                                [[6, 11], [2, 6, 11], [11]], [[7], [2, 7, 11], [2]],
                                [[7, 10], [2, 7, 10], [7]], [[7, 11], [2, 7, 11], [7]], [[8], [3, 8], [3]],
                                [[8, 11], [4, 8, 11], [4]], [[9], [4, 9], [4]], [[10], [2, 5, 10], [5]],
                                [[11], [6, 11], [6]]]

###################################################################################

def pitches_to_tones(pitches):
  return [p % 12 for p in pitches]

###################################################################################

def tones_to_pitches(tones, base_octave=5):
  return [(base_octave * 12) + t for t in tones]

###################################################################################

def find_closest_value(lst, val):

  closest_value = min(lst, key=lambda x: abs(val - x))
  closest_value_indexes = [i for i in range(len(lst)) if lst[i] == closest_value]
  
  return [closest_value, abs(val - closest_value), closest_value_indexes]

###################################################################################

def transpose_tones_chord(tones_chord, transpose_value=0):
  return sorted([((60+t)+transpose_value) % 12 for t in sorted(set(tones_chord))])

###################################################################################

def transpose_tones(tones, transpose_value=0):
  return [((60+t)+transpose_value) % 12 for t in tones]

###################################################################################

def transpose_pitches_chord(pitches_chord, transpose_value=0):
  return [max(1, min(127, p+transpose_value)) for p in sorted(set(pitches_chord), reverse=True)]

###################################################################################

def transpose_pitches(pitches, transpose_value=0):
  return [max(1, min(127, p+transpose_value)) for p in pitches]

###################################################################################

def reverse_enhanced_score_notes(escore_notes):

  score = recalculate_score_timings(escore_notes)

  ematrix = escore_notes_to_escore_matrix(score, reverse_matrix=True)
  e_score = escore_matrix_to_original_escore_notes(ematrix)

  reversed_score = recalculate_score_timings(e_score)

  return reversed_score

###################################################################################

def count_patterns(lst, sublist):
    count = 0
    idx = 0
    for i in range(len(lst) - len(sublist) + 1):
        if lst[idx:idx + len(sublist)] == sublist:
            count += 1
            idx += len(sublist)
        else:
          idx += 1
    return count

###################################################################################

def find_lrno_patterns(seq):

  all_seqs = Counter()

  max_pat_len = math.ceil(len(seq) / 2)

  num_iter = 0

  for i in range(len(seq)):
    for j in range(i+1, len(seq)+1):
      if j-i <= max_pat_len:
        all_seqs[tuple(seq[i:j])] += 1
        num_iter += 1

  max_count = 0
  max_len = 0

  for val, count in all_seqs.items():

    if max_len < len(val):
      max_count = max(2, count)

    if count > 1:
      max_len = max(max_len, len(val))
      pval = val

  max_pats = []

  for val, count in all_seqs.items():
    if count == max_count and len(val) == max_len:
      max_pats.append(val)

  found_patterns = []

  for pat in max_pats:
    count = count_patterns(seq, list(pat))
    if count > 1:
      found_patterns.append([count, len(pat), pat])

  return found_patterns

###################################################################################

def delta_pitches(escore_notes, pitches_index=4):

  pitches = [p[pitches_index] for p in escore_notes]
  
  return [a-b for a, b in zip(pitches[:-1], pitches[1:])]

###################################################################################

def split_list(lst, val):
    return [lst[i:j] for i, j in zip([0] + [k + 1 for k, x in enumerate(lst) if x == val], [k for k, x in enumerate(lst) if x == val] + [len(lst)]) if j > i]

###################################################################################

def even_timings(escore_notes, 
                 times_idx=1, 
                 durs_idx=2
                 ):

  esn = copy.deepcopy(escore_notes)

  for e in esn:

    if e[times_idx] != 0:
      if e[times_idx] % 2 != 0:
        e[times_idx] += 1

    if e[durs_idx] % 2 != 0:
      e[durs_idx] += 1

  return esn

###################################################################################

def delta_score_to_abs_score(delta_score_notes, 
                            times_idx=1
                            ):

  abs_score = copy.deepcopy(delta_score_notes)

  abs_time = 0

  for i, e in enumerate(delta_score_notes):

    dtime = e[times_idx]
    
    abs_time += dtime

    abs_score[i][times_idx] = abs_time
    
  return abs_score

###################################################################################


def adjust_numbers_to_sum(numbers, target_sum):

  current_sum = sum(numbers)
  difference = target_sum - current_sum

  non_zero_elements = [(i, num) for i, num in enumerate(numbers) if num != 0]

  total_non_zero = sum(num for _, num in non_zero_elements)

  increments = []
  for i, num in non_zero_elements:
      proportion = num / total_non_zero
      increment = proportion * difference
      increments.append(increment)

  for idx, (i, num) in enumerate(non_zero_elements):
      numbers[i] += int(round(increments[idx]))

  current_sum = sum(numbers)
  difference = target_sum - current_sum
  non_zero_indices = [i for i, num in enumerate(numbers) if num != 0]

  for i in range(abs(difference)):
      numbers[non_zero_indices[i % len(non_zero_indices)]] += 1 if difference > 0 else -1

  return numbers

###################################################################################

def find_next_bar(escore_notes, bar_time, start_note_idx, cur_bar):
  for e in escore_notes[start_note_idx:]:
    if e[1] // bar_time > cur_bar:
      return e, escore_notes.index(e)

###################################################################################

def align_escore_notes_to_bars(escore_notes,
                               bar_time=4000,
                               trim_durations=False,
                               split_durations=False,
                               even_timings=False
                               ):

  #=============================================================================
    
  escore = copy.deepcopy(escore_notes)
  
  if even_timings:
      for e in escore:
          if e[1] % 2 != 0:
              e[1] += 1
          
          if e[2] % 2 != 0:
              e[2] += 1

  aligned_escore_notes = copy.deepcopy(escore)

  abs_time = 0
  nidx = 0
  delta = 0
  bcount = 0
  next_bar = [0]

  #=============================================================================

  while next_bar:

    next_bar = find_next_bar(escore, bar_time, nidx, bcount)

    if next_bar:
      gescore_notes = escore[nidx:next_bar[1]]
    
    else:
      gescore_notes = escore[nidx:]

    original_timings = [delta] + [(b[1]-a[1]) for a, b in zip(gescore_notes[:-1], gescore_notes[1:])]
    adj_timings = adjust_numbers_to_sum(original_timings, bar_time)

    for t in adj_timings:

      abs_time += t

      aligned_escore_notes[nidx][1] = abs_time
      aligned_escore_notes[nidx][2] -= int(bar_time // 200)

      nidx += 1

    if next_bar:
      delta = escore[next_bar[1]][1]-escore[next_bar[1]-1][1]
      
    bcount += 1

  #=============================================================================

  aligned_adjusted_escore_notes = []
  bcount = 0

  for a in aligned_escore_notes:
    bcount = a[1] // bar_time
    nbtime = bar_time * (bcount+1)

    if a[1]+a[2] > nbtime and a[3] != 9:
      if trim_durations or split_durations:
        ddiff = ((a[1]+a[2])-nbtime)
        aa = copy.deepcopy(a)
        aa[2] = a[2] - ddiff
        aligned_adjusted_escore_notes.append(aa)

        if split_durations:
          aaa = copy.deepcopy(a)
          aaa[1] = a[1]+aa[2]
          aaa[2] = ddiff

          aligned_adjusted_escore_notes.append(aaa)

      else:
        aligned_adjusted_escore_notes.append(a)

    else:
      aligned_adjusted_escore_notes.append(a)

  #=============================================================================

  return aligned_adjusted_escore_notes

###################################################################################

def normalize_chord_durations(chord, 
                              dur_idx=2, 
                              norm_factor=100
                              ):

  nchord = copy.deepcopy(chord)
  
  for c in nchord:
    c[dur_idx] = int(round(max(1 / norm_factor, c[dur_idx] // norm_factor) * norm_factor))

  return nchord

###################################################################################

def normalize_chordified_score_durations(chordified_score, 
                                         dur_idx=2, 
                                         norm_factor=100
                                         ):

  ncscore = copy.deepcopy(chordified_score)
  
  for cc in ncscore:
    for c in cc:
      c[dur_idx] = int(round(max(1 / norm_factor, c[dur_idx] // norm_factor) * norm_factor))

  return ncscore

###################################################################################

def horizontal_ordered_list_search(list_of_lists, 
                                    query_list, 
                                    start_idx=0,
                                    end_idx=-1
                                    ):

  lol = list_of_lists

  results = []

  if start_idx > 0:
    lol = list_of_lists[start_idx:]

  if start_idx == -1:
    idx = -1
    for i, l in enumerate(list_of_lists):
      try:
        idx = l.index(query_list[0])
        lol = list_of_lists[i:]
        break
      except:
        continue

    if idx == -1:
      results.append(-1)
      return results
    else:
      results.append(i)

  if end_idx != -1:
    lol = list_of_lists[start_idx:start_idx+max(end_idx, len(query_list))]

  for i, q in enumerate(query_list):
    try:
      idx = lol[i].index(q)
      results.append(idx)
    except:
      results.append(-1)
      return results

  return results

###################################################################################

def escore_notes_to_escore_matrix(escore_notes,
                                  alt_velocities=False,
                                  flip_matrix=False,
                                  reverse_matrix=False
                                  ):

  last_time = escore_notes[-1][1]
  last_notes = [e for e in escore_notes if e[1] == last_time]
  max_last_dur = max([e[2] for e in last_notes])

  time_range = last_time+max_last_dur

  channels_list = sorted(set([e[3] for e in escore_notes]))

  escore_matrixes = []

  for cha in channels_list:

    escore_matrix = [[[-1, -1]] * 128 for _ in range(time_range)]

    pe = escore_notes[0]

    for i, note in enumerate(escore_notes):

        etype, time, duration, channel, pitch, velocity, patch = note

        time = max(0, time)
        duration = max(1, duration)
        channel = max(0, min(15, channel))
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        patch = max(0, min(128, patch))

        if alt_velocities:
            velocity -= (i % 2)

        if channel == cha:

          for t in range(time, min(time + duration, time_range)):

            escore_matrix[t][pitch] = [velocity, patch]

        pe = note

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    escore_matrixes.append(escore_matrix)

  return [channels_list, escore_matrixes]

###################################################################################

def escore_matrix_to_merged_escore_notes(full_escore_matrix,
                                        max_note_duration=4000
                                        ):

  merged_escore_notes = []

  mat_channels_list = full_escore_matrix[0]
  
  for m, cha in enumerate(mat_channels_list):

    escore_matrix = full_escore_matrix[1][m]

    result = []

    for j in range(len(escore_matrix[0])):

        count = 1

        for i in range(1, len(escore_matrix)):

          if escore_matrix[i][j] != [-1, -1] and escore_matrix[i][j][1] == escore_matrix[i-1][j][1] and count < max_note_duration:
              count += 1

          else:
              if count > 1:  
                result.append([i-count, count, j, escore_matrix[i-1][j]])

              count = 1

        if count > 1:
            result.append([len(escore_matrix)-count, count, j, escore_matrix[-1][j]])

    result.sort(key=lambda x: (x[0], -x[2]))

    for r in result:
      merged_escore_notes.append(['note', r[0], r[1], cha, r[2], r[3][0], r[3][1]])

  return sorted(merged_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

###################################################################################

def escore_matrix_to_original_escore_notes(full_escore_matrix):

  merged_escore_notes = []

  mat_channels_list = full_escore_matrix[0]

  for m, cha in enumerate(mat_channels_list):

    escore_matrix = full_escore_matrix[1][m]

    result = []

    for j in range(len(escore_matrix[0])):

        count = 1

        for i in range(1, len(escore_matrix)):

          if escore_matrix[i][j] != [-1, -1] and escore_matrix[i][j] == escore_matrix[i-1][j]:
              count += 1

          else:
              if count > 1:
                result.append([i-count, count, j, escore_matrix[i-1][j]])

              count = 1

        if count > 1:
            result.append([len(escore_matrix)-count, count, j, escore_matrix[-1][j]])

    result.sort(key=lambda x: (x[0], -x[2]))

    for r in result:
      merged_escore_notes.append(['note', r[0], r[1], cha, r[2], r[3][0], r[3][1]])

  return sorted(merged_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

###################################################################################

def escore_notes_to_binary_matrix(escore_notes, 
                                  channel=0, 
                                  patch=0,
                                  flip_matrix=False,
                                  reverse_matrix=False,
                                  encode_velocities=False
                                  ):

  escore = [e for e in escore_notes if e[3] == channel and e[6] == patch]

  if escore:
    last_time = escore[-1][1]
    last_notes = [e for e in escore if e[1] == last_time]
    max_last_dur = max([e[2] for e in last_notes])

    time_range = last_time+max_last_dur

    escore_matrix = []

    escore_matrix = [[0] * 128 for _ in range(time_range)]

    for note in escore:

        etype, time, duration, chan, pitch, velocity, pat = note

        time = max(0, time)
        duration = max(1, duration)
        chan = max(0, min(15, chan))
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))
        pat = max(0, min(128, pat))

        if channel == chan and patch == pat:

          for t in range(time, min(time + duration, time_range)):
            if encode_velocities:
                escore_matrix[t][pitch] = velocity
                
            else:
                escore_matrix[t][pitch] = 1

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    return escore_matrix

  else:
    return None

###################################################################################

def binary_matrix_to_original_escore_notes(binary_matrix, 
                                           channel=0, 
                                           patch=0, 
                                           velocity=-1,
                                           decode_velocities=False
                                           ):

  result = []

  for j in range(len(binary_matrix[0])):

      count = 1

      for i in range(1, len(binary_matrix)):

        if binary_matrix[i][j] != 0 and binary_matrix[i][j] == binary_matrix[i-1][j]:
            count += 1

        else:
          if count > 1:
            result.append([i-count, count, j, binary_matrix[i-1][j]])
          
          else:
            if binary_matrix[i-1][j] != 0:
              result.append([i-count, count, j, binary_matrix[i-1][j]])

          count = 1

      if count > 1:
          result.append([len(binary_matrix)-count, count, j, binary_matrix[-1][j]])
      
      else:
        if binary_matrix[i-1][j] != 0:
          result.append([i-count, count, j, binary_matrix[i-1][j]])

  result.sort(key=lambda x: (x[0], -x[2]))

  original_escore_notes = []

  vel = velocity

  for r in result:
    
    if velocity == -1 and not decode_velocities:
        vel = max(40, r[2])
        
    if decode_velocities:
        vel = r[3]

    original_escore_notes.append(['note', r[0], r[1], channel, r[2], vel, patch])

  return sorted(original_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

###################################################################################

def escore_notes_averages(escore_notes, 
                          times_index=1, 
                          durs_index=2,
                          chans_index=3, 
                          ptcs_index=4, 
                          vels_index=5,
                          average_drums=False,
                          score_is_delta=False,
                          return_ptcs_and_vels=False
                          ):
  
  if score_is_delta:
    if average_drums:
      times = [e[times_index] for e in escore_notes if e[times_index] != 0]
    else:
      times = [e[times_index] for e in escore_notes if e[times_index] != 0 and e[chans_index] != 9]

  else:
    descore_notes = delta_score_notes(escore_notes)
    if average_drums:
      times = [e[times_index] for e in descore_notes if e[times_index] != 0]
    else:
      times = [e[times_index] for e in descore_notes if e[times_index] != 0 and e[chans_index] != 9]
      
  if average_drums:
    durs = [e[durs_index] for e in escore_notes]
  else:
    durs = [e[durs_index] for e in escore_notes if e[chans_index] != 9]

  if len(times) == 0:
      times = [0]
      
  if len(durs) == 0:
      durs = [0]

  if return_ptcs_and_vels:
    if average_drums:
      ptcs = [e[ptcs_index] for e in escore_notes]
      vels = [e[vels_index] for e in escore_notes]
    else:
      ptcs = [e[ptcs_index] for e in escore_notes if e[chans_index] != 9]
      vels = [e[vels_index] for e in escore_notes if e[chans_index] != 9]
      
    if len(ptcs) == 0:
        ptcs = [0]
      
    if len(vels) == 0:
        vels = [0]

    return [sum(times) / len(times), sum(durs) / len(durs), sum(ptcs) / len(ptcs), sum(vels) / len(vels)]
  
  else:
    return [sum(times) / len(times), sum(durs) / len(durs)]

###################################################################################

def adjust_escore_notes_timings(escore_notes, 
                                adj_k=1, 
                                times_index=1, 
                                durs_index=2, 
                                score_is_delta=False, 
                                return_delta_scpre=False
                                ):

  if score_is_delta:
    adj_escore_notes = copy.deepcopy(escore_notes)
  else:
    adj_escore_notes = delta_score_notes(escore_notes)

  for e in adj_escore_notes:

    if e[times_index] != 0:
      e[times_index] = max(1, round(e[times_index] * adj_k))

    e[durs_index] = max(1, round(e[durs_index] * adj_k))

  if return_delta_scpre:
    return adj_escore_notes

  else:
    return delta_score_to_abs_score(adj_escore_notes)

###################################################################################

def escore_notes_delta_times(escore_notes,
                             times_index=1
                             ):

  descore_notes = delta_score_notes(escore_notes)

  return [e[times_index] for e in descore_notes]

###################################################################################

def escore_notes_durations(escore_notes,
                            durs_index=1
                            ):

  descore_notes = delta_score_notes(escore_notes)

  return [e[durs_index] for e in descore_notes]

###################################################################################

def ordered_lists_match_ratio(src_list, trg_list):

  zlist = list(zip(src_list, trg_list))

  return sum([a == b for a, b in zlist]) / len(list(zlist))

###################################################################################

def lists_intersections(src_list, trg_list):
  return list(set(src_list) & set(trg_list))

###################################################################################

def transpose_escore_notes(escore_notes, 
                            transpose_value=0, 
                            channel_index=3, 
                            pitches_index=4
                            ):

  tr_escore_notes = copy.deepcopy(escore_notes)

  for e in tr_escore_notes:
    if e[channel_index] != 9:
      e[pitches_index] = max(1, min(127, e[pitches_index] + transpose_value))

  return tr_escore_notes

###################################################################################

def transpose_escore_notes_to_pitch(escore_notes, 
                                    target_pitch_value=60, 
                                    channel_index=3, 
                                    pitches_index=4
                                    ):

  tr_escore_notes = copy.deepcopy(escore_notes)

  transpose_delta = int(round(target_pitch_value)) - int(round(escore_notes_averages(escore_notes, return_ptcs_and_vels=True)[2]))

  for e in tr_escore_notes:
    if e[channel_index] != 9:
      e[pitches_index] = max(1, min(127, e[pitches_index] + transpose_delta))

  return tr_escore_notes

###################################################################################

CHORDS_TYPES = ['WHITE', 'BLACK', 'UNKNOWN', 'MIXED WHITE', 'MIXED BLACK', 'MIXED GRAY']

###################################################################################

def tones_chord_type(tones_chord, 
                     return_chord_type_index=True,
                     use_filtered_chords=False,
                     use_full_chords=True
                     ):

  WN = WHITE_NOTES
  BN = BLACK_NOTES
  MX = WHITE_NOTES + BLACK_NOTES

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(tones_chord)

  ctype = 'UNKNOWN'

  if tones_chord in CHORDS:

    if sorted(set(tones_chord) & set(WN)) == tones_chord:
      ctype = 'WHITE'

    elif sorted(set(tones_chord) & set(BN)) == tones_chord:
      ctype = 'BLACK'

    if len(tones_chord) > 1 and sorted(set(tones_chord) & set(MX)) == tones_chord:

      if len(sorted(set(tones_chord) & set(WN))) == len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED GRAY'

      elif len(sorted(set(tones_chord) & set(WN))) > len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED WHITE'

      elif len(sorted(set(tones_chord) & set(WN))) < len(sorted(set(tones_chord) & set(BN))):
        ctype = 'MIXED BLACK'

  if return_chord_type_index:
    return CHORDS_TYPES.index(ctype)

  else:
    return ctype

###################################################################################

def tone_type(tone, 
              return_tone_type_index=True
              ):

  tone = tone % 12

  if tone in BLACK_NOTES:
    if return_tone_type_index:
      return CHORDS_TYPES.index('BLACK')
    else:
      return "BLACK"

  else:
    if return_tone_type_index:
      return CHORDS_TYPES.index('WHITE')
    else:
      return "WHITE"

###################################################################################

def lists_sym_differences(src_list, trg_list):
  return list(set(src_list) ^ set(trg_list))

###################################################################################

def lists_differences(long_list, short_list):
  return list(set(long_list) - set(short_list))

###################################################################################

def find_best_tones_chord(src_tones_chords,
                          trg_tones_chords,
                          find_longest=True
                          ):

  not_seen_trg_chords = []

  max_len = 0

  for tc in trg_tones_chords:
    if sorted(tc) in src_tones_chords:
      not_seen_trg_chords.append(sorted(tc))
      max_len = max(max_len, len(tc))

  if not not_seen_trg_chords:
    max_len = len(max(trg_tones_chords, key=len))
    not_seen_trg_chords = trg_tones_chords

  if find_longest:
    return random.choice([c for c in not_seen_trg_chords if len(c) == max_len])

  else:
    return random.choice(not_seen_trg_chords)

###################################################################################

def find_matching_tones_chords(tones_chord,
                               matching_chord_length=-1,
                               match_chord_type=True,
                               use_filtered_chords=True,
                               use_full_chords=True
                               ):

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  tones_chord = sorted(tones_chord)

  tclen = len(tones_chord)

  tctype = tones_chord_type(tones_chord, use_filtered_chords=use_filtered_chords)

  matches = []

  for tc in CHORDS:

    if matching_chord_length == -1:
      if len(tc) > tclen:
        if sorted(lists_intersections(tc, tones_chord)) == tones_chord:
          if match_chord_type:
            if tones_chord_type(tc, use_filtered_chords=use_filtered_chords) == tctype:
              tcdiffs = lists_differences(tc, tones_chord)
              if all(tone_type(d) == tctype % 3 for d in tcdiffs):
                matches.append(tc)
          else:
            matches.append(tc)

    else:

      if len(tc) == max(tclen, matching_chord_length):
        if sorted(lists_intersections(tc, tones_chord)) == tones_chord:
          if match_chord_type:
            if tones_chord_type(tc, use_filtered_chords=use_filtered_chords) == tctype:
              tcdiffs = lists_differences(tc, tones_chord)
              if all(tone_type(d) == tctype % 3 for d in tcdiffs):
                matches.append(tc)
          else:
            matches.append(tc)

  return sorted(matches, key=len)

###################################################################################

def adjust_list_of_values_to_target_average(list_of_values, 
                                            trg_avg, 
                                            min_value, 
                                            max_value
                                            ):

    filtered_values = [value for value in list_of_values if min_value <= value <= max_value]

    if not filtered_values:
        return list_of_values

    current_avg = sum(filtered_values) / len(filtered_values)
    scale_factor = trg_avg / current_avg

    adjusted_values = [value * scale_factor for value in filtered_values]

    total_difference = trg_avg * len(filtered_values) - sum(adjusted_values)
    adjustment_per_value = total_difference / len(filtered_values)

    final_values = [value + adjustment_per_value for value in adjusted_values]

    while abs(sum(final_values) / len(final_values) - trg_avg) > 1e-6:
        total_difference = trg_avg * len(final_values) - sum(final_values)
        adjustment_per_value = total_difference / len(final_values)
        final_values = [value + adjustment_per_value for value in final_values]

    final_values = [round(value) for value in final_values]

    adjusted_values = copy.deepcopy(list_of_values)

    j = 0

    for i in range(len(adjusted_values)):
        if min_value <= adjusted_values[i] <= max_value:
            adjusted_values[i] = final_values[j]
            j += 1

    return adjusted_values

###################################################################################

def adjust_escore_notes_to_average(escore_notes,
                                   trg_avg,
                                   min_value=1,
                                   max_value=4000,
                                   times_index=1,
                                   durs_index=2,
                                   score_is_delta=False,
                                   return_delta_scpre=False
                                   ):
    if score_is_delta:
      delta_escore_notes = copy.deepcopy(escore_notes)

    else:
      delta_escore_notes = delta_score_notes(escore_notes)

    times = [[e[times_index], e[durs_index]] for e in delta_escore_notes]

    filtered_values = [value for value in times if min_value <= value[0] <= max_value]

    if not filtered_values:
        return escore_notes

    current_avg = sum([v[0] for v in filtered_values]) / len([v[0] for v in filtered_values])
    scale_factor = trg_avg / current_avg

    adjusted_values = [[value[0] * scale_factor, value[1] * scale_factor] for value in filtered_values]

    total_difference = trg_avg * len([v[0] for v in filtered_values]) - sum([v[0] for v in adjusted_values])
    adjustment_per_value = total_difference / len(filtered_values)

    final_values = [[value[0] + adjustment_per_value, value[1] + adjustment_per_value] for value in adjusted_values]

    while abs(sum([v[0] for v in final_values]) / len(final_values) - trg_avg) > 1e-6:
        total_difference = trg_avg * len(final_values) - sum([v[0] for v in final_values])
        adjustment_per_value = total_difference / len(final_values)
        final_values = [[value[0] + adjustment_per_value, value[1] + adjustment_per_value] for value in final_values]

    final_values = [[round(value[0]), round(value[1])] for value in final_values]

    adjusted_delta_score = copy.deepcopy(delta_escore_notes)

    j = 0

    for i in range(len(adjusted_delta_score)):
        if min_value <= adjusted_delta_score[i][1] <= max_value:
            adjusted_delta_score[i][times_index] = final_values[j][0]
            adjusted_delta_score[i][durs_index] = final_values[j][1]
            j += 1

    adjusted_escore_notes = delta_score_to_abs_score(adjusted_delta_score)

    if return_delta_scpre:
      return adjusted_delta_score

    else:
      return adjusted_escore_notes

###################################################################################

def harmonize_enhanced_melody_score_notes_to_ms_SONG(escore_notes,
                                                      melody_velocity=-1,
                                                      melody_channel=3,
                                                      melody_patch=40,
                                                      melody_base_octave=4,
                                                      harmonized_tones_chords_velocity=-1,
                                                      harmonized_tones_chords_channel=0,
                                                      harmonized_tones_chords_patch=0
                                                    ):

  harmonized_tones_chords = harmonize_enhanced_melody_score_notes(escore_notes)

  harm_escore_notes = []

  time = 0

  for i, note in enumerate(escore_notes):

    time = note[1]
    dur = note[2]
    ptc = note[4]

    if melody_velocity == -1:
      vel = int(110 + ((ptc % 12) * 1.5))
    else:
      vel = melody_velocity

    harm_escore_notes.append(['note', time, dur, melody_channel, ptc, vel, melody_patch])

    for t in harmonized_tones_chords[i]:

      ptc = (melody_base_octave * 12) + t

      if harmonized_tones_chords_velocity == -1:
        vel = int(80 + ((ptc % 12) * 1.5))
      else:
        vel = harmonized_tones_chords_velocity

      harm_escore_notes.append(['note', time, dur, harmonized_tones_chords_channel, ptc, vel, harmonized_tones_chords_patch])

  return sorted(harm_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

###################################################################################

def check_and_fix_pitches_chord(pitches_chord,
                                remove_duplicate_pitches=True,
                                use_filtered_chords=False,
                                use_full_chords=True,
                                fix_bad_pitches=False,
                                ):
  
  if remove_duplicate_pitches:
    pitches_chord = sorted(set(pitches_chord), reverse=True)
  else:
    pitches_chord = sorted(pitches_chord, reverse=True)

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  chord = copy.deepcopy(pitches_chord)
    
  tones_chord = sorted(set([t % 12 for t in chord]))

  if tones_chord:

      if tones_chord not in CHORDS:
        
        if len(tones_chord) == 2:
          tones_counts = Counter([p % 12 for p in pitches_chord]).most_common()

          if tones_counts[0][1] > 1:
            tones_chord = [tones_counts[0][0]]
          
          elif tones_counts[1][1] > 1:
            tones_chord = [tones_counts[1][0]]
          
          else:
            tones_chord = [pitches_chord[0] % 12]

        else:
          tones_chord_combs = [list(comb) for i in range(len(tones_chord)-1, 0, -1) for comb in combinations(tones_chord, i)]

          for co in tones_chord_combs:
            if co in CHORDS:
              tones_chord = co
              break

          if len(tones_chord) == 1:
            tones_chord = [pitches_chord[0] % 12]
              
  chord.sort(reverse=True)

  new_chord = set()
  pipa = []

  for e in chord:
    if e % 12 in tones_chord:
      new_chord.add(tuple([e]))
      pipa.append(e)

    elif (e+1) % 12 in tones_chord:
      e += 1
      new_chord.add(tuple([e]))
      pipa.append(e)

    elif (e-1) % 12 in tones_chord:
      e -= 1
      new_chord.add(tuple([e]))
      pipa.append(e)

  if fix_bad_pitches:

    bad_chord = set()

    for e in chord:
    
      if e % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
      
      elif (e+1) % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
      
      elif (e-1) % 12 not in tones_chord:
        bad_chord.add(tuple([e]))
          
    for bc in bad_chord:

      bc = list(bc)

      tone = find_closest_tone(tones_chord, bc[0] % 12)

      new_pitch = ((bc[0] // 12) * 12) + tone

      if new_pitch not in pipa:
        new_chord.add(tuple([new_pitch]))
        pipa.append(new_pitch)

  new_pitches_chord = [e[0] for e in new_chord]

  return sorted(new_pitches_chord, reverse=True)

###################################################################################

ALL_CHORDS_TRANS = [[0], [0, 4], [0, 4, 7], [0, 4, 8], [0, 5], [0, 6], [0, 7], [0, 8], [1], [1, 5],
                    [1, 5, 9], [1, 6], [1, 7], [1, 8], [1, 9], [2], [2, 6], [2, 6, 10], [2, 7],
                    [2, 8], [2, 9], [2, 10], [3], [3, 7], [3, 7, 11], [3, 8], [3, 9], [3, 10],
                    [3, 11], [4], [4, 7], [4, 7, 11], [4, 8], [4, 9], [4, 10], [4, 11], [5],
                    [5, 9], [5, 10], [5, 11], [6], [6, 10], [6, 11], [7], [7, 11], [8], [9], [10],
                    [11]]

###################################################################################

def minkowski_distance(x, y, p=3, pad_value=float('inf')):

    if len(x) != len(y):
      return -1
    
    distance = 0
    
    for i in range(len(x)):

        if x[i] == pad_value or y[i] == pad_value:
          continue

        distance += abs(x[i] - y[i]) ** p

    return distance ** (1 / p)

###################################################################################

def dot_product(x, y, pad_value=None):
    return sum(xi * yi for xi, yi in zip(x, y) if xi != pad_value and yi != pad_value)

def norm(vector, pad_value=None):
    return sum(xi ** 2 for xi in vector if xi != pad_value) ** 0.5

def cosine_similarity(x, y, pad_value=None):
    if len(x) != len(y):
        return -1
    
    dot_prod = dot_product(x, y, pad_value)
    norm_x = norm(x, pad_value)
    norm_y = norm(y, pad_value)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return dot_prod / (norm_x * norm_y)

###################################################################################

def hamming_distance(arr1, arr2, pad_value):
    return sum(el1 != el2 for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value)

###################################################################################

def jaccard_similarity(arr1, arr2, pad_value):
    intersection = sum(el1 and el2 for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value)
    union = sum((el1 or el2) for el1, el2 in zip(arr1, arr2) if el1 != pad_value or el2 != pad_value)
    return intersection / union if union != 0 else 0

###################################################################################

def pearson_correlation(arr1, arr2, pad_value):
    filtered_pairs = [(el1, el2) for el1, el2 in zip(arr1, arr2) if el1 != pad_value and el2 != pad_value]
    if not filtered_pairs:
        return 0
    n = len(filtered_pairs)
    sum1 = sum(el1 for el1, el2 in filtered_pairs)
    sum2 = sum(el2 for el1, el2 in filtered_pairs)
    sum1_sq = sum(el1 ** 2 for el1, el2 in filtered_pairs)
    sum2_sq = sum(el2 ** 2 for el1, el2 in filtered_pairs)
    p_sum = sum(el1 * el2 for el1, el2 in filtered_pairs)
    num = p_sum - (sum1 * sum2 / n)
    den = ((sum1_sq - sum1 ** 2 / n) * (sum2_sq - sum2 ** 2 / n)) ** 0.5
    if den == 0:
        return 0
    return num / den

###################################################################################

def calculate_combined_distances(array_of_arrays,
                                  combine_hamming_distance=True,
                                  combine_jaccard_similarity=True, 
                                  combine_pearson_correlation=True,
                                  pad_value=None
                                  ):

  binary_arrays = array_of_arrays
  binary_array_len = len(binary_arrays)

  hamming_distances = [[0] * binary_array_len for _ in range(binary_array_len)]
  jaccard_similarities = [[0] * binary_array_len for _ in range(binary_array_len)]
  pearson_correlations = [[0] * binary_array_len for _ in range(binary_array_len)]

  for i in range(binary_array_len):
      for j in range(i + 1, binary_array_len):
          hamming_distances[i][j] = hamming_distance(binary_arrays[i], binary_arrays[j], pad_value)
          hamming_distances[j][i] = hamming_distances[i][j]
          
          jaccard_similarities[i][j] = jaccard_similarity(binary_arrays[i], binary_arrays[j], pad_value)
          jaccard_similarities[j][i] = jaccard_similarities[i][j]
          
          pearson_correlations[i][j] = pearson_correlation(binary_arrays[i], binary_arrays[j], pad_value)
          pearson_correlations[j][i] = pearson_correlations[i][j]

  max_hamming = max(max(row) for row in hamming_distances)
  min_hamming = min(min(row) for row in hamming_distances)
  normalized_hamming = [[(val - min_hamming) / (max_hamming - min_hamming) for val in row] for row in hamming_distances]

  max_jaccard = max(max(row) for row in jaccard_similarities)
  min_jaccard = min(min(row) for row in jaccard_similarities)
  normalized_jaccard = [[(val - min_jaccard) / (max_jaccard - min_jaccard) for val in row] for row in jaccard_similarities]

  max_pearson = max(max(row) for row in pearson_correlations)
  min_pearson = min(min(row) for row in pearson_correlations)
  normalized_pearson = [[(val - min_pearson) / (max_pearson - min_pearson) for val in row] for row in pearson_correlations]

  selected_metrics = 0

  if combine_hamming_distance:
    selected_metrics += normalized_hamming[i][j]
  
  if combine_jaccard_similarity:
    selected_metrics += (1 - normalized_jaccard[i][j])

  if combine_pearson_correlation:
    selected_metrics += (1 - normalized_pearson[i][j])

  combined_metric = [[selected_metrics for i in range(binary_array_len)] for j in range(binary_array_len)]

  return combined_metric

###################################################################################

def tones_chords_to_bits(tones_chords):

  bits_tones_chords = []

  for c in tones_chords:

    c.sort()

    bits = tones_chord_to_bits(c)

    bits_tones_chords.append(bits)

  return bits_tones_chords

###################################################################################

def tones_chords_to_ints(tones_chords):

  ints_tones_chords = []

  for c in tones_chords:

    c.sort()

    bits = tones_chord_to_bits(c)

    number = bits_to_int(bits)

    ints_tones_chords.append(number)

  return ints_tones_chords

###################################################################################

def tones_chords_to_types(tones_chords, 
                          return_chord_type_index=False
                          ):

  types_tones_chords = []

  for c in tones_chords:

    c.sort()

    ctype = tones_chord_type(c, return_chord_type_index=return_chord_type_index)

    types_tones_chords.append(ctype)

  return types_tones_chords

###################################################################################

def morph_tones_chord(tones_chord, 
                      trg_tone, 
                      use_filtered_chords=True,
                      use_full_chords=True
                      ):

  src_tones_chord = sorted(sorted(set(tones_chord)) + [trg_tone])

  combs = [list(comb) for i in range(len(src_tones_chord), 0, -1) for comb in combinations(src_tones_chord, i) if trg_tone in list(comb)]

  matches = []

  if use_filtered_chords:
    CHORDS = ALL_CHORDS_FILTERED
  
  else:
    CHORDS = ALL_CHORDS_SORTED

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  for c in combs:
    if sorted(set(c)) in CHORDS:
      matches.append(sorted(set(c)))

  max_len = len(max(matches, key=len))

  return random.choice([m for m in matches if len(m) == max_len])

###################################################################################

def compress_binary_matrix(binary_matrix, 
                           only_compress_zeros=False,
                           return_compression_ratio=False
                           ):

  compressed_bmatrix = []

  zm = [0] * len(binary_matrix[0])
  pm = [0] * len(binary_matrix[0])

  mcount = 0

  for m in binary_matrix:
    
    if only_compress_zeros:
      if m != zm:
        compressed_bmatrix.append(m)
        mcount += 1
    
    else:
      if m != pm:
        compressed_bmatrix.append(m)
        mcount += 1
    
    pm = m

  if return_compression_ratio:
    return [compressed_bmatrix, mcount / len(binary_matrix)]

  else:
    return compressed_bmatrix

###################################################################################

def solo_piano_escore_notes(escore_notes,
                            channels_index=3,
                            pitches_index=4,
                            patches_index=6,
                            keep_drums=False,
                            ):

  cscore = chordify_score([1000, copy.deepcopy(escore_notes)])

  sp_escore_notes = []

  for c in cscore:

    seen = []
    chord = []

    for cc in c:

      if cc[channels_index] != 9:
        if cc[pitches_index] not in seen:
            
            cc[channels_index] = 0
            cc[patches_index] = 0
        
            chord.append(cc)
            seen.append(cc[pitches_index])
      
      else:
        if keep_drums:
          if cc[pitches_index]+128 not in seen:
              chord.append(cc)
              seen.append(cc[pitches_index]+128)

    sp_escore_notes.append(chord)

  return flatten(sp_escore_notes)

###################################################################################

def strip_drums_from_escore_notes(escore_notes, 
                                  channels_index=3
                                  ):
  
  return [e for e in escore_notes if e[channels_index] != 9]

###################################################################################

def fixed_escore_notes_timings(escore_notes,
                               fixed_durations=False,
                               fixed_timings_multiplier=1,
                               custom_fixed_time=-1,
                               custom_fixed_dur=-1
                               ):

  fixed_timings_escore_notes = delta_score_notes(escore_notes, even_timings=True)

  mode_time = round(Counter([e[1] for e in fixed_timings_escore_notes if e[1] != 0]).most_common()[0][0] * fixed_timings_multiplier)

  if mode_time % 2 != 0:
    mode_time += 1

  mode_dur = round(Counter([e[2] for e in fixed_timings_escore_notes if e[2] != 0]).most_common()[0][0] * fixed_timings_multiplier)

  if mode_dur % 2 != 0:
    mode_dur += 1

  for e in fixed_timings_escore_notes:
    if e[1] != 0:
      
      if custom_fixed_time > 0:
        e[1] = custom_fixed_time
      
      else:
        e[1] = mode_time

    if fixed_durations:
      
      if custom_fixed_dur > 0:
        e[2] = custom_fixed_dur
      
      else:
        e[2] = mode_dur

  return delta_score_to_abs_score(fixed_timings_escore_notes)

###################################################################################

def cubic_kernel(x):
    abs_x = abs(x)
    if abs_x <= 1:
        return 1.5 * abs_x**3 - 2.5 * abs_x**2 + 1
    elif abs_x <= 2:
        return -0.5 * abs_x**3 + 2.5 * abs_x**2 - 4 * abs_x + 2
    else:
        return 0

###################################################################################

def resize_matrix(matrix, new_height, new_width):
    old_height = len(matrix)
    old_width = len(matrix[0])
    resized_matrix = [[0] * new_width for _ in range(new_height)]
    
    for i in range(new_height):
        for j in range(new_width):
            old_i = i * old_height / new_height
            old_j = j * old_width / new_width
            
            value = 0
            total_weight = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    i_m = min(max(int(old_i) + m, 0), old_height - 1)
                    j_n = min(max(int(old_j) + n, 0), old_width - 1)
                    
                    if matrix[i_m][j_n] == 0:
                        continue
                    
                    weight = cubic_kernel(old_i - i_m) * cubic_kernel(old_j - j_n)
                    value += matrix[i_m][j_n] * weight
                    total_weight += weight
            
            if total_weight > 0:
                value /= total_weight
            
            resized_matrix[i][j] = int(value > 0.5)
    
    return resized_matrix

###################################################################################

def square_binary_matrix(binary_matrix, 
                         matrix_size=128,
                         use_fast_squaring=False,
                         return_plot_points=False
                         ):

  if use_fast_squaring:

    step = round(len(binary_matrix) / matrix_size)

    samples = []

    for i in range(0, len(binary_matrix), step):
      samples.append(tuple([tuple(d) for d in binary_matrix[i:i+step]]))

    resized_matrix = []

    zmatrix = [[0] * matrix_size]

    for s in samples:

      samples_counts = Counter(s).most_common()

      best_sample = tuple([0] * matrix_size)
      pm = tuple(zmatrix[0])

      for sc in samples_counts:
        if sc[0] != tuple(zmatrix[0]) and sc[0] != pm:
          best_sample = sc[0]
          pm = sc[0]
          break
        
        pm = sc[0]

      resized_matrix.append(list(best_sample))

    resized_matrix = resized_matrix[:matrix_size]
    resized_matrix += zmatrix * (matrix_size - len(resized_matrix))
    
  else:
    resized_matrix = resize_matrix(binary_matrix, matrix_size, matrix_size)

  points = [(i, j) for i in range(matrix_size) for j in range(matrix_size) if resized_matrix[i][j] == 1]

  if return_plot_points:
    return [resized_matrix, points]

  else:
    return resized_matrix

###################################################################################

def mean(matrix):
    return sum(sum(row) for row in matrix) / (len(matrix) * len(matrix[0]))

###################################################################################

def variance(matrix, mean_value):
    return sum(sum((element - mean_value) ** 2 for element in row) for row in matrix) / (len(matrix) * len(matrix[0]))
    
###################################################################################

def covariance(matrix1, matrix2, mean1, mean2):
    return sum(sum((matrix1[i][j] - mean1) * (matrix2[i][j] - mean2) for j in range(len(matrix1[0]))) for i in range(len(matrix1))) / (len(matrix1) * len(matrix1[0]))

###################################################################################

def ssim_index(matrix1, matrix2, bit_depth=1):

    if len(matrix1) != len(matrix2) and len(matrix1[0]) != len(matrix2[0]):
      return -1

    K1, K2 = 0.01, 0.03
    L = bit_depth
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu1 = mean(matrix1)
    mu2 = mean(matrix2)
    
    sigma1_sq = variance(matrix1, mu1)
    sigma2_sq = variance(matrix2, mu2)
    
    sigma12 = covariance(matrix1, matrix2, mu1, mu2)
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim

###################################################################################

def find_most_similar_matrix(array_of_matrices, 
                             trg_matrix,
                             matrices_bit_depth=1,
                             return_most_similar_index=False
                             ):
   
    max_ssim = -float('inf')
    most_similar_index = -1

    for i, matrix in enumerate(array_of_matrices):

        ssim = ssim_index(matrix, trg_matrix, bit_depth=matrices_bit_depth)
        
        if ssim > max_ssim:
            max_ssim = ssim
            most_similar_index = i
    
    if return_most_similar_index:
      return most_similar_index
    
    else:
      return array_of_matrices[most_similar_index]

###################################################################################

def chord_to_pchord(chord):

  pchord = []

  for cc in chord:
    if cc[3] != 9:
      pchord.append(cc[4])

  return pchord

###################################################################################

def summarize_escore_notes(escore_notes, 
                           summary_length_in_chords=128, 
                           preserve_timings=True,
                           preserve_durations=False,
                           time_threshold=12,
                           min_sum_chord_len=2,
                           use_tones_chords=True
                           ):

    cscore = chordify_score([d[1:] for d in delta_score_notes(escore_notes)])

    summary_length_in_chords = min(len(cscore), summary_length_in_chords)

    ltthresh = time_threshold // 2
    uttresh = time_threshold * 2

    mc_time = Counter([c[0][0] for c in cscore if c[0][2] != 9 and ltthresh < c[0][0] < uttresh]).most_common()[0][0]

    pchords = []

    for c in cscore:
      if use_tones_chords:
        pchords.append([c[0][0]] + pitches_to_tones_chord(chord_to_pchord(c)))
        
      else:
        pchords.append([c[0][0]] + chord_to_pchord(c))

    step = round(len(pchords) / summary_length_in_chords)

    samples = []

    for i in range(0, len(pchords), step):
      samples.append(tuple([tuple(d) for d in pchords[i:i+step]]))

    summarized_escore_notes = []

    for i, s in enumerate(samples):

      best_chord = list([v[0] for v in Counter(s).most_common() if v[0][0] == mc_time and len(v[0]) > min_sum_chord_len])

      if not best_chord:
        best_chord = list([v[0] for v in Counter(s).most_common() if len(v[0]) > min_sum_chord_len])
        
        if not best_chord:
          best_chord = list([Counter(s).most_common()[0][0]])

      chord = copy.deepcopy(cscore[[ss for ss in s].index(best_chord[0])+(i*step)])

      if preserve_timings:

        if not preserve_durations:

          if i > 0:

            pchord = summarized_escore_notes[-1]

            for pc in pchord:
              pc[1] = min(pc[1], chord[0][0])

      else:

        chord[0][0] = 1

        for c in chord:
          c[1] = 1  

      summarized_escore_notes.append(chord)

    summarized_escore_notes = summarized_escore_notes[:summary_length_in_chords]

    return [['note'] + d for d in delta_score_to_abs_score(flatten(summarized_escore_notes), times_idx=0)]

###################################################################################

def compress_patches_in_escore_notes(escore_notes,
                                     num_patches=4,
                                     group_patches=False
                                     ):

  if num_patches > 4:
    n_patches = 4
  elif num_patches < 1:
    n_patches = 1
  else:
    n_patches = num_patches

  if group_patches:
    patches_set = sorted(set([e[6] for e in escore_notes]))
    trg_patch_list = []
    seen = []
    for p in patches_set:
      if p // 8 not in seen:
        trg_patch_list.append(p)
        seen.append(p // 8)

    trg_patch_list = sorted(trg_patch_list)

  else:
    trg_patch_list = sorted(set([e[6] for e in escore_notes]))

  if 128 in trg_patch_list and n_patches > 1:
    trg_patch_list = trg_patch_list[:n_patches-1] + [128]
  else:
    trg_patch_list = trg_patch_list[:n_patches]

  new_escore_notes = []

  for e in escore_notes:
    if e[6] in trg_patch_list:
      new_escore_notes.append(e)

  return new_escore_notes

###################################################################################

def compress_patches_in_escore_notes_chords(escore_notes,
                                            max_num_patches_per_chord=4,
                                            group_patches=True,
                                            root_grouped_patches=False
                                            ):

  if max_num_patches_per_chord > 4:
    n_patches = 4
  elif max_num_patches_per_chord < 1:
    n_patches = 1
  else:
    n_patches = max_num_patches_per_chord

  cscore = chordify_score([1000, sorted(escore_notes, key=lambda x: (x[1], x[6]))])

  new_escore_notes = []

  for c in cscore:

    if group_patches:
      patches_set = sorted(set([e[6] for e in c]))
      trg_patch_list = []
      seen = []
      for p in patches_set:
        if p // 8 not in seen:
          trg_patch_list.append(p)
          seen.append(p // 8)

      trg_patch_list = sorted(trg_patch_list)

    else:
      trg_patch_list = sorted(set([e[6] for e in c]))

    if 128 in trg_patch_list and n_patches > 1:
      trg_patch_list = trg_patch_list[:n_patches-1] + [128]
    else:
      trg_patch_list = trg_patch_list[:n_patches]

    for ccc in c:

      cc = copy.deepcopy(ccc)

      if group_patches:
        if cc[6] // 8 in [t // 8 for t in trg_patch_list]:
          if root_grouped_patches:
            cc[6] = (cc[6] // 8) * 8
          new_escore_notes.append(cc)

      else:
        if cc[6] in trg_patch_list:
          new_escore_notes.append(cc)

  return new_escore_notes

###################################################################################

def escore_notes_to_image_matrix(escore_notes,
                                  num_img_channels=3,
                                  filter_out_zero_rows=False,
                                  filter_out_duplicate_rows=False,
                                  flip_matrix=False,
                                  reverse_matrix=False
                                  ):

  escore_notes = sorted(escore_notes, key=lambda x: (x[1], x[6]))

  if num_img_channels > 1:
    n_mat_channels = 3
  else:
    n_mat_channels = 1

  if escore_notes:
    last_time = escore_notes[-1][1]
    last_notes = [e for e in escore_notes if e[1] == last_time]
    max_last_dur = max([e[2] for e in last_notes])

    time_range = last_time+max_last_dur

    escore_matrix = []

    escore_matrix = [[0] * 128 for _ in range(time_range)]

    for note in escore_notes:

        etype, time, duration, chan, pitch, velocity, pat = note

        time = max(0, time)
        duration = max(2, duration)
        chan = max(0, min(15, chan))
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        patch = max(0, min(128, pat))

        if chan != 9:
          pat = patch + 128
        else:
          pat = 127

        seen_pats = []

        for t in range(time, min(time + duration, time_range)):

          mat_value = escore_matrix[t][pitch]

          mat_value_0 = (mat_value // (256 * 256)) % 256
          mat_value_1 = (mat_value // 256) % 256

          cur_num_chans = 0

          if 0 < mat_value < 256 and pat not in seen_pats:
            cur_num_chans = 1
          elif 256 < mat_value < (256 * 256) and pat not in seen_pats:
            cur_num_chans = 2

          if cur_num_chans < n_mat_channels:

            if n_mat_channels == 1:

              escore_matrix[t][pitch] = pat
              seen_pats.append(pat)

            elif n_mat_channels == 3:

              if cur_num_chans == 0:
                escore_matrix[t][pitch] = pat
                seen_pats.append(pat)
              elif cur_num_chans == 1:
                escore_matrix[t][pitch] = (256 * 256 * mat_value_0) + (256 * pat)
                seen_pats.append(pat)
              elif cur_num_chans == 2:
                escore_matrix[t][pitch] = (256 * 256 * mat_value_0) + (256 * mat_value_1) + pat
                seen_pats.append(pat)

    if filter_out_zero_rows:
      escore_matrix = [e for e in escore_matrix if sum(e) != 0]

    if filter_out_duplicate_rows:

      dd_escore_matrix = []

      pr = [-1] * 128
      for e in escore_matrix:
        if e != pr:
          dd_escore_matrix.append(e)
          pr = e
      
      escore_matrix = dd_escore_matrix

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    return escore_matrix

  else:
    return None

###################################################################################

def find_value_power(value, number):
    return math.floor(math.log(value, number))

###################################################################################

def image_matrix_to_original_escore_notes(image_matrix,
                                          velocity=-1
                                          ):

  result = []

  for j in range(len(image_matrix[0])):

      count = 1

      for i in range(1, len(image_matrix)):

        if image_matrix[i][j] != 0 and image_matrix[i][j] == image_matrix[i-1][j]:
            count += 1

        else:
          if count > 1:
            result.append([i-count, count, j, image_matrix[i-1][j]])

          else:
            if image_matrix[i-1][j] != 0:
              result.append([i-count, count, j, image_matrix[i-1][j]])

          count = 1

      if count > 1:
          result.append([len(image_matrix)-count, count, j, image_matrix[-1][j]])

      else:
        if image_matrix[i-1][j] != 0:
          result.append([i-count, count, j, image_matrix[i-1][j]])

  result.sort(key=lambda x: (x[0], -x[2]))

  original_escore_notes = []

  vel = velocity

  for r in result:

    if velocity == -1:
      vel = max(40, r[2])

    ptc0 = 0
    ptc1 = 0
    ptc2 = 0

    if find_value_power(r[3], 256) == 0:
      ptc0 = r[3] % 256

    elif find_value_power(r[3], 256) == 1:
      ptc0 = r[3] // 256
      ptc1 = (r[3] // 256) % 256

    elif find_value_power(r[3], 256) == 2:
      ptc0 = (r[3] // 256) // 256
      ptc1 = (r[3] // 256) % 256
      ptc2 = r[3] % 256

    ptcs = [ptc0, ptc1, ptc2]
    patches = [p for p in ptcs if p != 0]

    for i, p in enumerate(patches):

      if p < 128:
        patch = 128
        channel = 9

      else:
        patch = p % 128
        chan = p // 8

        if chan == 9:
          chan += 1

        channel = min(15, chan)

      original_escore_notes.append(['note', r[0], r[1], channel, r[2], vel, patch])

  output_score = sorted(original_escore_notes, key=lambda x: (x[1], -x[4], x[6]))

  adjust_score_velocities(output_score, 127)

  return output_score

###################################################################################

def escore_notes_delta_times(escore_notes, 
                             timings_index=1, 
                             channels_index=3, 
                             omit_zeros=False, 
                             omit_drums=False
                            ):

  if omit_drums:

    score = [e for e in escore_notes if e[channels_index] != 9]
    dtimes = [score[0][timings_index]] + [b[timings_index]-a[timings_index] for a, b in zip(score[:-1], score[1:])]

  else:
    dtimes = [escore_notes[0][timings_index]] + [b[timings_index]-a[timings_index] for a, b in zip(escore_notes[:-1], escore_notes[1:])]
  
  if omit_zeros:
    dtimes = [d for d in dtimes if d != 0]
  
  return dtimes

###################################################################################

def monophonic_check(escore_notes, times_index=1):
  return len(escore_notes) == len(set([e[times_index] for e in escore_notes]))

###################################################################################

def count_escore_notes_patches(escore_notes, patches_index=6):
  return [list(c) for c in Counter([e[patches_index] for e in escore_notes]).most_common()]

###################################################################################

def escore_notes_medley(list_of_escore_notes, 
                        list_of_labels=None,
                        pause_time_value=255
                        ):

  if list_of_labels is not None:
    labels = [str(l) for l in list_of_labels] + ['No label'] * (len(list_of_escore_notes)-len(list_of_labels))

  medley = []

  time = 0

  for i, m in enumerate(list_of_escore_notes):

    if list_of_labels is not None:
      medley.append(['text_event', time, labels[i]])

    pe = m[0]

    for mm in m:

      time += mm[1] - pe[1]

      mmm = copy.deepcopy(mm)
      mmm[1] = time

      medley.append(mmm)

      pe = mm

    time += pause_time_value

  return medley

###################################################################################

def proportions_counter(list_of_values):

  counts = Counter(list_of_values).most_common()
  clen = sum([c[1] for c in counts])

  return [[c[0], c[1], c[1] / clen] for c in counts]

###################################################################################

def smooth_escore_notes(escore_notes):

  values = [e[4] % 24 for e in escore_notes]

  smoothed = [values[0]]

  for i in range(1, len(values)):
      if abs(smoothed[-1] - values[i]) >= 12:
          if smoothed[-1] < values[i]:
              smoothed.append(values[i] - 12)
          else:
              smoothed.append(values[i] + 12)
      else:
          smoothed.append(values[i])

  smoothed_score = copy.deepcopy(escore_notes)

  for i, e in enumerate(smoothed_score):
    esn_octave = escore_notes[i][4] // 12
    e[4] = (esn_octave * 12) + smoothed[i]

  return smoothed_score

###################################################################################

def add_base_to_escore_notes(escore_notes,
                             base_octave=2, 
                             base_channel=2, 
                             base_patch=35, 
                             base_max_velocity=120,
                             return_base=False
                             ):
  

  score = copy.deepcopy(escore_notes)

  cscore = chordify_score([1000, score])

  base_score = []

  for c in cscore:
    chord = sorted([e for e in c if e[3] != 9], key=lambda x: x[4], reverse=True)
    base_score.append(chord[-1])

  base_score = smooth_escore_notes(base_score)

  for e in base_score:
    e[3] = base_channel
    e[4] = (base_octave * 12) + (e[4] % 12)
    e[5] = e[4]
    e[6] = base_patch

  adjust_score_velocities(base_score, base_max_velocity)

  if return_base:
    final_score = sorted(base_score, key=lambda x: (x[1], -x[4], x[6]))

  else:
    final_score = sorted(escore_notes + base_score, key=lambda x: (x[1], -x[4], x[6]))

  return final_score

###################################################################################

def add_drums_to_escore_notes(escore_notes, 
                              heavy_drums_pitches=[36, 38, 47],
                              heavy_drums_velocity=110,
                              light_drums_pitches=[51, 54],
                              light_drums_velocity=127,
                              drums_max_velocity=127,
                              drums_ratio_time_divider=4,
                              return_drums=False
                              ):

  score = copy.deepcopy([e for e in escore_notes if e[3] != 9])

  cscore = chordify_score([1000, score])

  drums_score = []

  for c in cscore:
    min_dur = max(1, min([e[2] for e in c]))
    if not (c[0][1] % drums_ratio_time_divider):
      drum_note = ['note', c[0][1], min_dur, 9, heavy_drums_pitches[c[0][4] % len(heavy_drums_pitches)], heavy_drums_velocity, 128]
    else:
      drum_note = ['note', c[0][1], min_dur, 9, light_drums_pitches[c[0][4] % len(light_drums_pitches)], light_drums_velocity, 128]
    drums_score.append(drum_note)

  adjust_score_velocities(drums_score, drums_max_velocity)

  if return_drums:
    final_score = sorted(drums_score, key=lambda x: (x[1], -x[4], x[6]))

  else:
    final_score = sorted(score + drums_score, key=lambda x: (x[1], -x[4], x[6]))

  return final_score

###################################################################################

def find_pattern_start_indexes(values, pattern):

  start_indexes = []

  count = 0

  for i in range(len(values)- len(pattern)):
    chunk = values[i:i+len(pattern)]

    if chunk == pattern:
      start_indexes.append(i)

  return start_indexes

###################################################################################

def escore_notes_lrno_pattern(escore_notes, mode='chords'):

  cscore = chordify_score([1000, escore_notes])

  checked_cscore = advanced_check_and_fix_chords_in_chordified_score(cscore)

  chords_toks = []
  chords_idxs = []

  for i, c in enumerate(checked_cscore[0]):

    pitches = sorted([p[4] for p in c if p[3] != 9], reverse=True)
    tchord = pitches_to_tones_chord(pitches)

    if tchord:
      
      if mode == 'chords':
        token = ALL_CHORDS_FULL.index(tchord)
      
      elif mode == 'high pitches':
        token = pitches[0]

      elif mode == 'high pitches tones':
        token = pitches[0] % 12

      else:
        token = ALL_CHORDS_FULL.index(tchord)

      chords_toks.append(token)
      chords_idxs.append(i)

  lrno_pats = find_lrno_patterns(chords_toks)

  if lrno_pats:

    lrno_pattern = list(lrno_pats[0][2])

    start_idx = chords_idxs[find_pattern_start_indexes(chords_toks, lrno_pattern)[0]]
    end_idx = chords_idxs[start_idx + len(lrno_pattern)]

    return recalculate_score_timings(flatten(cscore[start_idx:end_idx]))

  else:
    return None

###################################################################################

def chordified_score_pitches(chordified_score, 
                             mode='dominant',
                             return_tones=False,
                             omit_drums=True,
                             score_patch=-1,
                             channels_index=3,
                             pitches_index=4,
                             patches_index=6                          
                            ):

  results = []

  for c in chordified_score:
    
    if -1 < score_patch < 128:
      ptcs = sorted([e[pitches_index] for e in c if e[channels_index] != 9 and e[patches_index] == score_patch], reverse=True)
    
    else:
      ptcs = sorted([e[pitches_index] for e in c if e[channels_index] != 9], reverse=True)

    if ptcs:

      if mode == 'dominant':
        
        mtone = statistics.mode([p % 12 for p in ptcs])
        
        if return_tones:
          results.append(mtone)
        
        else:
          results.append(sorted(set([p for p in ptcs if p % 12 == mtone]), reverse=True))
      
      elif mode == 'high':
        
        if return_tones:
          results.append(ptcs[0] % 12)

        else:
          results.append([ptcs[0]])

      elif mode == 'base':

        if return_tones:
          results.append(ptcs[-1] % 12)

        else:
          results.append([ptcs[-1]])

      elif mode == 'average':

        if return_tones:
          results.append(statistics.mean(ptcs) % 12)

        else:
          results.append([statistics.mean(ptcs)])

      else:

        mtone = statistics.mode([p % 12 for p in ptcs])
        
        if return_tones:
          results.append(mtone)
        
        else:
          results.append(sorted(set([p for p in ptcs if p % 12 == mtone]), reverse=True))

    else:

      if not omit_drums:
        
        if return_tones:
          results.append(-1)
        
        else:
          results.append([-1])

  return results
  
###################################################################################

def escore_notes_times_tones(escore_notes, 
                             tones_mode='dominant', 
                             return_abs_times=True,
                             omit_drums=False
                             ):

  cscore = chordify_score([1000, escore_notes])
  
  tones = chordified_score_pitches(cscore, return_tones=True, mode=tones_mode, omit_drums=omit_drums)

  if return_abs_times:
    times = sorted([c[0][1] for c in cscore])
  
  else:
    times = escore_notes_delta_times(escore_notes, omit_zeros=True, omit_drums=omit_drums)
    
    if len(times) != len(tones):
      times = [0] + times

  return [[t, to] for t, to in zip(times, tones)]

###################################################################################

def escore_notes_middle(escore_notes, 
                        length=10, 
                        use_chords=True
                        ):

  if use_chords:
    score = chordify_score([1000, escore_notes])

  else:
    score = escore_notes

  middle_idx = len(score) // 2

  slen = min(len(score) // 2, length // 2)

  start_idx = middle_idx - slen
  end_idx = middle_idx + slen

  if use_chords:
    return flatten(score[start_idx:end_idx])

  else:
    return score[start_idx:end_idx]

###################################################################################

ALL_CHORDS_FULL = [[0], [0, 3], [0, 3, 5], [0, 3, 5, 8], [0, 3, 5, 9], [0, 3, 5, 10], [0, 3, 6],
                  [0, 3, 6, 9], [0, 3, 6, 10], [0, 3, 7], [0, 3, 7, 10], [0, 3, 8], [0, 3, 9],
                  [0, 3, 10], [0, 4], [0, 4, 6], [0, 4, 6, 9], [0, 4, 6, 10], [0, 4, 7],
                  [0, 4, 7, 10], [0, 4, 8], [0, 4, 9], [0, 4, 10], [0, 5], [0, 5, 8], [0, 5, 9],
                  [0, 5, 10], [0, 6], [0, 6, 9], [0, 6, 10], [0, 7], [0, 7, 10], [0, 8], [0, 9],
                  [0, 10], [1], [1, 4], [1, 4, 6], [1, 4, 6, 9], [1, 4, 6, 10], [1, 4, 6, 11],
                  [1, 4, 7], [1, 4, 7, 10], [1, 4, 7, 11], [1, 4, 8], [1, 4, 8, 11], [1, 4, 9],
                  [1, 4, 10], [1, 4, 11], [1, 5], [1, 5, 8], [1, 5, 8, 11], [1, 5, 9],
                  [1, 5, 10], [1, 5, 11], [1, 6], [1, 6, 9], [1, 6, 10], [1, 6, 11], [1, 7],
                  [1, 7, 10], [1, 7, 11], [1, 8], [1, 8, 11], [1, 9], [1, 10], [1, 11], [2],
                  [2, 5], [2, 5, 8], [2, 5, 8, 11], [2, 5, 9], [2, 5, 10], [2, 5, 11], [2, 6],
                  [2, 6, 9], [2, 6, 10], [2, 6, 11], [2, 7], [2, 7, 10], [2, 7, 11], [2, 8],
                  [2, 8, 11], [2, 9], [2, 10], [2, 11], [3], [3, 5], [3, 5, 8], [3, 5, 8, 11],
                  [3, 5, 9], [3, 5, 10], [3, 5, 11], [3, 6], [3, 6, 9], [3, 6, 10], [3, 6, 11],
                  [3, 7], [3, 7, 10], [3, 7, 11], [3, 8], [3, 8, 11], [3, 9], [3, 10], [3, 11],
                  [4], [4, 6], [4, 6, 9], [4, 6, 10], [4, 6, 11], [4, 7], [4, 7, 10], [4, 7, 11],
                  [4, 8], [4, 8, 11], [4, 9], [4, 10], [4, 11], [5], [5, 8], [5, 8, 11], [5, 9],
                  [5, 10], [5, 11], [6], [6, 9], [6, 10], [6, 11], [7], [7, 10], [7, 11], [8],
                  [8, 11], [9], [10], [11]]

###################################################################################

def escore_notes_to_parsons_code(escore_notes,
                                 times_index=1,
                                 pitches_index=4,
                                 return_as_list=False
                                 ):
  
  parsons = "*"
  parsons_list = []

  prev = ['note', -1, -1, -1, -1, -1, -1]

  for e in escore_notes:
    if e[times_index] != prev[times_index]:

      if e[pitches_index] > prev[pitches_index]:
          parsons += "U"
          parsons_list.append(1)

      elif e[pitches_index] < prev[pitches_index]:
          parsons += "D"
          parsons_list.append(-1)

      elif e[pitches_index] == prev[pitches_index]:
          parsons += "R"
          parsons_list.append(0)
      
      prev = e

  if return_as_list:
    return parsons_list
  
  else:
    return parsons

###################################################################################

def all_consequtive(list_of_values):
  return all(b > a for a, b in zip(list_of_values[:-1], list_of_values[1:]))

###################################################################################

def escore_notes_patches(escore_notes, patches_index=6):
  return sorted(set([e[patches_index] for e in escore_notes]))

###################################################################################

def build_suffix_array(lst):

    n = len(lst)

    suffixes = [(lst[i:], i) for i in range(n)]
    suffixes.sort()
    suffix_array = [suffix[1] for suffix in suffixes]

    return suffix_array

###################################################################################

def build_lcp_array(lst, suffix_array):

    n = len(lst)
    rank = [0] * n
    lcp = [0] * n

    for i, suffix in enumerate(suffix_array):
      rank[suffix] = i

    h = 0

    for i in range(n):
      if rank[i] > 0:

        j = suffix_array[rank[i] - 1]

        while i + h < n and j + h < n and lst[i + h] == lst[j + h]:
          h += 1

        lcp[rank[i]] = h

        if h > 0:
          h -= 1

    return lcp

###################################################################################

def find_lrno_pattern_fast(lst):
    n = len(lst)
    if n == 0:
      return []

    suffix_array = build_suffix_array(lst)
    lcp_array = build_lcp_array(lst, suffix_array)

    max_len = 0
    start_index = 0

    for i in range(1, n):
      if lcp_array[i] > max_len:
        if suffix_array[i] + lcp_array[i] <= suffix_array[i - 1] or suffix_array[i - 1] + lcp_array[i - 1] <= suffix_array[i]:
          max_len = lcp_array[i]
          start_index = suffix_array[i]

    return lst[start_index:start_index + max_len]

###################################################################################

def find_chunk_indexes(original_list, chunk, ignore_index=-1):

  chunk_length = len(chunk)

  for i in range(len(original_list) - chunk_length + 1):

    chunk_index = 0
    start_index = ignore_index

    for j in range(i, len(original_list)):
      if original_list[j] == chunk[chunk_index]:

        if start_index == ignore_index:
          start_index = j

        chunk_index += 1

        if chunk_index == chunk_length:
          return [start_index, j]

      elif original_list[j] != ignore_index:
        break

  return None

###################################################################################

def escore_notes_lrno_pattern_fast(escore_notes, 
                                   channels_index=3, 
                                   pitches_index=4, 
                                   zero_start_time=True
                                  ):

  cscore = chordify_score([1000, escore_notes])

  score_chords = []

  for c in cscore:

    tchord = sorted(set([e[pitches_index] % 12 for e in c if e[channels_index] != 9]))

    chord_tok = -1

    if tchord:

      if tchord not in ALL_CHORDS_FULL:
        tchord = check_and_fix_tones_chord(tchord)

      chord_tok = ALL_CHORDS_FULL.index(tchord)

    score_chords.append(chord_tok)

  schords = [c for c in score_chords if c != -1]

  lrno = find_lrno_pattern_fast(schords)

  if lrno:

    sidx, eidx = find_chunk_indexes(score_chords, lrno)

    escore_notes_lrno_pattern = flatten(cscore[sidx:eidx+1])

    if escore_notes_lrno_pattern is not None:

      if zero_start_time:
        return recalculate_score_timings(escore_notes_lrno_pattern)

      else:
        return escore_notes_lrno_pattern

    else:
      return None
  
  else:
    return None

###################################################################################

def escore_notes_durations_counter(escore_notes, 
                                   min_duration=0, 
                                   durations_index=2, 
                                   channels_index=3
                                   ):
  
  escore = [e for e in escore_notes if e[channels_index] != 9]
  durs = [e[durations_index] for e in escore if e[durations_index] >= min_duration]
  zero_durs = sum([1 for e in escore if e[durations_index] == 0])
  
  return [len(durs), len(escore), zero_durs, Counter(durs).most_common()]

###################################################################################

def count_bad_chords_in_chordified_score(chordified_score,  
                                         pitches_index=4,
                                         patches_index=6,
                                         max_patch=127, 
                                         use_full_chords=False
                                         ):

  if use_full_chords:
    CHORDS = ALL_CHORDS_FULL

  else:
    CHORDS = ALL_CHORDS_SORTED

  bad_chords_count = 0

  for c in chordified_score:

    cpitches = [e[pitches_index] for e in c if e[patches_index] <= max_patch]
    tones_chord = sorted(set([p % 12 for p in cpitches]))

    if tones_chord:
      if tones_chord not in CHORDS:
        bad_chords_count += 1

  return [bad_chords_count, len(chordified_score)]

###################################################################################

def needleman_wunsch_aligner(seq1,
                             seq2,
                             align_idx,
                             gap_penalty=-1,
                             match_score=2,
                             mismatch_penalty=-1
                             ):
    
    n = len(seq1)
    m = len(seq2)
    
    score_matrix = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        score_matrix[i][0] = gap_penalty * i
    for j in range(1, m + 1):
        score_matrix[0][j] = gap_penalty * j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1][align_idx] == seq2[j-1][align_idx] else mismatch_penalty)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    align1, align2 = [], []
    i, j = n, m
    
    while i > 0 and j > 0:
        
        score = score_matrix[i][j]
        score_diag = score_matrix[i-1][j-1]
        score_up = score_matrix[i-1][j]
        score_left = score_matrix[i][j-1]

        if score == score_diag + (match_score if seq1[i-1][align_idx] == seq2[j-1][align_idx] else mismatch_penalty):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif score == score_up + gap_penalty:
            align1.append(seq1[i-1])
            align2.append([None] * 6)
            i -= 1
        elif score == score_left + gap_penalty:
            align1.append([None] * 6)
            align2.append(seq2[j-1])
            j -= 1

    while i > 0:
        align1.append(seq1[i-1])
        align2.append([None] * 6)
        i -= 1
    while j > 0:
        align1.append([None] * 6)
        align2.append(seq2[j-1])
        j -= 1

    align1.reverse()
    align2.reverse()

    return align1, align2

###################################################################################

def align_escore_notes_to_escore_notes(src_escore_notes,
                                       trg_escore_notes,
                                       recalculate_scores_timings=True,
                                       pitches_idx=4
                                       ):
    
    if recalculate_scores_timings:
        src_escore_notes = recalculate_score_timings(src_escore_notes)
        trg_escore_notes = recalculate_score_timings(trg_escore_notes)
        
    src_align1, trg_align2 = needleman_wunsch_aligner(src_escore_notes, trg_escore_notes, pitches_idx)

    aligned_scores = [[al[0], al[1]] for al in zip(src_align1, trg_align2) if al[0][0] is not None and al[1][0] is not None]
    
    return aligned_scores

###################################################################################

def t_to_n(arr, si, t):

    ct = 0
    ci = si
    
    while ct + arr[ci][1] < t and ci < len(arr)-1:
        ct += arr[ci][1]
        ci += 1

    return ci+1

###################################################################################

def max_sum_chunk_idxs(arr, t=255):

    n = t_to_n(arr, 0, t)
    
    if n > len(arr):
        return [0, n]

    max_sum = 0
    max_sum_start_index = 0
    
    max_sum_start_idxs = [0, len(arr), sum([a[0] for a in arr])]

    for i in range(len(arr)):

        n = t_to_n(arr, i, t)

        current_sum  = sum([a[0] for a in arr[i:n]])
        current_time = sum([a[1] for a in arr[i:n]])

        if current_sum > max_sum and current_time <= t:
            max_sum = current_sum
            max_sum_start_idxs = [i, n, max_sum]

    return max_sum_start_idxs

###################################################################################

def find_highest_density_escore_notes_chunk(escore_notes, max_chunk_time=512):

    dscore = delta_score_notes(escore_notes)
    
    cscore = chordify_score([d[1:] for d in dscore])

    notes_counts = [[len(c), c[0][0]] for c in cscore]

    msc_idxs = max_sum_chunk_idxs(notes_counts, max_chunk_time)

    chunk_dscore = [['note'] + c for c in flatten(cscore[msc_idxs[0]:msc_idxs[1]])]

    chunk_escore = recalculate_score_timings(delta_score_to_abs_score(chunk_dscore))
    
    return chunk_escore

###################################################################################

def advanced_add_drums_to_escore_notes(escore_notes,
                                       main_beat_min_dtime=5,
                                       main_beat_dtime_thres=1,
                                       drums_durations_value=2,
                                       drums_pitches_velocities=[(36, 100), 
                                                                 (38, 100), 
                                                                 (41, 125)],
                                       recalculate_score_timings=True,
                                       intro_drums_count=4,
                                       intro_drums_time_k=4,
                                       intro_drums_pitch_velocity=[37, 110]
                                      ):

    #===========================================================
    
    new_dscore = delta_score_notes(escore_notes)

    times = [d[1] for d in new_dscore if d[1] != 0]

    time = [c[0] for c in Counter(times).most_common() if c[0] >= main_beat_min_dtime][0]

    #===========================================================

    if intro_drums_count > 0:

        drums_score = []

        for i in range(intro_drums_count):

            if i == 0:
                dtime = 0

            else:
                dtime = time
            
            drums_score.append(['note', 
                                dtime * intro_drums_time_k, 
                                drums_durations_value, 
                                9, 
                                intro_drums_pitch_velocity[0], 
                                intro_drums_pitch_velocity[1], 
                                128]
                              )
            
        new_dscore[0][1] = time * intro_drums_time_k

        new_dscore = drums_score + new_dscore

    #===========================================================

    for e in new_dscore:
    
        if abs(e[1] - time) == main_beat_dtime_thres:
            e[1] = time

        if recalculate_score_timings:
        
            if e[1] % time != 0 and e[1] > time:
                if e[1] % time < time // 2:
                    e[1] -= e[1] % time
        
                else:
                    e[1] += time - (e[1] % time)

    #===========================================================

    drums_score = []
    
    dtime = 0
    
    idx = 0
    
    for i, e in enumerate(new_dscore):
        
        drums_score.append(e)
        
        dtime += e[1]

        if e[1] != 0:
            idx += 1

        if i >= intro_drums_count:
    
            if (e[1] % time == 0 and e[1] != 0) or i == 0:
                
                if idx % 2 == 0 and e[1] != 0:
                    drums_score.append(['note', 
                                        0, 
                                        drums_durations_value, 
                                        9, 
                                        drums_pitches_velocities[0][0], 
                                        drums_pitches_velocities[0][1], 
                                        128]
                                      )
                    
                if idx % 2 != 0 and e[1] != 0:
                    drums_score.append(['note', 
                                        0, 
                                        drums_durations_value, 
                                        9, 
                                        drums_pitches_velocities[1][0], 
                                        drums_pitches_velocities[1][1], 
                                        128]
                                      )
        
                if idx % 4 == 0 and e[1] != 0:
                    drums_score.append(['note', 
                                        0, 
                                        drums_durations_value, 
                                        9, 
                                        drums_pitches_velocities[2][0], 
                                        drums_pitches_velocities[2][1], 
                                        128]
                                      )

    #===========================================================
    
    return delta_score_to_abs_score(drums_score)

###################################################################################

MIDI_TEXT_EVENTS = ['text_event',
                    'copyright_text_event',
                    'track_name',
                    'instrument_name',
                    'lyric',
                    'marker',
                    'cue_point',
                    'text_event_08',
                    'text_event_09',
                    'text_event_0a',
                    'text_event_0b',
                    'text_event_0c',
                    'text_event_0d',
                    'text_event_0e',
                    'text_event_0f'
                  ]

###################################################################################

import hashlib
import re

###################################################################################

def get_md5_hash(data):
    return hashlib.md5(data).hexdigest()

###################################################################################

def is_valid_md5_hash(string):
    return bool(re.match(r'^[a-fA-F0-9]{32}$', string))

###################################################################################

def clean_string(original_string,
                 regex=r'[^a-zA-Z0-9 ]',
                 remove_duplicate_spaces=True,
                 title=False
                ):
    
    cstr1 = re.sub(regex, '', original_string)

    if title:
        cstr1 = cstr1.title()

    if remove_duplicate_spaces:
        return re.sub(r'[ ]+', ' ', cstr1).strip()

    else:
        return cstr1
    
###################################################################################
    
def encode_to_ord(text, chars_range=[], sub_char='', chars_shift=0):

    if not chars_range:
        chars_range = [32] + list(range(65, 91)) + list(range(97, 123))

    if sub_char:
        chars_range.append(ord(sub_char))

    chars_range = sorted(set(chars_range))

    encoded = []
    
    for char in text:
        if ord(char) in chars_range:
            encoded.append(chars_range.index(ord(char)) + chars_shift)
            
        else:
            if sub_char:
                encoded.append(chars_range.index(ord(sub_char)) + chars_shift)
        
    
    return [encoded, chars_range]

###################################################################################

def decode_from_ord(ord_list, chars_range=[], sub_char='', chars_shift=0):

    if not chars_range:
        chars_range = [32] + list(range(65, 91)) + list(range(97, 123))

    if sub_char:
        chars_range.append(ord(sub_char))

    chars_range = sorted(set(chars_range))
        
    return ''.join(chr(chars_range[num-chars_shift]) if 0 <= num-chars_shift < len(chars_range) else sub_char for num in ord_list)

###################################################################################

def lists_similarity(list1, list2, by_elements=True, by_sum=True):
    
    if len(list1) != len(list2):
        return -1
    
    element_ratios = []
    total_counts1 = sum(list1)
    total_counts2 = sum(list2)

    for a, b in zip(list1, list2):
        if a == 0 and b == 0:
            element_ratios.append(1)
        elif a == 0 or b == 0:
            element_ratios.append(0)
        else:
            element_ratios.append(min(a, b) / max(a, b))

    average_element_ratio = sum(element_ratios) / len(element_ratios)

    total_counts_ratio = min(total_counts1, total_counts2) / max(total_counts1, total_counts2)

    if by_elements and by_sum:
        return (average_element_ratio + total_counts_ratio) / 2

    elif by_elements and not by_sum:
        return average_element_ratio

    elif not by_elements and by_sum:
        return total_counts_ratio

    else:
        return -1

###################################################################################

def find_indexes(lst, value, mode='equal', dual_mode=True):

    indexes = []

    if mode == 'equal' or dual_mode:    
        indexes.extend([index for index, elem in enumerate(lst) if elem == value])

    if mode == 'smaller':
        indexes.extend([index for index, elem in enumerate(lst) if elem < value])

    if mode == 'larger':
        indexes.extend([index for index, elem in enumerate(lst) if elem > value])

    return sorted(set(indexes))

###################################################################################

NUMERALS = ["one", "two", "three", "four", 
            "five", "six", "seven", "eight", 
            "nine", "ten", "eleven", "twelve", 
            "thirteen", "fourteen", "fifteen", "sixteen"
           ]

SEMITONES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

BASIC_SCALES = ['Major', 'Minor']

###################################################################################

def alpha_str(string):
    astr = re.sub(r'[^a-zA-Z ()0-9]', '', string).strip()
    return re.sub(r'\s+', ' ', astr).strip()

###################################################################################

def escore_notes_to_text_description(escore_notes, 
                                     song_name='', 
                                     artist_name='',
                                     timings_divider=16,
                                     return_feat_dict=False,
                                     return_feat_dict_vals=False
                                    ):
    
    #==============================================================================
    
    feat_dict = {}
    feat_dict_vals = {}

    #==============================================================================

    song_time_min = (escore_notes[-1][1] * timings_divider) / 1000 / 60

    if song_time_min < 1.5:
        song_length = 'short'
    
    elif 1.5 <= song_time_min < 2.5:
        song_length = 'average'
    
    elif song_time_min >= 2.5:
        song_length = 'long'
        
    feat_dict['song_len'] = song_length.capitalize()
    feat_dict_vals['song_len'] = song_time_min

    #==============================================================================

    escore_times = [e[1] for e in escore_notes if e[3] != 9]
    
    comp_type = ''

    if len(escore_times) > 0:
        if len(escore_times) == len(set(escore_times)):
            comp_type = 'monophonic melody'
            ctype = 'melody'
            ctv = 0
        
        elif len(escore_times) >= len(set(escore_times)) and 1 in Counter(escore_times).values():
            comp_type = 'melody and accompaniment'
            ctype = 'song'
            ctv = 1
        
        elif len(escore_times) >= len(set(escore_times)) and 1 not in Counter(escore_times).values():
            comp_type = 'accompaniment'
            ctype = 'song'
            ctv = 2
    
    else:
        comp_type = 'drum track'
        ctype = 'drum track'
        ctv = 3
        
    feat_dict['song_type'] = comp_type.capitalize()
    feat_dict_vals['song_type'] = ctv

    #==============================================================================

    all_patches = [e[6] for e in escore_notes]

    patches = ordered_set(all_patches)[:16]
    
    instruments = [alpha_str(Number2patch[p]) for p in patches if p < 128]
    
    if instruments:

        nd_patches_counts = Counter([p for p in all_patches if p < 128]).most_common()

        dominant_instrument = alpha_str(Number2patch[nd_patches_counts[0][0]])
        
        feat_dict['most_com_instr'] = instruments
        feat_dict_vals['most_com_instr'] = [p for p in patches if p < 128]
        
    else:
        feat_dict['most_com_instr'] = None
        feat_dict_vals['most_com_instr'] = []

    if 128 in patches:
        drums_present = True

        drums_pitches = [e[4] for e in escore_notes if e[3] == 9]

        most_common_drums = [alpha_str(Notenum2percussion[p[0]]) for p in Counter(drums_pitches).most_common(3) if p[0] in Notenum2percussion]
        
        feat_dict['most_com_drums'] = most_common_drums
        feat_dict_vals['most_com_drums'] = [p[0] for p in Counter(drums_pitches).most_common(3)]

    else:
        drums_present = False
        
        feat_dict['most_com_drums'] = None
        
        feat_dict_vals['most_com_drums'] = []

    #==============================================================================

    pitches = [e[4] for e in escore_notes if e[3] != 9]
    
    key = ''
    
    if pitches:
        key = SEMITONES[statistics.mode(pitches) % 12]
        
        feat_dict['key'] = key.title()
        feat_dict_vals['key'] = statistics.mode(pitches) % 12
        
    else:
        feat_dict['key'] = None
        feat_dict_vals['key'] = -1

    #==============================================================================
    
    scale = ''
    mood = ''
    
    feat_dict['scale'] = None
    feat_dict['mood'] = None
    feat_dict_vals['scale'] = -1
    feat_dict_vals['mood'] = -1
    
    if pitches:
    
        result = escore_notes_scale(escore_notes)
        
        scale = result[0]
        mood = result[1].split(' ')[0].lower()
        
        feat_dict['scale'] = scale.title()
        feat_dict['mood'] = mood.title()
        
        res = escore_notes_scale(escore_notes, return_scale_indexes=True)
        feat_dict_vals['scale'] = res[0]            
        feat_dict_vals['mood'] = res[1]

    #==============================================================================
        
    feat_dict['rythm'] = None
    feat_dict['tempo'] = None
    feat_dict['tone'] = None
    feat_dict['dynamics'] = None

    feat_dict_vals['rythm'] = -1
    feat_dict_vals['tempo'] = -1
    feat_dict_vals['tone'] = -1
    feat_dict_vals['dynamics'] = -1

    if pitches:
    
        escore_averages = escore_notes_averages(escore_notes, return_ptcs_and_vels=True)

        if escore_averages[0] < (128 / timings_divider):
            rythm = 'fast'
            ryv = 0
        
        elif (128 / timings_divider) <= escore_averages[0] <= (192 / timings_divider):
            rythm = 'average'
            ryv = 1
        
        elif escore_averages[0] > (192 / timings_divider):
            rythm = 'slow'
            ryv = 2
        
        if escore_averages[1] < (256 / timings_divider):
            tempo = 'fast'
            tev = 0
        
        elif (256 / timings_divider) <= escore_averages[1] <= (384 / timings_divider):
            tempo = 'average'
            tev = 1
        
        elif escore_averages[1] > (384 / timings_divider):
            tempo = 'slow'
            tev = 2
        
        if escore_averages[2] < 50:
            tone = 'bass'
            tov = 0
        
        elif 50 <= escore_averages[2] <= 70:
            tone = 'midrange'
            tov = 1
        
        elif escore_averages[2] > 70:
            tone = 'treble'
            tov = 2
        
        if escore_averages[3] < 64:
            dynamics = 'quiet'
            dyn = 0
        
        elif 64 <= escore_averages[3] <= 96:
            dynamics = 'average'
            dyn = 1
        
        elif escore_averages[3] > 96:
            dynamics = 'loud'
            dyn = 2
            
        feat_dict['rythm'] = rythm.title()
        feat_dict['tempo'] = tempo.title()
        feat_dict['tone'] = tone.title()
        feat_dict['dynamics'] = dynamics.title()
        
        feat_dict_vals['rythm'] = ryv
        feat_dict_vals['tempo'] = tev
        feat_dict_vals['tone'] = tov
        feat_dict_vals['dynamics'] = dyn

    #==============================================================================
            
    mono_melodies = escore_notes_monoponic_melodies([e for e in escore_notes if e[6] < 88])
 
    lead_melodies = []
    base_melodies = []
    
    feat_dict['lead_mono_mels'] = None
    feat_dict['base_mono_mels'] = None
    
    feat_dict_vals['lead_mono_mels'] = []
    feat_dict_vals['base_mono_mels'] = []
 
    if mono_melodies:
            
        for mel in mono_melodies:
            
            escore_avgs = escore_notes_pitches_range(escore_notes, range_patch = mel[0])
            
            if mel[0] in LEAD_INSTRUMENTS and escore_avgs[3] > 60:
                lead_melodies.append([Number2patch[mel[0]], mel[1]])
                feat_dict_vals['lead_mono_mels'].append(mel[0])

            elif mel[0] in BASE_INSTRUMENTS and escore_avgs[3] <= 60:
                base_melodies.append([Number2patch[mel[0]], mel[1]])
                feat_dict_vals['base_mono_mels'].append(mel[0])
    
        if lead_melodies:
            lead_melodies.sort(key=lambda x: x[1], reverse=True)
            feat_dict['lead_mono_mels'] = lead_melodies
            
        if base_melodies:
            base_melodies.sort(key=lambda x: x[1], reverse=True)
            feat_dict['base_mono_mels'] = base_melodies

    #==============================================================================
        
    description = ''

    if song_name != '':
        description = 'The song "' + song_name + '"'
    
    if artist_name != '':
        description += ' by ' + artist_name

    if song_name != '' or artist_name != '':
        description += '.'
        description += '\n'
    
    description += 'The song is '

    if song_length != 'average':
        description += 'a ' + song_length

    else:
        description += 'an ' + song_length

    description += ' duration '

    description += comp_type + ' composition'

    if comp_type != 'drum track':

        if drums_present:
            description += ' with drums'
    
        else:
            description += ' without drums'
    
        if key and scale:
            description += ' in ' + key + ' ' + scale

    description += '.'

    description += '\n'
    
    if pitches:
        
        if comp_type not in ['monophonic melody', 'drum track']:

            description += 'This ' + mood + ' song has '
                
        elif comp_type == 'monophonic melody':
            
            description += 'This ' + mood + ' melody has '
            
        else:
            description += 'TThis drum track has '
            
        description += rythm + ' rythm, '
        description += tempo + ' tempo, '
        description += tone + ' tone and '
        description += dynamics + ' dynamics.'

        description += '\n'
        
        if instruments:
            
            if comp_type not in ['monophonic melody', 'drum track']:
                
                description += 'The song '
                
                if len(instruments) > 1:
                
                    description += 'features ' + NUMERALS[max(0, min(15, len(instruments)-1))] + ' instruments: '
                    description += ', '.join(instruments[:-1]) + ' and ' + instruments[-1] + '.'
                    
                else:
                    description += 'features one instrument: ' + instruments[0] + '.'
                    
                
                description += '\n'
                
                if instruments[0] != dominant_instrument:
                    description += 'The song opens with ' + instruments[0]

                    description += ' and primarily performed on ' + dominant_instrument + '.'
                    
                else:
                    description += 'The song opens with and performed on ' + instruments[0] + '.'
                    
                description += '\n'
                    
    if lead_melodies or base_melodies:
        
        tm_count = len(lead_melodies + base_melodies)
        
        if tm_count == 1:
            
            if lead_melodies:
                description += 'The song has one distinct lead melody played on ' + lead_melodies[0][0] + '.'
                
            else:
                description += 'The song has one distinct base melody played on ' + base_melodies[0][0] + '.'
            
            description += '\n'
            
        else:
            
            if lead_melodies and not base_melodies:
                
                if len(lead_melodies) == 1:
                    mword = 'melody'
                    
                else:
                    mword = 'melodies'
            
                description += 'The song has ' + NUMERALS[len(lead_melodies)-1] + ' distinct lead ' + mword + ' played on '
                
                if len(lead_melodies) > 1:               
                    description += ', '.join([l[0] for l in lead_melodies[:-1]]) + ' and ' + lead_melodies[-1][0] + '.'
                    
                else:
                    description += lead_melodies[0][0] + '.'
                    
                description += '\n'
            
            elif base_melodies and not lead_melodies:
                
                if len(base_melodies) == 1:
                    mword = 'melody'
                    
                else:
                    mword = 'melodies'
        
                description += 'The song has ' + NUMERALS[len(base_melodies)-1] + ' distinct base ' + mword + ' played on '
                
                if len(base_melodies) > 1:               
                    description += ', '.join([b[0] for b in base_melodies[:-1]]) + ' and ' + base_melodies[-1][0] + '.'
                    
                else:
                    description += base_melodies[0][0] + '.'
                
                description += '\n'        

            elif lead_melodies and base_melodies:
                
                if len(lead_melodies) == 1:
                    lmword = 'melody'
                    
                else:
                    lmword = 'melodies'
        
                description += 'The song has ' + NUMERALS[len(lead_melodies)-1] + ' distinct lead ' + lmword + ' played on '
                
                if len(lead_melodies) > 1:               
                    description += ', '.join([l[0] for l in lead_melodies[:-1]]) + ' and ' + lead_melodies[-1][0] + '.'
                    
                else:
                    description += lead_melodies[0][0] + '.'
                    
                if len(base_melodies) == 1:
                    bmword = 'melody'
                    
                else:
                    bmword = 'melodies'
        
                description += ' And ' + NUMERALS[len(base_melodies)-1] + ' distinct base ' + bmword + ' played on '
                
                if len(base_melodies) > 1:               
                    description += ', '.join([b[0] for b in base_melodies[:-1]]) + ' and ' + base_melodies[-1][0] + '.'
                    
                else:
                    description += base_melodies[0][0] + '.'
                
                description += '\n'

    if drums_present and most_common_drums:
        
        if len(most_common_drums) > 1:
            description += 'The drum track has predominant '
            description += ', '.join(most_common_drums[:-1]) + ' and ' + most_common_drums[-1] + '.'
            
        else:
            description += 'The drum track is a solo '
            description += most_common_drums[0] + '.'

        description += '\n'
        
    #==============================================================================
        
    final_feat_dict = []
        
    if return_feat_dict:
        final_feat_dict.append(feat_dict)
    
    if return_feat_dict_vals:
        final_feat_dict.append(feat_dict_vals)
        
    if return_feat_dict or return_feat_dict_vals:
        return final_feat_dict
    
    else:
        return description

###################################################################################

#==================================================================================
#
# Below constants code is a courtesy of MidiTok
#
# Retrieved on 12/29/2024
#
# https://github.com/Natooz/MidiTok/blob/main/src/miditok/constants.py
#
#==================================================================================

MIDI_FILES_EXTENSIONS = [".mid", ".midi", ".kar", ".MID", ".MIDI", ".KAR"]

# The recommended pitches for piano in the GM2 specs are from 21 to 108
PIANO_PITCH_RANGE = range(21, 109)

# Chord params
# "chord_unknown" specifies the range of number of notes that can form "unknown" chords
# (that do not fit in "chord_maps") to add in tokens.
# Known chord maps, with 0 as root note
BASIC_CHORDS_MAP = {
    "min": (0, 3, 7),
    "maj": (0, 4, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7dom": (0, 4, 7, 10),
    "7min": (0, 3, 7, 10),
    "7maj": (0, 4, 7, 11),
    "7halfdim": (0, 3, 6, 10),
    "7dim": (0, 3, 6, 9),
    "7aug": (0, 4, 8, 11),
    "9maj": (0, 4, 7, 10, 14),
    "9min": (0, 4, 7, 10, 13),
    }

# Drums
# Recommended range from the GM2 specs
DRUMS_PITCH_RANGE = range(27, 90)

# Used with chords
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# http://newt.phys.unsw.edu.au/jw/notes.html
# https://www.midi.org/specifications

# index i = program i+1 in the GM2 specs (7. Appendix A)
# index i = program i as retrieved by packages
MIDI_INSTRUMENTS = [
    # Piano
    {"name": "Acoustic Grand Piano", "pitch_range": range(21, 109)},
    {"name": "Bright Acoustic Piano", "pitch_range": range(21, 109)},
    {"name": "Electric Grand Piano", "pitch_range": range(21, 109)},
    {"name": "Honky-tonk Piano", "pitch_range": range(21, 109)},
    {"name": "Electric Piano 1", "pitch_range": range(28, 104)},
    {"name": "Electric Piano 2", "pitch_range": range(28, 104)},
    {"name": "Harpsichord", "pitch_range": range(41, 90)},
    {"name": "Clavi", "pitch_range": range(36, 97)},
    # Chromatic Percussion
    {"name": "Celesta", "pitch_range": range(60, 109)},
    {"name": "Glockenspiel", "pitch_range": range(72, 109)},
    {"name": "Music Box", "pitch_range": range(60, 85)},
    {"name": "Vibraphone", "pitch_range": range(53, 90)},
    {"name": "Marimba", "pitch_range": range(48, 85)},
    {"name": "Xylophone", "pitch_range": range(65, 97)},
    {"name": "Tubular Bells", "pitch_range": range(60, 78)},
    {"name": "Dulcimer", "pitch_range": range(60, 85)},
    # Organs
    {"name": "Drawbar Organ", "pitch_range": range(36, 97)},
    {"name": "Percussive Organ", "pitch_range": range(36, 97)},
    {"name": "Rock Organ", "pitch_range": range(36, 97)},
    {"name": "Church Organ", "pitch_range": range(21, 109)},
    {"name": "Reed Organ", "pitch_range": range(36, 97)},
    {"name": "Accordion", "pitch_range": range(53, 90)},
    {"name": "Harmonica", "pitch_range": range(60, 85)},
    {"name": "Tango Accordion", "pitch_range": range(53, 90)},
    # Guitars
    {"name": "Acoustic Guitar (nylon)", "pitch_range": range(40, 85)},
    {"name": "Acoustic Guitar (steel)", "pitch_range": range(40, 85)},
    {"name": "Electric Guitar (jazz)", "pitch_range": range(40, 87)},
    {"name": "Electric Guitar (clean)", "pitch_range": range(40, 87)},
    {"name": "Electric Guitar (muted)", "pitch_range": range(40, 87)},
    {"name": "Overdriven Guitar", "pitch_range": range(40, 87)},
    {"name": "Distortion Guitar", "pitch_range": range(40, 87)},
    {"name": "Guitar Harmonics", "pitch_range": range(40, 87)},
    # Bass
    {"name": "Acoustic Bass", "pitch_range": range(28, 56)},
    {"name": "Electric Bass (finger)", "pitch_range": range(28, 56)},
    {"name": "Electric Bass (pick)", "pitch_range": range(28, 56)},
    {"name": "Fretless Bass", "pitch_range": range(28, 56)},
    {"name": "Slap Bass 1", "pitch_range": range(28, 56)},
    {"name": "Slap Bass 2", "pitch_range": range(28, 56)},
    {"name": "Synth Bass 1", "pitch_range": range(28, 56)},
    {"name": "Synth Bass 2", "pitch_range": range(28, 56)},
    # Strings & Orchestral instruments
    {"name": "Violin", "pitch_range": range(55, 94)},
    {"name": "Viola", "pitch_range": range(48, 85)},
    {"name": "Cello", "pitch_range": range(36, 73)},
    {"name": "Contrabass", "pitch_range": range(28, 56)},
    {"name": "Tremolo Strings", "pitch_range": range(28, 94)},
    {"name": "Pizzicato Strings", "pitch_range": range(28, 94)},
    {"name": "Orchestral Harp", "pitch_range": range(23, 104)},
    {"name": "Timpani", "pitch_range": range(36, 58)},
    # Ensembles
    {"name": "String Ensembles 1", "pitch_range": range(28, 97)},
    {"name": "String Ensembles 2", "pitch_range": range(28, 97)},
    {"name": "SynthStrings 1", "pitch_range": range(36, 97)},
    {"name": "SynthStrings 2", "pitch_range": range(36, 97)},
    {"name": "Choir Aahs", "pitch_range": range(48, 80)},
    {"name": "Voice Oohs", "pitch_range": range(48, 80)},
    {"name": "Synth Voice", "pitch_range": range(48, 85)},
    {"name": "Orchestra Hit", "pitch_range": range(48, 73)},
    # Brass
    {"name": "Trumpet", "pitch_range": range(58, 95)},
    {"name": "Trombone", "pitch_range": range(34, 76)},
    {"name": "Tuba", "pitch_range": range(29, 56)},
    {"name": "Muted Trumpet", "pitch_range": range(58, 83)},
    {"name": "French Horn", "pitch_range": range(41, 78)},
    {"name": "Brass Section", "pitch_range": range(36, 97)},
    {"name": "Synth Brass 1", "pitch_range": range(36, 97)},
    {"name": "Synth Brass 2", "pitch_range": range(36, 97)},
    # Reed
    {"name": "Soprano Sax", "pitch_range": range(54, 88)},
    {"name": "Alto Sax", "pitch_range": range(49, 81)},
    {"name": "Tenor Sax", "pitch_range": range(42, 76)},
    {"name": "Baritone Sax", "pitch_range": range(37, 69)},
    {"name": "Oboe", "pitch_range": range(58, 92)},
    {"name": "English Horn", "pitch_range": range(52, 82)},
    {"name": "Bassoon", "pitch_range": range(34, 73)},
    {"name": "Clarinet", "pitch_range": range(50, 92)},
    # Pipe
    {"name": "Piccolo", "pitch_range": range(74, 109)},
    {"name": "Flute", "pitch_range": range(60, 97)},
    {"name": "Recorder", "pitch_range": range(60, 97)},
    {"name": "Pan Flute", "pitch_range": range(60, 97)},
    {"name": "Blown Bottle", "pitch_range": range(60, 97)},
    {"name": "Shakuhachi", "pitch_range": range(55, 85)},
    {"name": "Whistle", "pitch_range": range(60, 97)},
    {"name": "Ocarina", "pitch_range": range(60, 85)},
    # Synth Lead
    {"name": "Lead 1 (square)", "pitch_range": range(21, 109)},
    {"name": "Lead 2 (sawtooth)", "pitch_range": range(21, 109)},
    {"name": "Lead 3 (calliope)", "pitch_range": range(36, 97)},
    {"name": "Lead 4 (chiff)", "pitch_range": range(36, 97)},
    {"name": "Lead 5 (charang)", "pitch_range": range(36, 97)},
    {"name": "Lead 6 (voice)", "pitch_range": range(36, 97)},
    {"name": "Lead 7 (fifths)", "pitch_range": range(36, 97)},
    {"name": "Lead 8 (bass + lead)", "pitch_range": range(21, 109)},
    # Synth Pad
    {"name": "Pad 1 (new age)", "pitch_range": range(36, 97)},
    {"name": "Pad 2 (warm)", "pitch_range": range(36, 97)},
    {"name": "Pad 3 (polysynth)", "pitch_range": range(36, 97)},
    {"name": "Pad 4 (choir)", "pitch_range": range(36, 97)},
    {"name": "Pad 5 (bowed)", "pitch_range": range(36, 97)},
    {"name": "Pad 6 (metallic)", "pitch_range": range(36, 97)},
    {"name": "Pad 7 (halo)", "pitch_range": range(36, 97)},
    {"name": "Pad 8 (sweep)", "pitch_range": range(36, 97)},
    # Synth SFX
    {"name": "FX 1 (rain)", "pitch_range": range(36, 97)},
    {"name": "FX 2 (soundtrack)", "pitch_range": range(36, 97)},
    {"name": "FX 3 (crystal)", "pitch_range": range(36, 97)},
    {"name": "FX 4 (atmosphere)", "pitch_range": range(36, 97)},
    {"name": "FX 5 (brightness)", "pitch_range": range(36, 97)},
    {"name": "FX 6 (goblins)", "pitch_range": range(36, 97)},
    {"name": "FX 7 (echoes)", "pitch_range": range(36, 97)},
    {"name": "FX 8 (sci-fi)", "pitch_range": range(36, 97)},
    # Ethnic Misc.
    {"name": "Sitar", "pitch_range": range(48, 78)},
    {"name": "Banjo", "pitch_range": range(48, 85)},
    {"name": "Shamisen", "pitch_range": range(50, 80)},
    {"name": "Koto", "pitch_range": range(55, 85)},
    {"name": "Kalimba", "pitch_range": range(48, 80)},
    {"name": "Bag pipe", "pitch_range": range(36, 78)},
    {"name": "Fiddle", "pitch_range": range(55, 97)},
    {"name": "Shanai", "pitch_range": range(48, 73)},
    # Percussive
    {"name": "Tinkle Bell", "pitch_range": range(72, 85)},
    {"name": "Agogo", "pitch_range": range(60, 73)},
    {"name": "Steel Drums", "pitch_range": range(52, 77)},
    {"name": "Woodblock", "pitch_range": range(128)},
    {"name": "Taiko Drum", "pitch_range": range(128)},
    {"name": "Melodic Tom", "pitch_range": range(128)},
    {"name": "Synth Drum", "pitch_range": range(128)},
    {"name": "Reverse Cymbal", "pitch_range": range(128)},
    # SFX
    {"name": "Guitar Fret Noise, Guitar Cutting Noise", "pitch_range": range(128)},
    {"name": "Breath Noise, Flute Key Click", "pitch_range": range(128)},
    {
        "name": "Seashore, Rain, Thunder, Wind, Stream, Bubbles",
        "pitch_range": range(128),
    },
    {"name": "Bird Tweet, Dog, Horse Gallop", "pitch_range": range(128)},
    {
        "name": "Telephone Ring, Door Creaking, Door, Scratch, Wind Chime",
        "pitch_range": range(128),
    },
    {"name": "Helicopter, Car Sounds", "pitch_range": range(128)},
    {
        "name": "Applause, Laughing, Screaming, Punch, Heart Beat, Footstep",
        "pitch_range": range(128),
    },
    {"name": "Gunshot, Machine Gun, Lasergun, Explosion", "pitch_range": range(128)},
]

INSTRUMENTS_CLASSES = [
    {"name": "Piano", "program_range": range(8)},  # 0
    {"name": "Chromatic Percussion", "program_range": range(8, 16)},
    {"name": "Organ", "program_range": range(16, 24)},
    {"name": "Guitar", "program_range": range(24, 32)},
    {"name": "Bass", "program_range": range(32, 40)},
    {"name": "Strings", "program_range": range(40, 48)},  # 5
    {"name": "Ensemble", "program_range": range(48, 56)},
    {"name": "Brass", "program_range": range(56, 64)},
    {"name": "Reed", "program_range": range(64, 72)},
    {"name": "Pipe", "program_range": range(72, 80)},
    {"name": "Synth Lead", "program_range": range(80, 88)},  # 10
    {"name": "Synth Pad", "program_range": range(88, 96)},
    {"name": "Synth Effects", "program_range": range(96, 104)},
    {"name": "Ethnic", "program_range": range(104, 112)},
    {"name": "Percussive", "program_range": range(112, 120)},
    {"name": "Sound Effects", "program_range": range(120, 128)},  # 15
    {"name": "Drums", "program_range": range(-1, 0)},
]

# To easily get the class index of any instrument program
CLASS_OF_INST = [
    i
    for i, inst_class in enumerate(INSTRUMENTS_CLASSES)
    for _ in inst_class["program_range"]
]

# index i = program i+1 in the GM2 specs (8. Appendix B)
# index i = program i retrieved by packages
DRUMS_SETS = {
    0: "Standard",
    8: "Room",
    16: "Power",
    24: "Electronic",
    25: "Analog",
    32: "Jazz",
    40: "Brush",
    48: "Orchestra",
    56: "SFX",
}

# Control changes list (without specifications):
# https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
# Undefined and general control changes are not considered here
# All these attributes can take values from 0 to 127, with some of them being on/off
CONTROL_CHANGES = {
    # MSB
    0: "Bank Select",
    1: "Modulation Depth",
    2: "Breath Controller",
    4: "Foot Controller",
    5: "Portamento Time",
    6: "Data Entry",
    7: "Channel Volume",
    8: "Balance",
    10: "Pan",
    11: "Expression Controller",
    # LSB
    32: "Bank Select",
    33: "Modulation Depth",
    34: "Breath Controller",
    36: "Foot Controller",
    37: "Portamento Time",
    38: "Data Entry",
    39: "Channel Volume",
    40: "Balance",
    42: "Pan",
    43: "Expression Controller",
    # On / Off control changes, ≤63 off, ≥64 on
    64: "Damper Pedal",
    65: "Portamento",
    66: "Sostenuto",
    67: "Soft Pedal",
    68: "Legato Footswitch",
    69: "Hold 2",
    # Continuous controls
    70: "Sound Variation",
    71: "Timbre/Harmonic Intensity",
    72: "Release Time",
    73: "Attack Time",
    74: "Brightness",
    75: "Decay Time",
    76: "Vibrato Rate",
    77: "Vibrato Depth",
    78: "Vibrato Delay",
    84: "Portamento Control",
    88: "High Resolution Velocity Prefix",
    # Effects depths
    91: "Reverb Depth",
    92: "Tremolo Depth",
    93: "Chorus Depth",
    94: "Celeste Depth",
    95: "Phaser Depth",
    # Registered parameters numbers
    96: "Data Increment",
    97: "Data Decrement",
    #  98: 'Non-Registered Parameter Number (NRPN) - LSB',
    #  99: 'Non-Registered Parameter Number (NRPN) - MSB',
    100: "Registered Parameter Number (RPN) - LSB",
    101: "Registered Parameter Number (RPN) - MSB",
    # Channel mode controls
    120: "All Sound Off",
    121: "Reset All Controllers",
    122: "Local Control On/Off",
    123: "All Notes Off",
    124: "Omni Mode Off",  # + all notes off
    125: "Omni Mode On",  # + all notes off
    126: "Mono Mode On",  # + poly off, + all notes off
    127: "Poly Mode On",  # + mono off, +all notes off
}

###################################################################################

def patches_onset_times(escore_notes, times_idx=1, patches_idx=6):
    
    patches = [e[patches_idx] for e in escore_notes]

    patches_oset = ordered_set(patches)

    patches_onset_times = []

    for p in patches_oset:
        for e in escore_notes:
            if e[patches_idx] == p:
                patches_onset_times.append([p, e[times_idx]])
                break

    return patches_onset_times

###################################################################################

def count_escore_notes_patches(escore_notes, patches_idx=6):

    patches = [e[patches_idx] for e in escore_notes]

    return Counter(patches).most_common()

###################################################################################

def escore_notes_monoponic_melodies(escore_notes,
                                    bad_notes_ratio=0.0,
                                    times_idx=1,
                                    patches_idx=6
                                    ):

    patches = escore_notes_patches(escore_notes, patches_index=patches_idx)

    monophonic_melodies = []

    for p in patches:
        patch_score = [e for e in escore_notes if e[patches_idx] == p]

        ps_times = [e[times_idx] for e in patch_score]

        if len(ps_times) <= len(set(ps_times)) * (1+bad_notes_ratio):
            monophonic_melodies.append([p, len(patch_score)])
            
    return monophonic_melodies

###################################################################################

from itertools import groupby
from operator import itemgetter

def group_by_threshold(data, threshold, groupby_idx):

    data.sort(key=itemgetter(groupby_idx))

    grouped_data = []
    cluster = []

    for i, item in enumerate(data):
        if not cluster:
            cluster.append(item)
        elif abs(item[groupby_idx] - cluster[-1][groupby_idx]) <= threshold:
            cluster.append(item)
        else:
            grouped_data.append(cluster)
            cluster = [item]
    
    if cluster:
        grouped_data.append(cluster)
        
    return grouped_data

###################################################################################

def split_escore_notes_by_time(escore_notes, time_threshold=256):

    dscore = delta_score_notes(escore_notes, timings_clip_value=time_threshold-1)

    score_chunks = []

    ctime = 0
    pchunk_idx = 0

    for i, e in enumerate(dscore):
        
        ctime += e[1]

        if ctime >= time_threshold:
            score_chunks.append(escore_notes[pchunk_idx:i])
            pchunk_idx = i
            ctime = 0

    return score_chunks

###################################################################################

def escore_notes_grouped_patches(escore_notes, time_threshold=256):
    
    split_score_chunks = split_escore_notes_by_time(escore_notes,
                                                    time_threshold=time_threshold
                                                    )

    chunks_patches = []

    for s in split_score_chunks:
        chunks_patches.append(escore_notes_patches(s))

    return chunks_patches

###################################################################################

def computeLPSArray(pattern, M, lps):
    length = 0
    i = 1
    
    lps[0] = 0
    
    while i < M:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
                
###################################################################################

def find_pattern_idxs(sub_pattern, pattern):

    lst = pattern
    pattern = sub_pattern
    
    M = len(pattern)
    N = len(lst)
    
    lps = [0] * M
    j = 0  # index for pattern[]

    computeLPSArray(pattern, M, lps)
    
    i = 0  # index for lst[]
    indexes = []
    
    while i < N:
        if pattern[j] == lst[i]:
            i += 1
            j += 1
        
        if j == M:
            end_index = i - 1
            start_index = end_index - M + 1
            indexes.append((start_index, end_index))
            j = lps[j-1]
        elif i < N and pattern[j] != lst[i]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
                
    return indexes

###################################################################################

def escore_notes_patch_lrno_patterns(escore_notes, 
                                     patch=0, 
                                     zero_score_timings=False,
                                     pitches_idx=4,
                                     patches_idx=6
                                    ):

    patch_escore = [e for e in escore_notes if e[patches_idx] == patch]

    if patch_escore:
        
        patch_cscore = chordify_score([1000, patch_escore])
    
        patch_tscore = []
    
        for c in patch_cscore:
    
            tones_chord = sorted(set([p[pitches_idx] % 12 for p in c]))
    
            if tones_chord not in ALL_CHORDS_SORTED:
                tnoes_chord = check_and_fix_tones_chord(tones_chord)
    
            patch_tscore.append(ALL_CHORDS_SORTED.index(tones_chord))
    
        pattern = find_lrno_pattern_fast(patch_tscore)
    
        patterns_idxs = find_pattern_idxs(pattern, patch_tscore)
    
        patch_lrno_scores = []
    
        for idxs in patterns_idxs:
            
            score = patch_escore[idxs[0]:idxs[1]]
            
            if zero_score_timings:
                score = recalculate_score_timings(score)
                
            patch_lrno_scores.append(score)
    
        return patch_lrno_scores
        
    else:
        return []

###################################################################################

ALL_BASE_CHORDS_SORTED = [[0], [0, 2], [0, 2, 4], [0, 2, 4, 6], [0, 2, 4, 6, 8], [0, 2, 4, 6, 8, 10],
                         [0, 2, 4, 6, 9], [0, 2, 4, 6, 10], [0, 2, 4, 7], [0, 2, 4, 7, 9],
                         [0, 2, 4, 7, 10], [0, 2, 4, 8], [0, 2, 4, 8, 10], [0, 2, 4, 9], [0, 2, 4, 10],
                         [0, 2, 5], [0, 2, 5, 7], [0, 2, 5, 7, 9], [0, 2, 5, 7, 10], [0, 2, 5, 8],
                         [0, 2, 5, 8, 10], [0, 2, 5, 9], [0, 2, 5, 10], [0, 2, 6], [0, 2, 6, 8],
                         [0, 2, 6, 8, 10], [0, 2, 6, 9], [0, 2, 6, 10], [0, 2, 7], [0, 2, 7, 9],
                         [0, 2, 7, 10], [0, 2, 8], [0, 2, 8, 10], [0, 2, 9], [0, 2, 10], [0, 3],
                         [0, 3, 5], [0, 3, 5, 7], [0, 3, 5, 7, 9], [0, 3, 5, 7, 10], [0, 3, 5, 8],
                         [0, 3, 5, 8, 10], [0, 3, 5, 9], [0, 3, 5, 10], [0, 3, 6], [0, 3, 6, 8],
                         [0, 3, 6, 8, 10], [0, 3, 6, 9], [0, 3, 6, 10], [0, 3, 7], [0, 3, 7, 9],
                         [0, 3, 7, 10], [0, 3, 8], [0, 3, 8, 10], [0, 3, 9], [0, 3, 10], [0, 4],
                         [0, 4, 6], [0, 4, 6, 8], [0, 4, 6, 8, 10], [0, 4, 6, 9], [0, 4, 6, 10],
                         [0, 4, 7], [0, 4, 7, 9], [0, 4, 7, 10], [0, 4, 8], [0, 4, 8, 10], [0, 4, 9],
                         [0, 4, 10], [0, 5], [0, 5, 7], [0, 5, 7, 9], [0, 5, 7, 10], [0, 5, 8],
                         [0, 5, 8, 10], [0, 5, 9], [0, 5, 10], [0, 6], [0, 6, 8], [0, 6, 8, 10],
                         [0, 6, 9], [0, 6, 10], [0, 7], [0, 7, 9], [0, 7, 10], [0, 8], [0, 8, 10],
                         [0, 9], [0, 10]]

###################################################################################

MAJOR_SCALE_CHORDS_COUNTS = [[317, 6610], [320, 6468], [267, 6460], [89, 6329], [301, 6228], [178, 6201],
                             [0, 5822], [314, 5805], [309, 5677], [319, 5545], [288, 5494], [233, 5395],
                             [112, 2232], [194, 1956], [127, 1935], [216, 1884], [256, 1871], [283, 1815],
                             [201, 1768], [16, 1756], [105, 1743], [38, 1727], [23, 1718], [249, 1386],
                             [272, 796], [91, 770], [191, 740], [303, 735], [181, 718], [306, 717],
                             [235, 703], [183, 690], [94, 686], [13, 686], [269, 677], [280, 675],
                             [102, 665], [92, 662], [293, 659], [212, 658], [114, 656], [37, 653],
                             [180, 651], [215, 644], [316, 640], [290, 636], [5, 636], [110, 625],
                             [270, 625], [3, 624], [238, 615], [123, 609], [34, 591], [254, 584],
                             [258, 571], [126, 567], [2, 559], [246, 556], [104, 556], [203, 550],
                             [291, 537], [311, 522], [304, 520], [193, 509], [236, 496], [199, 493],
                             [15, 468], [25, 452], [312, 444], [282, 443], [248, 433], [21, 408],
                             [268, 281], [179, 273], [144, 259], [90, 252], [162, 250], [234, 250],
                             [1, 246], [221, 214], [73, 213], [43, 213], [45, 213], [134, 212], [318, 210],
                             [119, 210], [159, 209], [120, 209], [302, 207], [310, 201], [289, 195],
                             [42, 193], [264, 193], [220, 185], [131, 183], [55, 180], [315, 180],
                             [132, 176], [30, 174], [31, 172], [209, 171], [227, 169], [217, 163],
                             [223, 159], [70, 158], [39, 157], [36, 153], [214, 142], [196, 141],
                             [285, 141], [8, 137], [208, 133], [125, 133], [147, 130], [186, 130],
                             [97, 130], [49, 130], [58, 130], [128, 130], [138, 128], [241, 125],
                             [228, 124], [263, 120], [251, 120], [275, 119], [296, 118], [259, 116],
                             [99, 114], [10, 113], [50, 111], [273, 111], [139, 111], [298, 106], [18, 105],
                             [153, 105], [7, 101], [277, 101], [243, 99], [96, 99], [9, 96], [160, 96],
                             [188, 95], [115, 94], [24, 93], [107, 92], [204, 90], [150, 90], [148, 84],
                             [202, 83], [213, 82], [187, 82], [35, 80], [113, 79], [98, 78], [239, 77],
                             [59, 77], [26, 76], [281, 76], [184, 75], [64, 75], [124, 75], [71, 75],
                             [257, 75], [95, 74], [294, 73], [192, 70], [247, 70], [61, 67], [307, 66],
                             [242, 65], [218, 65], [146, 64], [276, 63], [6, 63], [68, 60], [284, 59],
                             [103, 59], [297, 56], [14, 56], [185, 55], [57, 55], [40, 55], [129, 54],
                             [274, 52], [308, 52], [46, 51], [224, 49], [240, 47], [135, 46], [17, 45],
                             [295, 45], [106, 45], [48, 44], [157, 44], [206, 43], [195, 42], [158, 42],
                             [69, 41], [117, 41], [225, 40], [222, 37], [226, 35], [261, 34], [164, 32],
                             [75, 32], [28, 32], [11, 32], [250, 31], [44, 30], [137, 28], [47, 26],
                             [133, 26], [255, 25], [182, 24], [136, 24], [197, 23], [93, 23], [237, 22],
                             [287, 22], [165, 22], [79, 21], [271, 21], [109, 21], [253, 20], [76, 20],
                             [168, 19], [155, 19], [149, 19], [108, 19], [4, 18], [51, 18], [292, 18],
                             [198, 18], [41, 17], [286, 17], [19, 17], [219, 17], [173, 17], [66, 16],
                             [54, 16], [229, 16], [140, 16], [175, 15], [171, 15], [82, 15], [130, 15],
                             [20, 15], [230, 15], [244, 14], [145, 14], [84, 14], [305, 14], [278, 14],
                             [86, 13], [60, 13], [232, 12], [100, 12], [141, 12], [52, 12], [189, 12],
                             [252, 12], [56, 11], [53, 11], [143, 10], [151, 10], [154, 10], [163, 9],
                             [116, 9], [27, 9], [65, 9], [313, 9], [205, 9], [170, 8], [62, 8], [299, 7],
                             [142, 7], [231, 7], [156, 6], [22, 6], [63, 6], [152, 6], [77, 5], [67, 5],
                             [166, 5], [174, 5], [85, 4], [72, 4], [190, 4], [111, 4], [101, 4], [200, 4],
                             [12, 4], [245, 3], [300, 3], [279, 3], [81, 2], [210, 2], [32, 2], [265, 2],
                             [260, 2], [74, 2], [161, 1], [207, 1], [29, 1], [118, 1], [262, 1], [121, 1]]

###################################################################################

MINOR_SCALE_CHORDS_COUNTS = [[267, 10606], [89, 10562], [301, 10522], [320, 10192], [178, 10191],
                             [317, 10153], [233, 10101], [314, 10065], [288, 9914], [0, 9884], [309, 9694],
                             [319, 9648], [114, 1963], [193, 1778], [25, 1705], [104, 1689], [248, 1671],
                             [282, 1614], [283, 1610], [127, 1530], [203, 1525], [37, 1508], [215, 1473],
                             [105, 1465], [38, 1462], [258, 1445], [112, 1419], [94, 1413], [280, 1391],
                             [194, 1388], [126, 1384], [16, 1374], [272, 1370], [23, 1364], [238, 1351],
                             [306, 1342], [303, 1340], [5, 1338], [183, 1334], [102, 1333], [290, 1322],
                             [269, 1312], [191, 1311], [249, 1305], [15, 1291], [246, 1290], [316, 1288],
                             [13, 1279], [216, 1278], [235, 1275], [256, 1268], [311, 1241], [293, 1228],
                             [91, 1219], [180, 1173], [34, 1167], [2, 1138], [212, 1131], [123, 1118],
                             [201, 1103], [270, 1017], [304, 961], [181, 958], [92, 943], [3, 940],
                             [236, 932], [254, 923], [291, 921], [110, 920], [21, 911], [312, 891],
                             [199, 832], [268, 431], [179, 395], [234, 395], [302, 385], [144, 368],
                             [90, 365], [289, 362], [310, 352], [318, 350], [1, 332], [55, 323], [315, 322],
                             [8, 307], [162, 304], [97, 302], [186, 302], [241, 300], [10, 299], [217, 289],
                             [275, 275], [128, 267], [73, 266], [243, 265], [125, 262], [296, 259],
                             [298, 251], [36, 250], [39, 250], [99, 249], [214, 231], [119, 230],
                             [120, 227], [188, 227], [159, 226], [264, 225], [263, 225], [138, 223],
                             [31, 222], [227, 219], [134, 216], [277, 214], [70, 210], [209, 207],
                             [30, 203], [49, 186], [46, 185], [45, 184], [221, 172], [281, 170], [96, 169],
                             [131, 169], [224, 165], [148, 159], [59, 157], [43, 157], [7, 157], [247, 155],
                             [208, 153], [132, 152], [274, 150], [223, 149], [135, 148], [273, 148],
                             [240, 137], [220, 132], [185, 131], [239, 131], [42, 130], [147, 119],
                             [213, 117], [307, 115], [24, 112], [95, 108], [192, 107], [150, 106],
                             [294, 105], [106, 104], [58, 102], [103, 102], [17, 100], [129, 100], [61, 99],
                             [9, 98], [139, 96], [295, 96], [284, 96], [146, 96], [218, 95], [184, 94],
                             [308, 87], [195, 87], [40, 86], [14, 85], [50, 82], [250, 82], [285, 81],
                             [57, 79], [259, 79], [6, 79], [276, 78], [228, 78], [35, 76], [187, 75],
                             [242, 73], [206, 73], [160, 72], [113, 72], [117, 72], [261, 72], [98, 71],
                             [202, 70], [115, 70], [158, 69], [71, 68], [48, 67], [28, 67], [204, 66],
                             [157, 64], [124, 63], [257, 59], [196, 59], [69, 59], [68, 57], [251, 55],
                             [225, 50], [137, 50], [107, 49], [165, 49], [297, 48], [64, 46], [153, 45],
                             [226, 44], [198, 44], [287, 43], [26, 43], [219, 41], [253, 40], [109, 40],
                             [66, 39], [47, 39], [41, 39], [76, 38], [11, 38], [136, 38], [130, 36],
                             [155, 35], [18, 31], [93, 31], [20, 30], [271, 29], [4, 28], [292, 28],
                             [237, 27], [182, 26], [62, 26], [164, 25], [151, 25], [108, 25], [286, 24],
                             [145, 24], [305, 24], [75, 24], [56, 23], [149, 23], [252, 23], [197, 23],
                             [255, 23], [313, 21], [60, 18], [244, 17], [278, 17], [189, 17], [100, 16],
                             [299, 15], [200, 13], [175, 13], [111, 13], [22, 13], [170, 12], [232, 11],
                             [86, 11], [141, 11], [52, 11], [65, 10], [173, 10], [133, 10], [222, 10],
                             [143, 10], [154, 9], [82, 8], [19, 8], [85, 8], [44, 8], [84, 8], [163, 7],
                             [205, 7], [230, 7], [54, 7], [174, 7], [116, 7], [27, 7], [171, 7], [229, 6],
                             [81, 5], [79, 4], [142, 4], [231, 4], [210, 3], [168, 3], [53, 3], [51, 3],
                             [74, 3], [265, 3], [260, 3], [152, 2], [245, 2], [279, 2], [190, 2], [12, 2],
                             [101, 2], [262, 1], [63, 1], [72, 1], [207, 1], [166, 1], [83, 1], [176, 1],
                             [118, 1], [67, 1], [172, 1], [29, 1], [121, 1], [77, 1], [266, 1], [156, 1],
                             [211, 1], [300, 1], [87, 1], [140, 1], [161, 1]]

###################################################################################

def get_weighted_score(src_order, trg_order):
    
    score = 0
    
    for i, (item, count) in enumerate(src_order):
        if item in trg_order:
            score += count * abs(i - trg_order.index(item))
            
        else:
            score += count * len(trg_order)
            
    return score

###################################################################################

def escore_notes_scale(escore_notes,
                       score_mult_factor=3,
                       start_note=0,
                       num_notes=-1,
                       return_scale_indexes=False
                      ):

    trg_chords = []

    for i in range(-score_mult_factor, score_mult_factor):

        trans_escore_notes = transpose_escore_notes(escore_notes[start_note:start_note+num_notes], i)
    
        cscore = chordify_score([1000, trans_escore_notes])
    
        tones_chords = []
    
        for c in cscore:
    
            seen = []
            pitches = []
            
            for e in c:
                
                if e[4] not in seen:
                    pitches.append(e[4])
                    seen.append(e[4])
                    
            if pitches:
                
                tones_chord = sorted(set([p % 12 for p in pitches]))
    
                if tones_chord not in ALL_CHORDS_SORTED:
                    tones_chord = check_and_fix_tones_chord(tones_chord)
                
                tones_chords.append(ALL_CHORDS_SORTED.index(tones_chord))
                
        if tones_chords:
            trg_chords.extend(tones_chords)
    
    #========================================================================
            
    scales_results = []
    
    #========================================================================
    
    if trg_chords:
    
        #========================================================================

        src_order = Counter(trg_chords).most_common()       
       
        trg1_items = [item for item, count in MAJOR_SCALE_CHORDS_COUNTS]
        trg2_items = [item for item, count in MINOR_SCALE_CHORDS_COUNTS]
        

        trg1_score = get_weighted_score(src_order, trg1_items)
        trg2_score = get_weighted_score(src_order, trg2_items)

        #========================================================================
        
        if trg1_score <= trg2_score:
            
            if return_scale_indexes:
                scales_results.append(1)

            else:
                scales_results.append('Major')
            
        else:
            if return_scale_indexes:
                scales_results.append(0)
                
            else:
                scales_results.append('Minor')
            
        #========================================================================
                
        best_match = None
        best_score = float('inf')
        
        for trg_order in ALL_MOOD_TYPES:

            trg_items = [item for item, count in trg_order]

            trg_score = get_weighted_score(src_order, trg_items)

            if trg_score < best_score:
                best_score = trg_score

                if return_scale_indexes:
                    best_match = ALL_MOOD_TYPES.index(trg_order)

                else:
                    best_match = ALL_MOOD_TYPES_LABELS[ALL_MOOD_TYPES.index(trg_order)]
                    
        scales_results.append(best_match)
            
    else:
        if return_scale_indexes:
            scales_results.extend([-1, -1])
            
        else:
            scales_results.extend(['Unknown', 'Unknown'])
            
    return scales_results
        
###################################################################################

HAPPY_MAJOR = [(317, 1916), (89, 1876), (320, 1840), (267, 1817), (301, 1795), (178, 1750),
             (314, 1725), (0, 1691), (319, 1658), (288, 1624), (309, 1599), (233, 1559),
             (112, 1050), (127, 972), (201, 884), (194, 879), (216, 860), (38, 831),
             (256, 828), (23, 822), (105, 820), (283, 756), (16, 734), (249, 622),
             (91, 254), (303, 242), (34, 237), (316, 235), (110, 235), (123, 234),
             (212, 230), (92, 225), (181, 225), (114, 219), (272, 218), (290, 213),
             (235, 208), (180, 207), (269, 206), (2, 201), (3, 199), (203, 198), (37, 195),
             (254, 191), (199, 189), (311, 189), (293, 187), (5, 186), (270, 185),
             (183, 184), (291, 183), (94, 183), (25, 182), (304, 181), (258, 176),
             (215, 173), (191, 172), (193, 168), (104, 167), (282, 164), (238, 162),
             (248, 157), (15, 156), (13, 156), (126, 153), (21, 150), (102, 150),
             (306, 150), (312, 144), (280, 141), (236, 139), (162, 116), (120, 114),
             (246, 113), (134, 109), (43, 108), (221, 105), (264, 103), (73, 100),
             (159, 98), (42, 95), (45, 94), (220, 93), (131, 91), (119, 91), (227, 90),
             (209, 88), (70, 86), (144, 86), (31, 85), (223, 84), (58, 82), (1, 80),
             (132, 79), (30, 76), (90, 75), (268, 75), (259, 74), (234, 72), (179, 72),
             (147, 70), (318, 69), (208, 67), (315, 66), (55, 66), (49, 64), (310, 63),
             (138, 62), (214, 61), (263, 60), (204, 59), (302, 58), (196, 58), (115, 56),
             (107, 53), (18, 53), (153, 52), (289, 52), (9, 50), (10, 50), (217, 49),
             (243, 48), (39, 48), (99, 48), (7, 47), (188, 46), (26, 46), (68, 46),
             (36, 45), (125, 43), (202, 43), (285, 42), (24, 42), (277, 41), (98, 40),
             (251, 39), (113, 39), (8, 38), (128, 38), (187, 37), (35, 36), (213, 36),
             (97, 35), (186, 35), (61, 34), (150, 34), (160, 33), (124, 32), (96, 32),
             (257, 32), (275, 31), (241, 31), (296, 30), (64, 30), (297, 29), (298, 29),
             (117, 29), (46, 28), (273, 28), (206, 28), (157, 27), (242, 26), (224, 26),
             (185, 26), (222, 26), (59, 25), (135, 24), (158, 23), (28, 23), (294, 22),
             (69, 22), (276, 21), (274, 21), (225, 21), (148, 20), (50, 20), (48, 20),
             (281, 19), (139, 19), (307, 19), (228, 19), (75, 18), (164, 18), (44, 18),
             (133, 18), (79, 17), (184, 17), (57, 17), (240, 17), (239, 17), (295, 17),
             (247, 16), (95, 16), (261, 15), (308, 15), (287, 14), (76, 14), (165, 14),
             (175, 14), (82, 14), (284, 14), (71, 14), (253, 12), (155, 12), (86, 12),
             (4, 12), (93, 12), (171, 12), (137, 12), (66, 11), (232, 11), (168, 11),
             (103, 11), (192, 11), (54, 10), (145, 10), (40, 10), (51, 10), (182, 10),
             (226, 10), (14, 10), (129, 9), (218, 9), (146, 9), (237, 9), (19, 9), (108, 9),
             (197, 9), (140, 8), (229, 8), (6, 7), (17, 7), (56, 6), (106, 6), (271, 6),
             (109, 6), (163, 5), (143, 5), (65, 5), (154, 5), (27, 5), (116, 5), (205, 5),
             (195, 5), (250, 5), (198, 5), (41, 5), (136, 5), (47, 4), (52, 4), (141, 4),
             (230, 4), (84, 4), (173, 4), (255, 4), (11, 4), (100, 4), (189, 4), (244, 4),
             (278, 4), (219, 3), (20, 3), (286, 3), (130, 3), (170, 3), (151, 3), (53, 2),
             (77, 2), (166, 2), (67, 2), (156, 2), (63, 2), (60, 2), (292, 2), (62, 2),
             (142, 1), (231, 1), (85, 1), (174, 1), (81, 1), (152, 1), (262, 1), (72, 1),
             (161, 1), (29, 1), (118, 1), (207, 1), (149, 1), (300, 1), (299, 1), (252, 1)]

###################################################################################

MELANCHOLIC_MAJOR = [(317, 451), (301, 430), (89, 426), (320, 419), (267, 416), (178, 415),
                     (314, 401), (319, 400), (0, 394), (309, 390), (288, 389), (233, 365),
                     (37, 224), (215, 207), (258, 203), (126, 191), (114, 185), (203, 183),
                     (283, 141), (127, 131), (38, 127), (216, 115), (194, 113), (112, 112),
                     (23, 109), (105, 105), (249, 103), (16, 99), (306, 96), (256, 92), (13, 87),
                     (280, 86), (181, 86), (102, 85), (92, 84), (104, 84), (15, 84), (191, 83),
                     (246, 83), (270, 81), (94, 74), (3, 73), (238, 72), (272, 72), (236, 72),
                     (201, 72), (183, 70), (293, 66), (193, 63), (254, 63), (212, 61), (282, 60),
                     (123, 58), (5, 57), (25, 55), (291, 53), (34, 52), (316, 50), (304, 48),
                     (91, 47), (2, 47), (110, 46), (248, 45), (303, 38), (311, 38), (45, 36),
                     (180, 35), (199, 34), (235, 33), (162, 33), (221, 33), (21, 32), (144, 32),
                     (132, 31), (179, 29), (90, 29), (43, 29), (217, 29), (312, 28), (39, 28),
                     (128, 28), (302, 27), (268, 27), (36, 27), (125, 27), (269, 26), (134, 26),
                     (234, 26), (73, 25), (318, 25), (55, 25), (1, 24), (290, 23), (8, 22),
                     (310, 22), (315, 22), (97, 20), (186, 20), (241, 20), (275, 20), (296, 20),
                     (289, 20), (119, 18), (298, 18), (31, 17), (6, 17), (95, 17), (184, 17),
                     (273, 17), (223, 16), (276, 15), (120, 15), (239, 15), (30, 15), (208, 14),
                     (59, 14), (159, 13), (146, 13), (42, 13), (209, 13), (26, 13), (264, 13),
                     (147, 13), (187, 13), (242, 13), (115, 12), (220, 12), (70, 12), (226, 12),
                     (47, 12), (148, 12), (24, 11), (49, 11), (131, 10), (227, 10), (214, 10),
                     (136, 9), (225, 9), (69, 9), (138, 9), (158, 9), (106, 9), (98, 9), (257, 8),
                     (263, 8), (297, 8), (50, 8), (204, 8), (259, 8), (7, 8), (294, 8), (281, 8),
                     (9, 8), (113, 7), (202, 7), (17, 7), (124, 7), (213, 7), (57, 7), (96, 7),
                     (247, 7), (285, 6), (185, 6), (130, 6), (219, 6), (218, 6), (58, 6), (139, 5),
                     (35, 5), (240, 5), (195, 5), (250, 5), (20, 5), (284, 5), (150, 5), (261, 5),
                     (48, 5), (107, 4), (196, 4), (251, 4), (292, 4), (41, 4), (228, 4), (61, 4),
                     (71, 4), (160, 4), (109, 4), (103, 4), (192, 4), (206, 4), (137, 4), (274, 3),
                     (18, 3), (305, 3), (295, 3), (93, 3), (308, 3), (182, 3), (237, 3), (271, 3),
                     (198, 3), (168, 3), (51, 3), (140, 3), (229, 3), (54, 3), (155, 3), (10, 3),
                     (99, 3), (157, 2), (64, 2), (143, 2), (224, 2), (253, 2), (307, 2), (66, 2),
                     (40, 2), (129, 2), (188, 2), (11, 2), (243, 2), (28, 1), (117, 1), (4, 1),
                     (313, 1), (62, 1), (151, 1), (56, 1), (135, 1), (46, 1), (165, 1), (79, 1),
                     (299, 1), (60, 1), (149, 1), (22, 1), (111, 1), (200, 1)]

###################################################################################

MELANCHOLIC_MINOR = [(89, 3681), (267, 3628), (317, 3472), (301, 3408), (320, 3290), (178, 3261),
                     (314, 3261), (288, 3206), (0, 3140), (233, 3050), (319, 2894), (309, 2841),
                     (114, 570), (283, 559), (104, 544), (193, 529), (215, 509), (37, 507),
                     (127, 482), (126, 468), (38, 456), (282, 432), (248, 417), (25, 415),
                     (194, 414), (216, 412), (112, 411), (258, 407), (23, 403), (105, 399),
                     (249, 399), (303, 387), (203, 386), (15, 366), (256, 356), (16, 351),
                     (290, 343), (316, 343), (269, 332), (235, 323), (91, 312), (311, 296),
                     (272, 286), (34, 273), (94, 271), (180, 269), (212, 265), (123, 260),
                     (306, 259), (270, 254), (102, 246), (201, 246), (238, 246), (280, 242),
                     (110, 236), (183, 236), (191, 232), (293, 230), (5, 228), (2, 228), (291, 226),
                     (304, 225), (13, 219), (312, 207), (21, 207), (181, 203), (92, 195),
                     (246, 192), (3, 191), (254, 181), (236, 173), (199, 155), (268, 124),
                     (179, 114), (144, 103), (90, 103), (302, 102), (318, 101), (234, 99),
                     (289, 86), (1, 84), (310, 83), (31, 79), (120, 79), (55, 78), (315, 72),
                     (162, 72), (264, 71), (73, 70), (209, 69), (159, 61), (227, 61), (263, 60),
                     (49, 58), (138, 57), (119, 51), (273, 49), (70, 49), (10, 47), (8, 44),
                     (97, 44), (186, 44), (241, 44), (275, 44), (99, 44), (146, 43), (239, 42),
                     (296, 39), (214, 39), (217, 39), (95, 38), (148, 37), (36, 36), (281, 34),
                     (307, 33), (125, 33), (218, 32), (59, 31), (134, 31), (160, 31), (184, 31),
                     (129, 29), (208, 29), (223, 29), (71, 29), (30, 29), (96, 27), (147, 27),
                     (228, 27), (57, 27), (6, 27), (284, 26), (50, 26), (139, 26), (247, 24),
                     (24, 24), (250, 24), (115, 24), (204, 24), (259, 24), (9, 23), (240, 23),
                     (274, 23), (220, 23), (58, 23), (103, 22), (40, 22), (131, 22), (243, 22),
                     (106, 22), (285, 22), (46, 22), (295, 21), (308, 21), (221, 21), (14, 20),
                     (45, 20), (42, 20), (195, 20), (294, 19), (188, 19), (277, 19), (185, 18),
                     (192, 18), (17, 18), (135, 18), (224, 18), (7, 17), (61, 17), (150, 16),
                     (225, 14), (69, 14), (158, 14), (128, 14), (257, 14), (149, 13), (64, 13),
                     (298, 13), (39, 13), (213, 12), (113, 12), (43, 11), (132, 11), (28, 11),
                     (35, 10), (124, 10), (47, 10), (136, 10), (41, 10), (130, 10), (157, 10),
                     (202, 10), (165, 10), (66, 9), (155, 9), (219, 9), (153, 9), (18, 9), (255, 9),
                     (11, 9), (60, 8), (22, 8), (111, 8), (107, 8), (299, 7), (143, 7), (232, 7),
                     (86, 7), (175, 7), (276, 6), (313, 6), (56, 6), (62, 6), (278, 6), (151, 6),
                     (26, 6), (117, 6), (206, 6), (196, 6), (98, 5), (187, 5), (242, 5), (200, 5),
                     (109, 5), (198, 5), (229, 5), (54, 5), (305, 5), (261, 5), (48, 5), (76, 5),
                     (226, 5), (145, 4), (20, 4), (251, 4), (68, 4), (292, 4), (253, 4), (287, 4),
                     (244, 3), (4, 3), (189, 3), (93, 2), (182, 2), (237, 2), (297, 2), (100, 2),
                     (173, 2), (53, 2), (142, 2), (231, 2), (85, 2), (174, 2), (271, 2), (137, 2),
                     (82, 2), (171, 2), (164, 1), (44, 1), (133, 1), (222, 1), (163, 1), (65, 1),
                     (154, 1), (27, 1), (116, 1), (205, 1)]

###################################################################################

NEUTRAL_MAJOR = [(320, 574), (89, 542), (0, 535), (317, 488), (319, 458), (314, 439),
                 (178, 424), (267, 405), (233, 375), (301, 330), (309, 321), (288, 287),
                 (283, 77), (112, 76), (38, 71), (23, 67), (216, 61), (127, 59), (291, 54),
                 (316, 52), (269, 51), (290, 51), (34, 50), (303, 50), (110, 49), (280, 47),
                 (13, 45), (311, 44), (306, 43), (238, 43), (272, 43), (3, 42), (21, 42),
                 (16, 41), (270, 41), (183, 39), (102, 39), (92, 39), (312, 37), (105, 37),
                 (194, 37), (199, 35), (191, 35), (246, 35), (5, 35), (181, 34), (304, 34),
                 (94, 33), (293, 31), (91, 29), (268, 27), (236, 27), (256, 27), (144, 24),
                 (90, 24), (179, 23), (234, 23), (302, 23), (235, 23), (2, 23), (318, 22),
                 (1, 22), (254, 22), (123, 22), (315, 22), (212, 22), (249, 22), (8, 21),
                 (97, 21), (186, 21), (241, 21), (289, 21), (180, 21), (310, 21), (201, 21),
                 (104, 20), (214, 19), (55, 18), (296, 17), (275, 17), (36, 17), (125, 17),
                 (193, 16), (58, 16), (147, 16), (10, 15), (37, 14), (215, 14), (15, 14),
                 (25, 14), (114, 14), (217, 13), (282, 12), (259, 12), (9, 12), (98, 12),
                 (187, 12), (99, 11), (126, 10), (248, 10), (188, 10), (243, 10), (277, 10),
                 (264, 10), (96, 10), (73, 10), (162, 10), (43, 10), (128, 10), (203, 8),
                 (150, 8), (221, 8), (39, 8), (24, 8), (113, 8), (274, 6), (295, 6), (308, 6),
                 (159, 6), (258, 6), (120, 6), (42, 6), (131, 6), (220, 6), (30, 6), (132, 6),
                 (7, 6), (298, 6), (119, 6), (228, 4), (185, 4), (71, 4), (240, 4), (160, 4),
                 (153, 4), (18, 4), (61, 4), (35, 4), (285, 4), (209, 4), (95, 4), (307, 4),
                 (146, 4), (184, 4), (239, 4), (202, 4), (247, 4), (273, 4), (257, 4), (281, 4),
                 (64, 2), (156, 2), (50, 2), (63, 2), (45, 2), (139, 2), (152, 2), (134, 2),
                 (124, 2), (107, 2), (12, 2), (11, 2), (223, 2), (213, 2), (196, 2), (101, 2),
                 (31, 2), (251, 2), (190, 2), (106, 2), (40, 2), (195, 2), (6, 2), (129, 2),
                 (250, 2), (218, 2), (284, 2), (294, 2), (57, 2), (59, 2), (148, 2)]

###################################################################################

NEUTRAL_MINOR = [(317, 530), (301, 499), (267, 454), (309, 438), (314, 422), (288, 420),
                 (178, 415), (320, 414), (89, 399), (319, 383), (0, 341), (233, 307),
                 (215, 133), (37, 127), (212, 123), (193, 121), (123, 121), (34, 119),
                 (191, 117), (126, 115), (104, 108), (112, 107), (272, 105), (23, 102),
                 (15, 96), (127, 92), (38, 87), (283, 85), (102, 84), (91, 83), (94, 83),
                 (306, 82), (216, 80), (2, 80), (280, 79), (293, 78), (5, 78), (13, 77),
                 (183, 76), (114, 74), (316, 69), (105, 68), (180, 64), (201, 62), (256, 58),
                 (16, 56), (246, 55), (203, 55), (303, 52), (194, 52), (282, 49), (311, 49),
                 (248, 47), (238, 43), (258, 41), (249, 39), (7, 32), (10, 29), (96, 29),
                 (25, 28), (125, 27), (214, 27), (36, 26), (134, 23), (99, 22), (310, 22),
                 (270, 21), (291, 20), (223, 20), (302, 20), (213, 19), (185, 19), (217, 19),
                 (3, 19), (221, 19), (45, 18), (268, 16), (289, 16), (235, 15), (179, 14),
                 (234, 14), (181, 14), (312, 13), (240, 13), (21, 13), (274, 13), (110, 13),
                 (92, 13), (236, 13), (31, 13), (120, 13), (304, 12), (269, 11), (113, 11),
                 (150, 10), (43, 10), (132, 10), (68, 9), (157, 9), (202, 9), (55, 9), (144, 9),
                 (315, 9), (318, 9), (42, 9), (131, 9), (188, 8), (70, 8), (159, 8), (241, 7),
                 (275, 7), (296, 7), (8, 7), (290, 7), (97, 7), (186, 7), (24, 7), (119, 7),
                 (227, 7), (254, 6), (219, 6), (35, 6), (273, 6), (124, 6), (294, 6), (247, 6),
                 (220, 6), (281, 6), (208, 6), (46, 6), (61, 6), (243, 5), (199, 5), (128, 5),
                 (30, 5), (11, 5), (218, 5), (192, 5), (162, 5), (257, 5), (138, 5), (264, 5),
                 (148, 4), (41, 4), (130, 4), (39, 4), (307, 4), (40, 4), (129, 4), (17, 4),
                 (106, 4), (195, 4), (224, 4), (135, 4), (209, 4), (276, 3), (297, 3), (26, 3),
                 (115, 3), (277, 3), (20, 3), (109, 3), (198, 3), (6, 3), (298, 3), (95, 3),
                 (184, 3), (1, 3), (165, 3), (66, 3), (155, 3), (73, 3), (69, 3), (158, 3),
                 (71, 3), (160, 3), (64, 3), (153, 3), (18, 3), (107, 3), (187, 2), (242, 2),
                 (59, 2), (239, 2), (226, 2), (163, 2), (14, 2), (65, 2), (263, 2), (103, 2),
                 (154, 2), (49, 2), (27, 2), (253, 2), (116, 2), (287, 2), (205, 2), (4, 1),
                 (93, 1), (182, 1), (237, 1), (271, 1), (292, 1), (222, 1), (19, 1), (108, 1),
                 (197, 1), (57, 1), (146, 1), (143, 1), (211, 1), (232, 1), (266, 1), (47, 1),
                 (86, 1), (87, 1), (136, 1), (175, 1), (176, 1), (225, 1), (82, 1), (83, 1),
                 (171, 1), (172, 1), (117, 1), (206, 1), (261, 1), (48, 1), (137, 1), (90, 1),
                 (204, 1), (250, 1), (259, 1), (284, 1)]

###################################################################################

SAD_MAJOR = [(267, 46), (301, 45), (178, 43), (89, 37), (288, 35), (233, 35), (215, 34),
             (317, 32), (320, 32), (309, 30), (314, 24), (0, 22), (319, 21), (114, 19),
             (203, 19), (258, 19), (37, 19), (193, 18), (126, 18), (15, 17), (104, 17),
             (248, 16), (282, 16), (112, 13), (134, 13), (105, 10), (221, 10), (194, 10),
             (45, 10), (162, 8), (43, 8), (201, 8), (132, 8), (256, 8), (16, 8), (127, 7),
             (283, 6), (38, 6), (306, 5), (223, 5), (216, 5), (31, 5), (23, 5), (120, 5),
             (272, 4), (123, 4), (293, 4), (119, 3), (181, 3), (125, 3), (94, 3), (236, 3),
             (212, 3), (183, 3), (270, 3), (2, 3), (238, 3), (291, 3), (91, 3), (304, 3),
             (209, 3), (312, 3), (264, 3), (163, 2), (148, 2), (157, 2), (316, 2), (217, 2),
             (13, 2), (65, 2), (208, 2), (7, 2), (214, 2), (34, 2), (36, 2), (102, 2),
             (154, 2), (249, 2), (263, 2), (96, 2), (10, 2), (191, 2), (27, 2), (49, 2),
             (99, 2), (116, 2), (138, 2), (180, 2), (205, 2), (227, 2), (235, 2), (226, 1),
             (298, 1), (307, 1), (213, 1), (159, 1), (292, 1), (144, 1), (147, 1), (290, 1),
             (47, 1), (39, 1), (40, 1), (42, 1), (305, 1), (68, 1), (1, 1), (9, 1),
             (303, 1), (136, 1), (128, 1), (129, 1), (131, 1), (313, 1), (90, 1), (98, 1),
             (311, 1), (225, 1), (218, 1), (185, 1), (220, 1), (62, 1), (179, 1), (187, 1),
             (59, 1), (246, 1), (69, 1), (57, 1), (247, 1), (240, 1), (30, 1), (151, 1),
             (188, 1), (239, 1), (234, 1), (242, 1), (280, 1), (158, 1), (146, 1), (281, 1),
             (274, 1), (56, 1), (243, 1), (273, 1), (268, 1), (276, 1)]

###################################################################################

SAD_MINOR = [(178, 1800), (267, 1764), (233, 1727), (309, 1671), (288, 1644), (0, 1610),
             (301, 1580), (320, 1532), (89, 1512), (317, 1454), (319, 1417), (314, 1383),
             (272, 238), (269, 232), (183, 230), (180, 224), (212, 219), (34, 217),
             (238, 217), (311, 214), (2, 212), (5, 210), (303, 208), (293, 206), (91, 202),
             (94, 202), (235, 200), (13, 199), (290, 198), (316, 192), (3, 190), (306, 188),
             (280, 187), (193, 185), (291, 184), (123, 183), (191, 182), (37, 179),
             (199, 172), (102, 169), (181, 164), (110, 163), (92, 163), (246, 161),
             (21, 157), (236, 156), (312, 154), (270, 146), (203, 146), (15, 144),
             (126, 135), (25, 135), (114, 135), (304, 132), (215, 131), (104, 131),
             (254, 130), (38, 124), (112, 124), (282, 123), (216, 114), (23, 111),
             (127, 102), (201, 101), (16, 100), (283, 96), (248, 96), (289, 92), (268, 92),
             (194, 92), (258, 91), (310, 87), (105, 86), (302, 81), (179, 77), (234, 77),
             (249, 76), (256, 76), (318, 60), (315, 57), (1, 53), (8, 49), (186, 47),
             (90, 47), (97, 47), (224, 47), (55, 46), (241, 46), (275, 46), (296, 45),
             (45, 43), (144, 42), (46, 38), (274, 37), (42, 36), (135, 36), (134, 34),
             (217, 31), (214, 30), (59, 30), (61, 30), (240, 28), (148, 28), (70, 28),
             (159, 28), (73, 27), (49, 27), (277, 26), (295, 26), (308, 26), (138, 26),
             (227, 26), (223, 25), (10, 25), (120, 25), (221, 24), (31, 24), (128, 24),
             (185, 23), (39, 23), (99, 23), (36, 23), (150, 21), (243, 21), (162, 21),
             (7, 20), (206, 18), (298, 18), (96, 18), (125, 18), (284, 16), (198, 16),
             (209, 16), (264, 16), (43, 16), (14, 15), (213, 15), (132, 15), (158, 14),
             (28, 14), (188, 13), (117, 13), (35, 13), (253, 12), (103, 12), (192, 12),
             (220, 12), (30, 12), (225, 11), (69, 11), (287, 11), (131, 11), (24, 10),
             (119, 10), (208, 10), (261, 9), (48, 9), (76, 9), (165, 9), (9, 9), (66, 9),
             (4, 9), (195, 8), (250, 8), (58, 8), (147, 8), (247, 8), (281, 8), (47, 7),
             (219, 7), (20, 7), (109, 7), (56, 7), (242, 6), (204, 6), (259, 6), (137, 6),
             (226, 6), (292, 6), (93, 6), (62, 6), (98, 6), (151, 6), (187, 5), (115, 5),
             (273, 5), (294, 5), (17, 5), (130, 5), (106, 5), (145, 5), (313, 5), (182, 5),
             (239, 5), (237, 5), (276, 4), (6, 4), (41, 4), (57, 4), (113, 4), (124, 4),
             (146, 4), (271, 4), (18, 4), (297, 3), (40, 3), (129, 3), (19, 3), (68, 3),
             (95, 3), (108, 3), (157, 3), (184, 3), (197, 3), (232, 3), (86, 3), (175, 3),
             (82, 3), (228, 3), (71, 3), (160, 3), (64, 3), (153, 3), (26, 2), (307, 2),
             (60, 2), (218, 2), (222, 2), (305, 2), (202, 2), (263, 2), (11, 2), (136, 2),
             (171, 2), (79, 2), (244, 1), (278, 1), (299, 1), (149, 1), (22, 1), (257, 1),
             (252, 1), (286, 1), (75, 1), (77, 1), (54, 1), (166, 1), (143, 1), (67, 1),
             (156, 1), (63, 1), (152, 1), (107, 1), (196, 1), (251, 1), (285, 1), (50, 1)]

###################################################################################

UPLIFTING_MAJOR = [(267, 3776), (317, 3723), (301, 3628), (320, 3603), (178, 3569), (89, 3448),
                 (309, 3337), (314, 3216), (0, 3180), (288, 3159), (233, 3061), (319, 3008),
                 (112, 981), (194, 917), (256, 916), (16, 874), (216, 843), (283, 835),
                 (201, 783), (105, 771), (127, 766), (23, 715), (38, 692), (249, 637),
                 (272, 459), (191, 448), (91, 437), (235, 437), (306, 423), (303, 404),
                 (280, 400), (13, 396), (183, 394), (269, 394), (94, 393), (102, 389),
                 (180, 386), (293, 371), (181, 370), (5, 358), (290, 348), (212, 342),
                 (238, 335), (246, 324), (270, 315), (92, 314), (3, 310), (254, 308),
                 (316, 301), (110, 295), (123, 291), (2, 285), (104, 268), (236, 255),
                 (304, 254), (311, 250), (34, 250), (193, 244), (291, 244), (199, 235),
                 (312, 232), (114, 219), (215, 216), (248, 205), (37, 201), (25, 201),
                 (15, 197), (126, 195), (282, 191), (21, 184), (258, 167), (268, 151),
                 (179, 148), (203, 142), (234, 128), (90, 123), (1, 119), (144, 116),
                 (289, 102), (302, 99), (228, 97), (310, 95), (318, 94), (119, 92), (159, 91),
                 (285, 89), (139, 85), (162, 83), (50, 81), (73, 78), (42, 78), (196, 77),
                 (30, 76), (131, 75), (251, 75), (220, 73), (39, 72), (55, 71), (45, 71),
                 (315, 70), (217, 70), (120, 69), (227, 67), (264, 64), (209, 63), (31, 63),
                 (134, 62), (36, 62), (273, 61), (70, 60), (43, 58), (221, 58), (8, 56),
                 (160, 55), (138, 55), (192, 55), (97, 54), (186, 54), (241, 53), (71, 53),
                 (49, 53), (128, 53), (132, 52), (223, 52), (298, 52), (296, 51), (275, 51),
                 (208, 50), (263, 50), (99, 50), (214, 50), (277, 50), (153, 49), (96, 48),
                 (148, 48), (218, 47), (14, 46), (18, 45), (103, 44), (281, 44), (150, 43),
                 (125, 43), (10, 43), (247, 42), (294, 41), (64, 41), (307, 40), (40, 40),
                 (129, 40), (239, 40), (7, 38), (284, 38), (243, 38), (146, 37), (6, 37),
                 (95, 37), (184, 37), (213, 36), (188, 36), (35, 35), (59, 35), (124, 34),
                 (107, 33), (24, 32), (17, 31), (257, 31), (147, 30), (195, 30), (202, 29),
                 (308, 28), (106, 28), (57, 28), (276, 26), (115, 26), (58, 26), (61, 25),
                 (9, 25), (242, 25), (113, 25), (11, 24), (204, 23), (259, 22), (46, 22),
                 (274, 21), (255, 21), (135, 21), (224, 21), (240, 20), (295, 19), (187, 19),
                 (250, 19), (48, 19), (297, 19), (185, 18), (26, 17), (149, 17), (98, 16),
                 (261, 14), (197, 14), (286, 14), (75, 14), (164, 14), (68, 13), (157, 13),
                 (173, 13), (271, 12), (137, 12), (226, 12), (44, 12), (230, 11), (109, 11),
                 (117, 11), (206, 11), (292, 11), (182, 11), (222, 11), (252, 11), (244, 10),
                 (278, 10), (84, 10), (305, 10), (198, 10), (237, 10), (108, 10), (60, 10),
                 (53, 9), (136, 9), (158, 9), (225, 9), (69, 9), (47, 9), (287, 8), (41, 8),
                 (100, 8), (189, 8), (52, 8), (141, 8), (28, 8), (219, 8), (19, 8), (93, 8),
                 (133, 8), (165, 7), (313, 7), (20, 7), (76, 6), (142, 6), (231, 6), (253, 6),
                 (130, 6), (151, 5), (51, 5), (140, 5), (229, 5), (168, 5), (4, 5), (299, 5),
                 (22, 5), (170, 5), (155, 4), (62, 4), (145, 4), (174, 4), (66, 3), (56, 3),
                 (72, 3), (54, 3), (143, 3), (154, 3), (85, 3), (77, 3), (166, 3), (67, 3),
                 (152, 3), (245, 3), (279, 3), (111, 3), (200, 3), (171, 3), (79, 3), (210, 2),
                 (265, 2), (74, 2), (163, 2), (65, 2), (27, 2), (116, 2), (205, 2), (260, 2),
                 (32, 2), (156, 2), (63, 2), (300, 2), (12, 2), (101, 2), (190, 2), (232, 1),
                 (121, 1), (81, 1), (86, 1), (175, 1), (82, 1)]

###################################################################################

UPLIFTING_MINOR = [(301, 5035), (233, 5017), (314, 4999), (89, 4970), (320, 4956), (319, 4954),
                 (0, 4793), (267, 4760), (309, 4744), (178, 4715), (317, 4697), (288, 4644),
                 (114, 1184), (25, 1127), (248, 1111), (282, 1010), (193, 943), (203, 938),
                 (105, 912), (104, 906), (258, 906), (280, 883), (246, 882), (283, 870),
                 (16, 867), (94, 857), (127, 854), (238, 845), (102, 834), (194, 830), (5, 822),
                 (306, 813), (38, 795), (183, 792), (249, 791), (13, 784), (191, 780),
                 (256, 778), (112, 777), (290, 774), (23, 748), (272, 741), (235, 737),
                 (269, 737), (293, 714), (215, 700), (37, 695), (201, 694), (303, 693),
                 (15, 685), (316, 684), (311, 682), (216, 672), (126, 666), (91, 622), (2, 618),
                 (180, 616), (254, 606), (270, 596), (304, 592), (236, 590), (181, 577),
                 (92, 572), (34, 558), (123, 554), (3, 540), (21, 534), (212, 524), (312, 517),
                 (110, 508), (199, 500), (291, 491), (128, 224), (243, 217), (298, 217),
                 (144, 214), (90, 214), (39, 210), (8, 207), (162, 206), (234, 205), (97, 204),
                 (186, 204), (241, 203), (217, 200), (268, 199), (10, 198), (1, 192), (55, 190),
                 (179, 190), (188, 187), (125, 184), (315, 184), (302, 182), (318, 180),
                 (275, 178), (296, 168), (289, 168), (277, 166), (73, 166), (36, 165),
                 (119, 162), (263, 161), (99, 160), (310, 160), (30, 157), (214, 135),
                 (138, 135), (264, 133), (159, 129), (134, 128), (131, 127), (227, 125),
                 (70, 125), (281, 122), (43, 120), (46, 119), (209, 118), (247, 117),
                 (132, 116), (120, 110), (221, 108), (208, 108), (31, 106), (45, 103), (49, 99),
                 (224, 96), (96, 95), (59, 94), (220, 91), (148, 90), (135, 90), (7, 88),
                 (273, 88), (147, 84), (239, 82), (274, 77), (307, 76), (294, 75), (223, 75),
                 (240, 73), (17, 73), (106, 73), (192, 72), (213, 71), (185, 71), (58, 71),
                 (24, 71), (139, 70), (103, 66), (9, 66), (276, 65), (42, 65), (129, 64),
                 (95, 64), (187, 63), (242, 60), (98, 60), (150, 59), (285, 58), (40, 57),
                 (261, 57), (184, 57), (218, 56), (50, 55), (195, 55), (284, 53), (48, 52),
                 (196, 52), (117, 52), (251, 50), (295, 49), (202, 49), (250, 49), (146, 48),
                 (259, 48), (228, 48), (206, 48), (14, 48), (57, 47), (35, 47), (61, 46),
                 (6, 45), (113, 45), (124, 43), (157, 42), (28, 42), (137, 41), (68, 41),
                 (297, 40), (308, 40), (257, 39), (115, 38), (158, 38), (107, 37), (204, 35),
                 (160, 35), (71, 33), (26, 32), (226, 31), (69, 31), (153, 30), (165, 27),
                 (64, 27), (287, 26), (136, 25), (109, 25), (225, 24), (164, 24), (76, 24),
                 (286, 23), (75, 23), (155, 23), (11, 22), (252, 22), (253, 22), (93, 22),
                 (271, 22), (47, 21), (108, 21), (41, 21), (198, 20), (197, 19), (237, 19),
                 (219, 19), (182, 18), (66, 18), (130, 17), (292, 17), (305, 17), (20, 16),
                 (145, 15), (4, 15), (18, 15), (255, 14), (100, 14), (189, 14), (62, 14),
                 (244, 13), (151, 13), (170, 12), (52, 11), (141, 11), (278, 10), (313, 10),
                 (56, 10), (149, 9), (133, 9), (84, 8), (173, 8), (60, 8), (200, 8), (65, 7),
                 (299, 7), (230, 7), (44, 7), (154, 6), (85, 6), (222, 6), (174, 5), (81, 5),
                 (111, 5), (163, 4), (27, 4), (116, 4), (205, 4), (19, 4), (22, 4), (210, 3),
                 (265, 3), (74, 3), (168, 3), (51, 3), (260, 3), (12, 2), (101, 2), (190, 2),
                 (245, 2), (279, 2), (142, 2), (231, 2), (175, 2), (82, 2), (171, 2), (79, 2),
                 (152, 1), (140, 1), (229, 1), (54, 1), (143, 1), (53, 1), (121, 1), (300, 1),
                 (262, 1), (72, 1), (161, 1), (29, 1), (118, 1), (207, 1)]

###################################################################################

ALL_MOOD_TYPES = [HAPPY_MAJOR,
                  UPLIFTING_MAJOR,
                  UPLIFTING_MINOR, 
                  NEUTRAL_MAJOR,
                  NEUTRAL_MINOR,
                  MELANCHOLIC_MAJOR,
                  MELANCHOLIC_MINOR,
                  SAD_MAJOR,
                  SAD_MINOR
                 ]

###################################################################################

ALL_MOOD_TYPES_LABELS = ['Happy Major',
                         'Uplifting Major',
                         'Uplifting Minor',
                         'Neutral Major',
                         'Neutral Minor',
                         'Melancholic Major',
                         'Melancholic Minor',
                         'Sad Major',
                         'Sad Minor'
                        ]

###################################################################################

LEAD_INSTRUMENTS = [0, 1, 2, 3, 4, 5, 6, 7, # Piano
                    8, 9, 10, 11, 12, 13, 14, 15, # Chromatic Percussion
                    16, 17, 18, 19, 20, 21, 22, 23, # Organ
                    24, 25, 26, 27, 28, 29, 30, 31, # Guitar
                    40, 41, 46, # Strings
                    52, 53, 54, # Ensemble
                    56, 57, 59, 60, # Brass
                    64, 65, 66, 67, 68, 69, 70, 71, # Reed
                    72, 73, 74, 75, 76, 77, 78, 79, # Pipe
                    80, 81, 87 # Synth Lead
                   ]

###################################################################################

BASE_INSTRUMENTS = [32, 33, 34, 35, 36, 37, 38, 39, # Bass
                    42, 43, # Strings
                    58, 61, 62, 63, # Brass
                    87 # Synth Lead
                   ]

###################################################################################

def escore_notes_pitches_range(escore_notes,
                               range_patch=-1,
                               pitches_idx=4,
                               patches_idx=6
                               ):

    pitches = []
    
    if -1 < range_patch < 129:
        pitches = [e[pitches_idx] for e in escore_notes if e[patches_idx] == range_patch]

    else:
        pitches = [e[pitches_idx] for e in escore_notes]

    if pitches:
        min_pitch = min(pitches)
        avg_pitch = sum(pitches) / len(pitches)
        mode_pitch = statistics.mode(pitches)
        max_pitch = max(pitches)

        return [max_pitch-min_pitch, min_pitch, max_pitch, avg_pitch, mode_pitch]

    else:
        return [ -1] * 6

###################################################################################

def escore_notes_core(escore_notes, core_len=128):

    cscore = chordify_score([1000, escore_notes])

    chords = []
    chords_idxs = []
    
    for i, c in enumerate(cscore):
        
        pitches = [e[4] for e in c if e[3] != 9]

        if pitches:
            tones_chord = sorted(set([p % 12 for p in pitches]))

            if tones_chord not in ALL_CHORDS_SORTED:
                tones_chord = check_and_fix_tones_chord(tones_chord)

            chords.append(ALL_CHORDS_SORTED.index(tones_chord))
            chords_idxs.append(i)

    mid = len(chords_idxs) // 2
    clen = core_len // 2
    
    sidx = chords_idxs[mid-clen]
    eidx = chords_idxs[mid+clen]

    core_chords = chords[mid-clen:mid+clen]
    core_score = flatten(cscore[sidx:eidx])
    
    return core_score, core_chords

###################################################################################

def multiprocessing_wrapper(function, 
                            data_list, 
                            num_workers=None, 
                            verbose=True):
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    results = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm.tqdm(
            pool.imap(function, data_list),
            total=len(data_list),
            disable=not verbose
        ):
            results.append(result)

    return results


###################################################################################

def rle_encode_ones(matrix, div_mod=-1):
    
    flat_list = [val for row in matrix for val in row]
    
    encoding = []
    i = 0
    
    while i < len(flat_list):
        
        if flat_list[i] == 1:
            
            start_index = i
            count = 1
            i += 1
            
            while i < len(flat_list) and flat_list[i] == 1:
                count += 1
                i += 1
            
            if div_mod > 0:
                encoding.append((start_index // div_mod, start_index % div_mod))
                
            else:
                encoding.append(start_index)
            
        else:
            i += 1
            
    return encoding

###################################################################################

def rle_decode_ones(encoding, size=(128, 128)):
    
    flat_list = [0] * (size[0] * size[1])
    
    for start_index in encoding:
        flat_list[start_index] = 1
        
    matrix = [flat_list[i * size[1]:(i + 1) * size[1]] for i in range(size[0])]
    
    return matrix

###################################################################################

def vertical_list_search(list_of_lists, trg_list):

    src_list = list_of_lists
    
    if not src_list or not trg_list:
        return []
    
    num_rows = len(src_list)
    k = len(trg_list)
    
    row_sets = [set(row) for row in src_list]
    
    results = []
    
    for start in range(num_rows - k + 1):
        valid = True
        
        for offset, target in enumerate(trg_list):

            if target not in row_sets[start + offset]:
                valid = False
                break
                
        if valid:
            results.append(list(range(start, start + k)))
            
    return results
    
###################################################################################

def smooth_values(values, window_size=3):

    smoothed = []
    
    for i in range(len(values)):

        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        
        window = values[start:end]
        
        smoothed.append(int(sum(window) / len(window)))
        
    return smoothed

###################################################################################

def is_mostly_wide_peaks_and_valleys(values, 
                                     min_range=32, 
                                     threshold=0.7, 
                                     smoothing_window=5
                                    ):

    if not values:
        return False

    smoothed_values = smooth_values(values, smoothing_window)

    value_range = max(smoothed_values) - min(smoothed_values)
    
    if value_range < min_range:
        return False

    if all(v == smoothed_values[0] for v in smoothed_values):
        return False

    trend_types = []
    
    for i in range(1, len(smoothed_values)):
        if smoothed_values[i] > smoothed_values[i - 1]:
            trend_types.append(1)
            
        elif smoothed_values[i] < smoothed_values[i - 1]:
            trend_types.append(-1)
            
        else:
            trend_types.append(0)

    trend_count = trend_types.count(1) + trend_types.count(-1)

    proportion = trend_count / len(trend_types)

    return proportion >= threshold

###################################################################################

def system_memory_utilization(return_dict=False):

    if return_dict:
        return dict(psutil.virtual_memory()._asdict())

    else:
        print('RAM memory % used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/(1024**3))

###################################################################################

def system_cpus_utilization(return_dict=False):

    if return_dict:
        return {'num_cpus': psutil.cpu_count(),
                'cpus_util': psutil.cpu_percent()
                }

    else:
        print('Number of CPUs:', psutil.cpu_count())
        print('CPUs utilization:', psutil.cpu_percent())

###################################################################################

def create_files_list(datasets_paths=['./'],
                      files_exts=['.mid', '.midi', '.kar', '.MID', '.MIDI', '.KAR'],
                      max_num_files_per_dir=-1,
                      randomize_dir_files=False,
                      max_total_files=-1,
                      randomize_files_list=True,
                      check_for_dupes=False,
                      use_md5_hashes=False,
                      return_dupes=False,
                      verbose=True
                     ):
    
    if verbose:
        print('=' * 70)
        print('Searching for files...')
        print('This may take a while on a large dataset in particular...')
        print('=' * 70)

    files_exts = tuple(files_exts)
    
    filez_set = defaultdict(None)
    dupes_list = []
 
    for dataset_addr in datasets_paths:
        
        if verbose:
            print('=' * 70)
            print('Processing', dataset_addr)
            print('=' * 70)
        
        for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(dataset_addr), disable=not verbose):
                
                if randomize_dir_files:
                    random.shuffle(filenames)
                    
                if max_num_files_per_dir > 0:
                    max_num_files = max_num_files_per_dir
                    
                else:
                    max_num_files = len(filenames)
                    
                for file in filenames[:max_num_files]:
                    if file.endswith(files_exts):
                        if check_for_dupes:
                        
                            if use_md5_hashes:
                                md5_hash = hashlib.md5(open(os.path.join(dirpath, file), 'rb').read()).hexdigest()
                                
                                if md5_hash not in filez_set:
                                    filez_set[md5_hash] = os.path.join(dirpath, file)
                                
                                else:
                                    dupes_list.append(os.path.join(dirpath, file))
                                    
                            else:
                                if file not in filez_set:
                                    filez_set[file] = os.path.join(dirpath, file)
                                
                                else:
                                    dupes_list.append(os.path.join(dirpath, file))
                        else:
                            fpath = os.path.join(dirpath, file)
                            filez_set[fpath] = fpath                              

    filez = list(filez_set.values())

    if verbose:
        print('Done!')
        print('=' * 70)
    
    if filez:
        if randomize_files_list:
            
            if verbose:
                print('Randomizing file list...')
                
            random.shuffle(filez)
            
            if verbose:
                print('Done!')
                print('=' * 70)
                
        if verbose:
            print('Found', len(filez), 'files.')
            print('Skipped', len(dupes_list), 'duplicate files.')
            print('=' * 70)
 
    else:
        if verbose:
            print('Could not find any files...')
            print('Please check dataset dirs and files extensions...')
            print('=' * 70)
    
    if max_total_files > 0:     
        if return_dupes:
            return filez[:max_total_files], dupes_list
        
        else:
            return filez[:max_total_files]
    
    else:
        if return_dupes:
            return filez, dupes_list
        
        else:
            return filez

###################################################################################

def has_consecutive_trend(nums, count):
    
    if len(nums) < count:
        return False
    
    increasing_streak = 1
    decreasing_streak = 1

    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            increasing_streak += 1
            decreasing_streak = 1
            
        elif nums[i] < nums[i - 1]:
            decreasing_streak += 1
            increasing_streak = 1
            
        else:
            increasing_streak = decreasing_streak = 1
        
        if increasing_streak == count or decreasing_streak == count:
            return True
    
    return False

###################################################################################

def escore_notes_primary_features(escore_notes):

    #=================================================================

    def mean(values):
        return sum(values) / len(values) if values else None

    def std(values):
        if not values:
            return None
        m = mean(values)
        return math.sqrt(sum((x - m) ** 2 for x in values) / len(values)) if m is not None else None

    def skew(values):
        if not values:
            return None
        m = mean(values)
        s = std(values)
        if s is None or s == 0:
            return None
        return sum(((x - m) / s) ** 3 for x in values) / len(values)

    def kurtosis(values):
        if not values:
            return None
        m = mean(values)
        s = std(values)
        if s is None or s == 0:
            return None
        return sum(((x - m) / s) ** 4 for x in values) / len(values) - 3

    def median(values):
        if not values:
            return None
        srt = sorted(values)
        n = len(srt)
        mid = n // 2
        if n % 2 == 0:
            return (srt[mid - 1] + srt[mid]) / 2.0
        return srt[mid]

    def percentile(values, p):
        if not values:
            return None
        srt = sorted(values)
        n = len(srt)
        k = (n - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < n:
            return srt[f] * (1 - c) + srt[f + 1] * c
        return srt[f]

    def diff(values):
        if not values or len(values) < 2:
            return []
        return [values[i + 1] - values[i] for i in range(len(values) - 1)]

    def mad(values):
        if not values:
            return None
        m = median(values)
        return median([abs(x - m) for x in values])

    def entropy(values):
        if not values:
            return None
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        total = len(values)
        ent = 0.0
        for count in freq.values():
            p_val = count / total
            ent -= p_val * math.log2(p_val)
        return ent

    def mode(values):
        if not values:
            return None
        freq = {}
        for v in values:
            freq[v] = freq.get(v, 0) + 1
        max_count = max(freq.values())
        modes = [k for k, count in freq.items() if count == max_count]
        return min(modes)


    #=================================================================
    
    sp_score = solo_piano_escore_notes(escore_notes)

    dscore = delta_score_notes(sp_score)
    
    seq = []
    
    for d in dscore:
        seq.extend([d[1], d[2], d[4]])

    #=================================================================

    n = len(seq)
    if n % 3 != 0:
        seq = seq[: n - (n % 3)]
    arr = [seq[i:i + 3] for i in range(0, len(seq), 3)]

    #=================================================================
    
    features = {}

    delta_times = [row[0] for row in arr]
    if delta_times:
        features['delta_times_mean'] = mean(delta_times)
        features['delta_times_std'] = std(delta_times)
        features['delta_times_min'] = min(delta_times)
        features['delta_times_max'] = max(delta_times)
        features['delta_times_skew'] = skew(delta_times)
        features['delta_times_kurtosis'] = kurtosis(delta_times)
        delta_zero_count = sum(1 for x in delta_times if x == 0)
        features['delta_times_zero_ratio'] = delta_zero_count / len(delta_times)
        nonzero_dt = [x for x in delta_times if x != 0]
        if nonzero_dt:
            features['delta_times_nonzero_mean'] = mean(nonzero_dt)
            features['delta_times_nonzero_std'] = std(nonzero_dt)
        else:
            features['delta_times_nonzero_mean'] = None
            features['delta_times_nonzero_std'] = None
        features['delta_times_mad'] = mad(delta_times)
        features['delta_times_cv'] = (features['delta_times_std'] / features['delta_times_mean']
                                      if features['delta_times_mean'] and features['delta_times_mean'] != 0 else None)
        features['delta_times_entropy'] = entropy(delta_times)
        features['delta_times_range'] = max(delta_times) - min(delta_times)
        features['delta_times_median'] = median(delta_times)
        features['delta_times_quantile_25'] = percentile(delta_times, 25)
        features['delta_times_quantile_75'] = percentile(delta_times, 75)
        if (features['delta_times_quantile_25'] is not None and features['delta_times_quantile_75'] is not None):
            features['delta_times_iqr'] = features['delta_times_quantile_75'] - features['delta_times_quantile_25']
        else:
            features['delta_times_iqr'] = None
    else:
        for key in ['delta_times_mean', 'delta_times_std', 'delta_times_min', 'delta_times_max',
                    'delta_times_skew', 'delta_times_kurtosis', 'delta_times_zero_ratio',
                    'delta_times_nonzero_mean', 'delta_times_nonzero_std', 'delta_times_mad',
                    'delta_times_cv', 'delta_times_entropy', 'delta_times_range', 'delta_times_median',
                    'delta_times_quantile_25', 'delta_times_quantile_75', 'delta_times_iqr']:
            features[key] = None

    #=================================================================

    durations = [row[1] for row in arr]
    if durations:
        features['durations_mean'] = mean(durations)
        features['durations_std'] = std(durations)
        features['durations_min'] = min(durations)
        features['durations_max'] = max(durations)
        features['durations_skew'] = skew(durations)
        features['durations_kurtosis'] = kurtosis(durations)
        features['durations_mad'] = mad(durations)
        features['durations_cv'] = (features['durations_std'] / features['durations_mean']
                                    if features['durations_mean'] and features['durations_mean'] != 0 else None)
        features['durations_entropy'] = entropy(durations)
        features['durations_range'] = max(durations) - min(durations)
        features['durations_median'] = median(durations)
        features['durations_quantile_25'] = percentile(durations, 25)
        features['durations_quantile_75'] = percentile(durations, 75)
        if features['durations_quantile_25'] is not None and features['durations_quantile_75'] is not None:
            features['durations_iqr'] = features['durations_quantile_75'] - features['durations_quantile_25']
        else:
            features['durations_iqr'] = None
    else:
        for key in ['durations_mean', 'durations_std', 'durations_min', 'durations_max',
                    'durations_skew', 'durations_kurtosis', 'durations_mad', 'durations_cv',
                    'durations_entropy', 'durations_range', 'durations_median', 'durations_quantile_25',
                    'durations_quantile_75', 'durations_iqr']:
            features[key] = None

    #=================================================================

    pitches = [row[2] for row in arr]
    if pitches:
        features['pitches_mean'] = mean(pitches)
        features['pitches_std'] = std(pitches)
        features['pitches_min'] = min(pitches)
        features['pitches_max'] = max(pitches)
        features['pitches_skew'] = skew(pitches)
        features['pitches_kurtosis'] = kurtosis(pitches)
        features['pitches_range'] = max(pitches) - min(pitches)
        features['pitches_median'] = median(pitches)
        features['pitches_quantile_25'] = percentile(pitches, 25)
        features['pitches_quantile_75'] = percentile(pitches, 75)
        if len(pitches) > 1:
            dps = diff(pitches)
            features['pitches_diff_mean'] = mean(dps)
            features['pitches_diff_std'] = std(dps)
        else:
            features['pitches_diff_mean'] = None
            features['pitches_diff_std'] = None
        features['pitches_mad'] = mad(pitches)
        if len(pitches) > 2:
            peaks = sum(1 for i in range(1, len(pitches)-1)
                        if pitches[i] > pitches[i-1] and pitches[i] > pitches[i+1])
            valleys = sum(1 for i in range(1, len(pitches)-1)
                          if pitches[i] < pitches[i-1] and pitches[i] < pitches[i+1])
        else:
            peaks, valleys = None, None
        features['pitches_peak_count'] = peaks
        features['pitches_valley_count'] = valleys
        if len(pitches) > 1:
            x = list(range(len(pitches)))
            denominator = (len(x) * sum(xi ** 2 for xi in x) - sum(x) ** 2)
            if denominator != 0:
                slope = (len(x) * sum(x[i] * pitches[i] for i in range(len(x))) -
                         sum(x) * sum(pitches)) / denominator
            else:
                slope = None
            features['pitches_trend_slope'] = slope
        else:
            features['pitches_trend_slope'] = None

        features['pitches_unique_count'] = len(set(pitches))
        pitch_class_hist = {i: 0 for i in range(12)}
        for p in pitches:
            pitch_class_hist[p % 12] += 1
        total_pitch = len(pitches)
        for i in range(12):
            features[f'pitches_pc_{i}'] = (pitch_class_hist[i] / total_pitch) if total_pitch > 0 else None

        max_asc = 0
        cur_asc = 0
        max_desc = 0
        cur_desc = 0
        for i in range(1, len(pitches)):
            if pitches[i] > pitches[i-1]:
                cur_asc += 1
                max_asc = max(max_asc, cur_asc)
                cur_desc = 0
            elif pitches[i] < pitches[i-1]:
                cur_desc += 1
                max_desc = max(max_desc, cur_desc)
                cur_asc = 0
            else:
                cur_asc = 0
                cur_desc = 0
        features['pitches_max_consecutive_ascending'] = max_asc if pitches else None
        features['pitches_max_consecutive_descending'] = max_desc if pitches else None
        p_intervals = diff(pitches)
        features['pitches_median_diff'] = median(p_intervals) if p_intervals else None
        if p_intervals:
            dc = sum(1 for i in range(1, len(p_intervals))
                     if (p_intervals[i] > 0 and p_intervals[i-1] < 0) or (p_intervals[i] < 0 and p_intervals[i-1] > 0))
            features['pitches_direction_changes'] = dc
        else:
            features['pitches_direction_changes'] = None
    else:
        for key in (['pitches_mean', 'pitches_std', 'pitches_min', 'pitches_max', 'pitches_skew',
                     'pitches_kurtosis', 'pitches_range', 'pitches_median', 'pitches_quantile_25',
                     'pitches_quantile_75', 'pitches_diff_mean', 'pitches_diff_std', 'pitches_mad',
                     'pitches_peak_count', 'pitches_valley_count', 'pitches_trend_slope',
                     'pitches_unique_count', 'pitches_max_consecutive_ascending', 'pitches_max_consecutive_descending',
                     'pitches_median_diff', 'pitches_direction_changes'] +
                    [f'pitches_pc_{i}' for i in range(12)]):
            features[key] = None

    #=================================================================

    overall = [x for row in arr for x in row]
    if overall:
        features['overall_mean'] = mean(overall)
        features['overall_std'] = std(overall)
        features['overall_min'] = min(overall)
        features['overall_max'] = max(overall)
        features['overall_cv'] = (features['overall_std'] / features['overall_mean']
                                  if features['overall_mean'] and features['overall_mean'] != 0 else None)
    else:
        for key in ['overall_mean', 'overall_std', 'overall_min', 'overall_max', 'overall_cv']:
            features[key] = None

    #=================================================================

    onsets = []
    cumulative = 0
    for dt in delta_times:
        onsets.append(cumulative)
        cumulative += dt
    if onsets and durations:
        overall_piece_duration = onsets[-1] + durations[-1]
    else:
        overall_piece_duration = None
    features['overall_piece_duration'] = overall_piece_duration
    features['overall_notes_density'] = (len(arr) / overall_piece_duration
                                         if overall_piece_duration and overall_piece_duration > 0 else None)
    features['rhythm_ratio'] = (features['durations_mean'] / features['delta_times_mean']
                                if features['delta_times_mean'] and features['delta_times_mean'] != 0 else None)
    features['overall_sum_delta_times'] = (sum(delta_times) if delta_times else None)
    features['overall_sum_durations'] = (sum(durations) if durations else None)
    features['overall_voicing_ratio'] = (sum(durations) / overall_piece_duration
                                         if overall_piece_duration and durations else None)
    features['overall_onset_std'] = std(onsets) if onsets else None

    #=================================================================

    chords_raw = []
    chords_pc = []
    current_group = []
    for i, note in enumerate(arr):
        dt = note[0]
        if i == 0:
            current_group = [i]
        else:
            if dt == 0:
                current_group.append(i)
            else:
                if len(current_group) >= 2:
                    chord_notes = [arr[j][2] for j in current_group]
                    chords_raw.append(tuple(sorted(chord_notes)))
                    chords_pc.append(tuple(sorted(set(p % 12 for p in chord_notes))))

                current_group = [i]

    if current_group and len(current_group) >= 2:
        chord_notes = [arr[j][2] for j in current_group]
        chords_raw.append(tuple(sorted(chord_notes)))
        chords_pc.append(tuple(sorted(set(p % 12 for p in chord_notes))))
    
    if chords_raw:
        chord_count = len(chords_raw)
        features['chords_count'] = chord_count
        features['chords_density'] = (chord_count / overall_piece_duration
                                      if overall_piece_duration and chord_count is not None else None)
        chord_sizes = [len(ch) for ch in chords_raw]
        features['chords_size_mean'] = mean(chord_sizes)
        features['chords_size_std'] = std(chord_sizes)
        features['chords_size_min'] = min(chord_sizes) if chord_sizes else None
        features['chords_size_max'] = max(chord_sizes) if chord_sizes else None
        features['chords_unique_raw_count'] = len(set(chords_raw))
        features['chords_unique_pc_count'] = len(set(chords_pc))
        features['chords_entropy_raw'] = entropy(chords_raw)
        features['chords_entropy_pc'] = entropy(chords_pc)
        if len(chords_raw) > 1:
            rep_raw = sum(1 for i in range(1, len(chords_raw)) if chords_raw[i] == chords_raw[i - 1])
            features['chords_repeat_ratio_raw'] = rep_raw / (len(chords_raw) - 1)
        else:
            features['chords_repeat_ratio_raw'] = None
        if len(chords_pc) > 1:
            rep_pc = sum(1 for i in range(1, len(chords_pc)) if chords_pc[i] == chords_pc[i - 1])
            features['chords_repeat_ratio_pc'] = rep_pc / (len(chords_pc) - 1)
        else:
            features['chords_repeat_ratio_pc'] = None
        if len(chords_raw) > 1:
            bigrams_raw = [(chords_raw[i], chords_raw[i + 1]) for i in range(len(chords_raw) - 1)]
            features['chords_bigram_entropy_raw'] = entropy(bigrams_raw)
        else:
            features['chords_bigram_entropy_raw'] = None
        if len(chords_pc) > 1:
            bigrams_pc = [(chords_pc[i], chords_pc[i + 1]) for i in range(len(chords_pc) - 1)]
            features['chords_bigram_entropy_pc'] = entropy(bigrams_pc)
        else:
            features['chords_bigram_entropy_pc'] = None
        features['chords_mode_raw'] = mode(chords_raw)
        features['chords_mode_pc'] = mode(chords_pc)
        if chords_pc:
            pc_sizes = [len(ch) for ch in chords_pc]
            features['chords_pc_size_mean'] = mean(pc_sizes)
        else:
            features['chords_pc_size_mean'] = None
    else:
        for key in ['chords_count', 'chords_density', 'chords_size_mean', 'chords_size_std',
                    'chords_size_min', 'chords_size_max', 'chords_unique_raw_count', 'chords_unique_pc_count',
                    'chords_entropy_raw', 'chords_entropy_pc', 'chords_repeat_ratio_raw', 'chords_repeat_ratio_pc',
                    'chords_bigram_entropy_raw', 'chords_bigram_entropy_pc', 'chords_mode_raw', 'chords_mode_pc',
                    'chords_pc_size_mean']:
            features[key] = None

    #=================================================================

    if delta_times:
        med_dt = features['delta_times_median']
        iqr_dt = features['delta_times_iqr']
        threshold_a = med_dt + 1.5 * iqr_dt if med_dt is not None and iqr_dt is not None else None
        threshold_b = percentile(delta_times, 90)
        if threshold_a is not None and threshold_b is not None:
            phrase_threshold = max(threshold_a, threshold_b)
        elif threshold_a is not None:
            phrase_threshold = threshold_a
        elif threshold_b is not None:
            phrase_threshold = threshold_b
        else:
            phrase_threshold = None
    else:
        phrase_threshold = None

    phrases = []
    current_phrase = []
    if onsets:
        current_phrase.append(0)
        for i in range(len(onsets) - 1):
            gap = onsets[i + 1] - onsets[i]
            if phrase_threshold is not None and gap > phrase_threshold:
                phrases.append(current_phrase)
                current_phrase = []
            current_phrase.append(i + 1)
        if current_phrase:
            phrases.append(current_phrase)
    if phrases:
        phrase_note_counts = []
        phrase_durations = []
        phrase_densities = []
        phrase_mean_pitches = []
        phrase_pitch_ranges = []
        phrase_start_times = []
        phrase_end_times = []
        for phrase in phrases:
            note_count = len(phrase)
            phrase_note_counts.append(note_count)
            ph_start = onsets[phrase[0]]
            ph_end = onsets[phrase[-1]] + durations[phrase[-1]]
            phrase_start_times.append(ph_start)
            phrase_end_times.append(ph_end)
            ph_duration = ph_end - ph_start
            phrase_durations.append(ph_duration)
            density = note_count / ph_duration if ph_duration > 0 else None
            phrase_densities.append(density)
            ph_pitches = [pitches[i] for i in phrase if i < len(pitches)]
            phrase_mean_pitches.append(mean(ph_pitches) if ph_pitches else None)
            phrase_pitch_ranges.append((max(ph_pitches) - min(ph_pitches)) if ph_pitches else None)
        if len(phrases) > 1:
            phrase_gaps = []
            for i in range(len(phrases) - 1):
                gap = phrase_start_times[i + 1] - phrase_end_times[i]
                phrase_gaps.append(gap if gap > 0 else 0)
        else:
            phrase_gaps = []
        features['phrases_count'] = len(phrases)
        features['phrases_avg_note_count'] = mean(phrase_note_counts) if phrase_note_counts else None
        features['phrases_std_note_count'] = std(phrase_note_counts) if phrase_note_counts else None
        features['phrases_min_note_count'] = min(phrase_note_counts) if phrase_note_counts else None
        features['phrases_max_note_count'] = max(phrase_note_counts) if phrase_note_counts else None
        features['phrases_avg_duration'] = mean(phrase_durations) if phrase_durations else None
        features['phrases_std_duration'] = std(phrase_durations) if phrase_durations else None
        features['phrases_min_duration'] = min(phrase_durations) if phrase_durations else None
        features['phrases_max_duration'] = max(phrase_durations) if phrase_durations else None
        features['phrases_avg_density'] = mean(phrase_densities) if phrase_densities else None
        features['phrases_std_density'] = std(phrase_densities) if phrase_densities else None
        features['phrases_avg_mean_pitch'] = mean(phrase_mean_pitches) if phrase_mean_pitches else None
        features['phrases_avg_pitch_range'] = mean(phrase_pitch_ranges) if phrase_pitch_ranges else None
        if phrase_gaps:
            features['phrases_avg_gap'] = mean(phrase_gaps)
            features['phrases_std_gap'] = std(phrase_gaps)
            features['phrases_min_gap'] = min(phrase_gaps)
            features['phrases_max_gap'] = max(phrase_gaps)
        else:
            features['phrases_avg_gap'] = None
            features['phrases_std_gap'] = None
            features['phrases_min_gap'] = None
            features['phrases_max_gap'] = None
        features['phrases_threshold'] = phrase_threshold
    else:
        for key in ['phrases_count', 'phrases_avg_note_count', 'phrases_std_note_count',
                    'phrases_min_note_count', 'phrases_max_note_count', 'phrases_avg_duration',
                    'phrases_std_duration', 'phrases_min_duration', 'phrases_max_duration',
                    'phrases_avg_density', 'phrases_std_density', 'phrases_avg_mean_pitch',
                    'phrases_avg_pitch_range', 'phrases_avg_gap', 'phrases_std_gap',
                    'phrases_min_gap', 'phrases_max_gap', 'phrases_threshold']:
            features[key] = None

    #=================================================================

    return features

###################################################################################

def winsorized_normalize(data, new_range=(0, 255), clip=1.5):

    #=================================================================

    new_min, new_max = new_range

    #=================================================================

    def percentile(values, p):
        
        srt = sorted(values)
        n = len(srt)
        if n == 1:
            return srt[0]
        k = (n - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < n:
            return srt[f] * (1 - c) + srt[f + 1] * c
            
        return srt[f]

    #=================================================================

    q1 = percentile(data, 25)
    q3 = percentile(data, 75)
    iqr = q3 - q1

    lower_bound_w = q1 - clip * iqr
    upper_bound_w = q3 + clip * iqr

    data_min = min(data)
    data_max = max(data)
    effective_low = max(lower_bound_w, data_min)
    effective_high = min(upper_bound_w, data_max)

    #=================================================================

    if effective_high == effective_low:
        
        if data_max == data_min:
            return [int(new_min)] * len(data)
            
        normalized = [(x - data_min) / (data_max - data_min) for x in data]
        
        return [int(round(new_min + norm * (new_max - new_min))) for norm in normalized]

    #=================================================================

    clipped = [x if x >= effective_low else effective_low for x in data]
    clipped = [x if x <= effective_high else effective_high for x in clipped]

    normalized = [(x - effective_low) / (effective_high - effective_low) for x in clipped]

    #=================================================================
    
    return [int(round(new_min + norm * (new_max - new_min))) for norm in normalized]

###################################################################################

def tokenize_features_to_ints_winsorized(features, new_range=(0, 255), clip=1.5, none_token=-1):

    values = []    
    tokens = []

    #=================================================================

    def process_value(val):
        
        if isinstance(val, (int, float)):
            return int(round(abs(val)))
            
        elif isinstance(val, (list, tuple)):
            return int(round(abs(sum(val) / len(val))))
            
        else:
            return int(abs(hash(val)) % (10 ** 8))

    #=================================================================

    for key in sorted(features.keys()):
        
        value = features[key]
        
        if value is None:
            tokens.append(none_token)
            values.append(none_token)
            
        else:
            tokens.append(process_value(value))
            
            if isinstance(value, (list, tuple)):
                values.append(sum(value) / len(value))
                
            else:
                values.append(value)

    #=================================================================
    
    norm_tokens = winsorized_normalize(tokens, new_range, clip)

    #=================================================================

    return values, tokens, norm_tokens

###################################################################################

def write_jsonl(records_dicts_list, 
                file_name='data', 
                file_ext='.jsonl', 
                file_mode='w', 
                line_sep='\n', 
                verbose=True
               ):

    if verbose:
        print('=' * 70)
        print('Writing', len(records_dicts_list), 'records to jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    l_count = 0

    with open(file_name, mode=file_mode) as f:
        for record in tqdm.tqdm(records_dicts_list, disable=not verbose):
            f.write(json.dumps(record) + line_sep)
            l_count += 1

    f.close()

    if verbose:
        print('=' * 70)
        print('Written total of', l_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

###################################################################################
        
def read_jsonl(file_name='data', 
               file_ext='.jsonl', 
               verbose=True
              ):

    if verbose:
        print('=' * 70)
        print('Reading jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    with open(file_name, 'r') as f:

        records = []
        gl_count = 0
        
        for i, line in tqdm.tqdm(enumerate(f), disable=not verbose):
            
            try:
                record = json.loads(line)
                records.append(record)
                gl_count += 1

            except KeyboardInterrupt:
                if verbose:
                    print('=' * 70)
                    print('Stoping...')
                    print('=' * 70)
                    
                f.close()
    
                return records
               
            except json.JSONDecodeError:
                if verbose:
                    print('=' * 70)
                    print('[ERROR] Line', i, 'is corrupted! Skipping it...')
                    print('=' * 70)
                    
                continue
                
    f.close()
    
    if verbose:
        print('=' * 70)
        print('Loaded total of', gl_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return records

###################################################################################

def read_jsonl_lines(lines_indexes_list, 
                     file_name='data', 
                     file_ext='.jsonl', 
                     verbose=True
                    ):

    if verbose:
        print('=' * 70)
        print('Reading jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    records = []
    l_count = 0

    lines_indexes_list.sort(reverse=True)

    with open(file_name, 'r') as f:
        for current_line_number, line in tqdm.tqdm(enumerate(f)):

            try:
                if current_line_number in lines_indexes_list:
                    record = json.loads(line)
                    records.append(record)
                    lines_indexes_list = lines_indexes_list[:-1]
                    l_count += 1

                if not lines_indexes_list:
                    break

            except KeyboardInterrupt:
                if verbose:
                    print('=' * 70)
                    print('Stoping...')
                    print('=' * 70)
                    
                f.close()
    
                return records
               
            except json.JSONDecodeError:
                if verbose:
                    print('=' * 70)
                    print('[ERROR] Line', current_line_number, 'is corrupted! Skipping it...')
                    print('=' * 70)
                    
                continue

    f.close()
    
    if verbose:
        print('=' * 70)
        print('Loaded total of', l_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return records

###################################################################################

def compute_base(x: int, n: int) -> int:

    if x < 0:
        raise ValueError("x must be non-negative.")
    if x == 0:
        return 2 
        
    b = max(2, int(x ** (1 / n)))
    
    if b ** n <= x:
        b += 1
        
    return b

###################################################################################

def encode_int_auto(x: int, n: int) -> tuple[int, list[int]]:
   
    base = compute_base(x, n)
    digits = [0] * n
    
    for i in range(n - 1, -1, -1):
        digits[i] = x % base
        x //= base
        
    return base, digits

###################################################################################

def decode_int_auto(base: int, digits: list[int]) -> int:
   
    x = 0
    for digit in digits:
        if digit < 0 or digit >= base:
            raise ValueError(f"Each digit must be in the range 0 to {base - 1}. Invalid digit: {digit}")
            
        x = x * base + digit
        
    return x

###################################################################################

def encode_int_manual(x, base, n):

    digits = [0] * n
    
    for i in range(n - 1, -1, -1):
        digits[i] = x % base
        x //= base

    return digits

###################################################################################

def escore_notes_pitches_chords_signature(escore_notes, 
                                          max_patch=128, 
                                          sort_by_counts=False, 
                                          use_full_chords=False
                                         ):
    
    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL
        
    else:
        CHORDS = ALL_CHORDS_SORTED
    
    max_patch = max(0, min(128, max_patch))

    escore_notes = [e for e in escore_notes if e[6] <= max_patch]

    if escore_notes:

        cscore = chordify_score([1000, escore_notes])
        
        sig = []
        dsig = []
        
        drums_offset = len(CHORDS) + 128
        
        bad_chords_counter = 0
        
        for c in cscore:
            
            all_pitches = [e[4] if e[3] != 9 else e[4]+128 for e in c]
            chord = sorted(set(all_pitches))
        
            pitches = sorted([p for p in chord if p < 128], reverse=True)
            drums = [(d+drums_offset)-128 for d in chord if d > 127]
        
            if pitches:
              if len(pitches) > 1:
                tones_chord = sorted(set([p % 12 for p in pitches]))
                     
                try:
                    sig_token = CHORDS.index(tones_chord) + 128
                except:
                    checked_tones_chord = check_and_fix_tones_chord(tones_chord, use_full_chords=use_full_chords)
                    sig_token = CHORDS.index(checked_tones_chord) + 128
                    bad_chords_counter += 1
                    
              elif len(pitches) == 1:
                sig_token = pitches[0]
        
              sig.append(sig_token)
        
            if drums:
              dsig.extend(drums)
        
        sig_p = {}
        
        for item in sig+dsig:
            
            if item in sig_p:
                sig_p[item] += 1
        
            else:
                sig_p[item] = 1
        
        sig_p[-1] = bad_chords_counter
        
        fsig = [list(v) for v in sig_p.items()]
    
        if sort_by_counts:
            fsig.sort(key=lambda x: x[1], reverse=True)
    
        return fsig

    else:
        return []

###################################################################################

def compute_sustain_intervals(events):

    intervals = []
    pedal_on = False
    current_start = None
    
    for t, cc in events:
        if not pedal_on and cc >= 64:

            pedal_on = True
            current_start = t
        elif pedal_on and cc < 64:

            pedal_on = False
            intervals.append((current_start, t))
            current_start = None

    if pedal_on:
        intervals.append((current_start, float('inf')))

    merged = []
    
    for interval in intervals:
        if merged and interval[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        else:
            merged.append(interval)
    return merged

###################################################################################

def apply_sustain_to_ms_score(score):

    sustain_by_channel = {}
    
    for track in score[1:]:
        for event in track:
            if event[0] == 'control_change' and event[3] == 64:
                channel = event[2]
                sustain_by_channel.setdefault(channel, []).append((event[1], event[4]))
    
    sustain_intervals_by_channel = {}
    
    for channel, events in sustain_by_channel.items():
        events.sort(key=lambda x: x[0])
        sustain_intervals_by_channel[channel] = compute_sustain_intervals(events)
    
    global_max_off = 0
    
    for track in score[1:]:
        for event in track:
            if event[0] == 'note':
                global_max_off = max(global_max_off, event[1] + event[2])
                
    for channel, intervals in sustain_intervals_by_channel.items():
        updated_intervals = []
        for start, end in intervals:
            if end == float('inf'):
                end = global_max_off
            updated_intervals.append((start, end))
        sustain_intervals_by_channel[channel] = updated_intervals
        
    if sustain_intervals_by_channel:
        
        for track in score[1:]:
            for event in track:
                if event[0] == 'note':
                    start = event[1]
                    nominal_dur = event[2]
                    nominal_off = start + nominal_dur
                    channel = event[3]
                    
                    intervals = sustain_intervals_by_channel.get(channel, [])
                    effective_off = nominal_off
        
                    for intv_start, intv_end in intervals:
                        if intv_start < nominal_off < intv_end:
                            effective_off = intv_end
                            break
                    
                    effective_dur = effective_off - start
                    
                    event[2] = effective_dur

    return score
    
###################################################################################

def copy_file(src_file: str, trg_dir: str, add_subdir: bool = False, verbose: bool = False):
   
    src_path = Path(src_file)
    target_directory = Path(trg_dir)

    if not src_path.is_file():
        if verbose:
            print("Source file does not exist or is not a file.")
        
        return None

    target_directory.mkdir(parents=True, exist_ok=True)
    
    if add_subdir:
        first_letter = src_path.name[0]
        target_directory = target_directory / first_letter
        target_directory.mkdir(parents=True, exist_ok=True)

    destination = target_directory / src_path.name

    try:
        shutil.copy2(src_path, destination)

    except:
        if verbose:
            print('File could not be copied!')
            
        return None
    
    if verbose:
        print('File copied!')

    return None

###################################################################################

def escore_notes_even_timings(escore_notes, in_place=True):

    if in_place:
        for e in escore_notes:
            if e[1] % 2 != 0:
                e[1] += 1
    
            if e[2] % 2 != 0:
                e[2] += 1

        return []

    else:
        escore = copy.deepcopy(escore_notes)
        
        for e in escore:
            if e[1] % 2 != 0:
                e[1] += 1
    
            if e[2] % 2 != 0:
                e[2] += 1

        return escore

###################################################################################

def both_chords(chord1, chord2, merge_threshold=2):
    
    if len(chord1) > 1 and len(chord2) > 0 and chord2[0][1]-chord1[0][1] <= merge_threshold:
        return True
    
    elif len(chord1) > 0 and len(chord2) > 1 and chord2[0][1]-chord1[0][1] <= merge_threshold:
        return True

    else:
        return False

def merge_chords(chord1, chord2, sort_drums_last=False):

    mchord = chord1

    seen = []

    for e in chord2:
        if tuple([e[4], e[6]]) not in seen:
            mchord.append(e)
            seen.append(tuple([e[4], e[6]]))

    for e in mchord[1:]:
        e[1] = mchord[0][1]
    
    if sort_drums_last:
        mchord.sort(key=lambda x: (-x[4], x[6]) if x[6] != 128 else (x[6], -x[4]))

    else:
        mchord.sort(key=lambda x: (-x[4], x[6]))

    return mchord
    
def merge_escore_notes(escore_notes, merge_threshold=2, sort_drums_last=False):

    cscore = chordify_score([1000, escore_notes])
    
    merged_chords = []
    merged_chord = cscore[0]
    
    for i in range(1, len(cscore)):

        cchord = cscore[i]

        if both_chords(merged_chord, cchord, merge_threshold=merge_threshold):
            merged_chord = merge_chords(merged_chord, cchord, sort_drums_last=sort_drums_last)

        else:
            merged_chords.append(merged_chord)
            merged_chord = cchord
            
    return flatten(merged_chords)

###################################################################################

def solo_piano_escore_notes_tokenized(escore_notes,
                                      compress_start_times=True,
                                      encode_velocities=False,
                                      verbose=False
                                      ):

    if verbose:
        print('=' * 70)
        print('Encoding MIDI...')
    
    sp_escore_notes = solo_piano_escore_notes(escore_notes)
    zscore = recalculate_score_timings(sp_escore_notes)
    dscore = delta_score_notes(zscore, timings_clip_value=127)
    
    score = []
    
    notes_counter = 0
    chords_counter = 1
    
    for i, e in enumerate(dscore):
        
        dtime = e[1]
        dur = e[2]
        ptc = e[4]
        vel = e[5]

        if compress_start_times:
            
            if i == 0:
                score.extend([0, dur+128, ptc+256])
                
                if encode_velocities:
                    score.append(vel+384)
                    
            else:
                if dtime == 0:
                    score.extend([dur+128, ptc+256])

                else:
                    score.extend([dtime, dur+128, ptc+256])
                    
                if encode_velocities:
                    score.append(vel+384)
                    
            if dtime != 0:
                chords_counter += 1

        else:
            score.extend([dtime, dur+128, ptc+256])

            if encode_velocities:
                score.append(vel+384)

            if dtime != 0:
                chords_counter += 1
                
        notes_counter += 1

    if verbose:
        print('Done!')
        print('=' * 70)
        
        print('Source MIDI composition has', len(zscore), 'notes')
        print('Source MIDI composition has', len([d[1] for d in dscore if d[1] !=0 ])+1, 'chords')
        print('-' * 70)
        print('Encoded sequence has', notes_counter, 'pitches')
        print('Encoded sequence has', chords_counter, 'chords')
        print('-' * 70)
        print('Final encoded sequence has', len(score), 'tokens')
        print('=' * 70)
        
    return score

###################################################################################

def equalize_closest_elements_dynamic(seq,
                                      min_val=128,
                                      max_val=256,
                                      splitting_factor=1.5,
                                      tightness_threshold=0.15
                                      ):

    candidates = [(i, x) for i, x in enumerate(seq) if min_val <= x <= max_val]
    
    if len(candidates) < 2:
        return seq.copy()

    sorted_candidates = sorted(candidates, key=lambda pair: pair[1])
    candidate_values = [val for _, val in sorted_candidates]
    
    differences = [candidate_values[i+1] - candidate_values[i] for i in range(len(candidate_values)-1)]
    
    def median(lst):
        
        n = len(lst)
        sorted_lst = sorted(lst)
        mid = n // 2
        
        if n % 2 == 0:
            return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
            
        else:
            return sorted_lst[mid]
    
    med_diff = median(differences)

    split_indices = [i for i, diff in enumerate(differences) if diff > splitting_factor * med_diff]
    
    clusters = []
    
    if split_indices:
        start = 0
        for split_index in split_indices:
            clusters.append(sorted_candidates[start:split_index+1])
            start = split_index + 1
        clusters.append(sorted_candidates[start:])
        
    else:
        clusters = [sorted_candidates]
    

    valid_clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    if not valid_clusters:
        return seq.copy()

    def cluster_spread(cluster):
        values = [val for (_, val) in cluster]
        return max(values) - min(values)
    
    valid_clusters.sort(key=lambda cluster: (len(cluster), -cluster_spread(cluster)), reverse=True)
    selected_cluster = valid_clusters[0]

    allowed_range_width = max_val - min_val
    spread = cluster_spread(selected_cluster)
    ratio = spread / allowed_range_width
    
    if ratio > tightness_threshold:
        return seq.copy()

    cluster_values = [val for (_, val) in selected_cluster]
    equal_value = sum(cluster_values) // len(cluster_values)
    

    result = list(seq)
    for idx, _ in selected_cluster:
        result[idx] = equal_value
    
    return result

###################################################################################

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

###################################################################################

def compress_tokens_sequence(seq, 
                             min_val=128, 
                             max_val=256, 
                             group_size=2, 
                             splitting_factor=1.5, 
                             tightness_threshold=0.15
                            ):
    
    comp_seq = equalize_closest_elements_dynamic(seq, 
                                                 min_val, 
                                                 max_val, 
                                                 splitting_factor=splitting_factor, 
                                                 tightness_threshold=tightness_threshold
                                                 )

    seq_split = sorted(chunk_list(comp_seq, group_size), key=lambda x: (-x[0], -x[1]))

    seq_grouped = [[[k]] + [vv[1:] for vv in v] for k, v in groupby(seq_split, key=lambda x: x[0])]

    return flatten(flatten(sorted(seq_grouped, key=lambda x: -x[1][0])))

###################################################################################

def merge_adjacent_pairs(values_counts):
   
    merged = []
    i = 0
    
    while i < len(values_counts):

        if i < len(values_counts) - 1:
            value1, count1 = values_counts[i]
            value2, count2 = values_counts[i + 1]
            
            if value2 - value1 == 1:
                if count2 > count1:
                    merged_value = value2
                    
                else:
                    merged_value = value1

                merged_count = count1 + count2
                merged.append((merged_value, merged_count))

                i += 2
                
                continue

        merged.append(values_counts[i])
        
        i += 1
        
    return merged

###################################################################################

def merge_escore_notes_start_times(escore_notes, num_merges=1):

    new_dscore = delta_score_notes(escore_notes)

    times = [e[1] for e in new_dscore if e[1] != 0]
    times_counts = sorted(Counter(times).most_common())

    prev_counts = []
    new_times_counts = times_counts
    
    mcount = 0
    
    while prev_counts != new_times_counts:
        prev_counts = new_times_counts
        new_times_counts = merge_adjacent_pairs(new_times_counts)
        
        mcount += 1

        if mcount == num_merges:
            break

    gtimes = [r[0] for r in new_times_counts]

    for e in new_dscore:
        if e[1] > 0:
            e[1] = find_closest_value(gtimes, e[1])[0]
            e[2] -= num_merges

    return delta_score_to_abs_score(new_dscore)

###################################################################################

def multi_instrumental_escore_notes_tokenized(escore_notes, compress_seq=False):

    melody_chords = []

    pe = escore_notes[0]
    
    for i, e in enumerate(escore_notes):
    
        dtime = max(0, min(255, e[1]-pe[1]))
        
        dur = max(0, min(255, e[2]))
        
        cha = max(0, min(15, e[3]))
    
        if cha == 9:
          pat = 128
        
        else:
          pat = max(0, min(127, e[6]))
        
        ptc = max(0, min(127, e[4]))
        
        vel = max(8, min(127, e[5]))
        velocity = round(vel / 15)-1
        
        dur_vel = (8 * dur) + velocity
        pat_ptc = (129 * pat) + ptc

        if compress_seq:
            if dtime != 0 or i == 0:
                melody_chords.extend([dtime, dur_vel+256, pat_ptc+2304])

            else:
                melody_chords.extend([dur_vel+256, pat_ptc+2304])

        else:
            melody_chords.extend([dtime, dur_vel+256, pat_ptc+2304])
        
        pe = e

    return melody_chords

###################################################################################

def merge_counts(data, return_lists=True):
    
    merged = defaultdict(int)
    
    for value, count in data:
        merged[value] += count

    if return_lists:
        return [[k, v] for k, v in merged.items()]

    else:
        return list(merged.items())
    
###################################################################################

def convert_escore_notes_pitches_chords_signature(signature, convert_to_full_chords=True):

    if convert_to_full_chords:
        SRC_CHORDS = ALL_CHORDS_SORTED
        TRG_CHORDS = ALL_CHORDS_FULL

    else:
        SRC_CHORDS = ALL_CHORDS_FULL
        TRG_CHORDS = ALL_CHORDS_SORTED

    cdiff = len(TRG_CHORDS) - len(SRC_CHORDS)

    pitches_counts = [c for c in signature if -1 < c[0] < 128]
    chords_counts = [c for c in signature if 127 < c[0] < len(SRC_CHORDS)+128]
    drums_counts = [[c[0]+cdiff, c[1]] for c in signature if len(SRC_CHORDS)+127 < c[0] < len(SRC_CHORDS)+256]
    bad_chords_count = [c for c in signature if c[0] == -1]

    new_chords_counts = []
    
    for c in chords_counts:
        tones_chord = SRC_CHORDS[c[0]-128]

        if tones_chord not in TRG_CHORDS:
            tones_chord = check_and_fix_tones_chord(tones_chord, use_full_chords=convert_to_full_chords)
            bad_chords_count[0][1] += 1
            
        new_chords_counts.append([TRG_CHORDS.index(tones_chord)+128, c[1]])

    return pitches_counts + merge_counts(new_chords_counts) + drums_counts + bad_chords_count

###################################################################################

def convert_bytes_in_nested_list(lst, encoding='utf-8', errors='ignore'):
    
    new_list = []
    
    for item in lst:
        if isinstance(item, list):
            new_list.append(convert_bytes_in_nested_list(item))
            
        elif isinstance(item, bytes):
            new_list.append(item.decode(encoding, errors=errors))
            
        else:
            new_list.append(item)
            
    return new_list

###################################################################################

def mult_pitches(pitches, min_oct=4, max_oct=6):
    
    tones_chord = sorted(set([p % 12 for p in pitches]))

    mult_ptcs = []

    for t in tones_chord:
        for i in range(min_oct, max_oct):
            mult_ptcs.append((i*12)+t)

    return mult_ptcs

###################################################################################

def find_next(pitches, cur_ptc):

    i = 0

    for i, p in enumerate(pitches):
        if p != cur_ptc:
            break

    return i

###################################################################################

def ordered_groups_unsorted(data, key_index):
   
    def keyfunc(sublist):
        return sublist[key_index]
    
    groups = []
    
    for key, group in groupby(data, key=keyfunc):
        groups.append((key, list(group)))
        
    return groups

###################################################################################

def ordered_groups(data, ptc_idx, pat_idx):
    
    groups = OrderedDict()
    
    for sublist in data:
        key = tuple([sublist[ptc_idx], sublist[pat_idx]])
        
        if key not in groups:
            groups[key] = []
            
        groups[key].append(sublist)
    
    return list(groups.items())

###################################################################################

def merge_melody_notes(escore_notes, pitches_idx=4, max_dur=255, last_dur=128):

    groups = ordered_groups_unsorted(escore_notes, pitches_idx)

    merged_melody_notes = []

    for i, (k, g) in enumerate(groups[:-1]):
        
        if len(g) == 1:
            merged_melody_notes.extend(g)

        else:
            dur = min(max_dur, groups[i+1][1][0][1] - g[0][1])
            
            merged_melody_notes.append(['note',
                                        g[0][1],
                                        dur,
                                        g[0][3],
                                        g[0][4],
                                        g[0][5],
                                        g[0][6]
                                       ])
 
    merged_melody_notes.append(['note',
                                groups[-1][1][0][1],
                                last_dur,
                                groups[-1][1][0][3],
                                groups[-1][1][0][4],
                                groups[-1][1][0][5],
                                groups[-1][1][0][6]
                               ])
            
    return merged_melody_notes

###################################################################################

def add_expressive_melody_to_enhanced_score_notes(escore_notes,
                                                  melody_start_chord=0,
                                                  melody_prime_pitch=60,
                                                  melody_step=1,
                                                  melody_channel=3,
                                                  melody_patch=40,
                                                  melody_notes_max_duration=255,
                                                  melody_last_note_dur=128,
                                                  melody_clip_max_min_durs=[],
                                                  melody_max_velocity=120,
                                                  acc_max_velocity=90,
                                                  return_melody=False
                                                  ):


    score = copy.deepcopy(escore_notes)

    adjust_score_velocities(score, acc_max_velocity)

    cscore = chordify_score([1000, score])

    melody_pitches = [melody_prime_pitch]
    
    for i, c in enumerate(cscore[melody_start_chord:]):
        
        if i % melody_step == 0:
        
            pitches = [e[4] for e in c if e[3] != 9]
        
            if pitches:
                cptc = find_closest_value(mult_pitches(pitches), melody_pitches[-1])[0]
                melody_pitches.append(cptc)
    
    song_f = []
    mel_f = []
    
    idx = 1
    
    for i, c in enumerate(cscore[:-melody_step]):
        pitches = [e[4] for e in c if e[3] != 9]
    
        if pitches and i >= melody_start_chord and i % melody_step == 0:
            dur = min(cscore[i+melody_step][0][1] - c[0][1], melody_notes_max_duration)
            
            mel_f.append(['note', 
                          c[0][1], 
                          dur, 
                          melody_channel, 
                          60+(melody_pitches[idx] % 24), 
                          100 + ((melody_pitches[idx] % 12) * 2), 
                          melody_patch
                         ])
            idx += 1
            
        song_f.extend(c)
        
    song_f.extend(flatten(cscore[-melody_step:]))
    
    if len(melody_clip_max_min_durs) == 2:
        for e in mel_f:
            if e[2] >= melody_clip_max_min_durs[0]:
                e[2] = melody_clip_max_min_durs[1]

    adjust_score_velocities(mel_f, melody_max_velocity)
    
    merged_melody_notes = merge_melody_notes(mel_f,
                                             max_dur=melody_notes_max_duration,
                                             last_dur=melody_last_note_dur
                                             )

    song_f = sorted(merged_melody_notes + song_f,
                    key=lambda x: x[1]
                   )

    if return_melody:
        return mel_f

    else:
        return song_f

###################################################################################
    
def list_md5_hash(ints_list):
    
    arr = array('H', ints_list)
    binary_data = arr.tobytes()
    
    return hashlib.md5(binary_data).hexdigest()

###################################################################################

def fix_escore_notes_durations(escore_notes,
                               min_notes_gap=1,
                               min_notes_dur=1,
                               times_idx=1,
                               durs_idx=2,
                               channels_idx = 3, 
                               pitches_idx=4,
                               patches_idx=6
                              ):

    notes = [e for e in escore_notes if e[channels_idx] != 9]
    drums = [e for e in escore_notes if e[channels_idx] == 9]
    
    escore_groups = ordered_groups(notes, pitches_idx, patches_idx)

    merged_score = []

    for k, g in escore_groups:
        if len(g) > 2:
            fg = fix_monophonic_score_durations(g, 
                                                min_notes_gap=min_notes_gap, 
                                                min_notes_dur=min_notes_dur
                                               )
            merged_score.extend(fg)

        elif len(g) == 2:

            if g[0][times_idx]+g[0][durs_idx] >= g[1][times_idx]:
                g[0][durs_idx] = max(1, g[1][times_idx] - g[0][times_idx] - min_notes_gap)
                
            merged_score.extend(g)

        else:
            merged_score.extend(g)

    return sorted(merged_score + drums, key=lambda x: x[times_idx])

###################################################################################

def create_nested_chords_tree(chords_list):
    
    tree = {}
    
    for chord in chords_list:
        
        node = tree
        
        for semitone in chord:
            if semitone not in node:
                node[semitone] = {}
                
            node = node[semitone]
            
        node.setdefault(-1, []).append(chord)
        
    return tree

###################################################################################

def get_chords_with_prefix(nested_chords_tree, prefix):
   
    node = nested_chords_tree
    
    for semitone in prefix:
        if semitone in node:
            node = node[semitone]
            
        else:
            return []

    collected_chords = []
    
    def recursive_collect(subnode):
        if -1 in subnode:
            collected_chords.extend(subnode[-1])
            
        for key, child in subnode.items():
            if key != -1:
                recursive_collect(child)
                
    recursive_collect(node)
    
    return collected_chords

###################################################################################

def get_chords_by_semitones(chords_list, chord_semitones):

    query_set = set(chord_semitones)
    results = []

    for chord in chords_list:
        
        chord_set = set(chord)
        
        if query_set.issubset(chord_set):
            results.append(sorted(set(chord)))
            
    return results

###################################################################################

def remove_duplicate_pitches_from_escore_notes(escore_notes, 
                                               pitches_idx=4, 
                                               patches_idx=6, 
                                               return_dupes_count=False
                                              ):
    
    cscore = chordify_score([1000, escore_notes])

    new_escore = []

    bp_count = 0

    for c in cscore:
        
        cho = []
        seen = []

        for cc in c:
            if [cc[pitches_idx], cc[patches_idx]] not in seen:
                cho.append(cc)
                seen.append([cc[pitches_idx], cc[patches_idx]])

            else:
                bp_count += 1

        new_escore.extend(cho)
        
    if return_dupes_count:
        return bp_count
        
    else:
        return new_escore

###################################################################################
    
def chunks_shuffle(lst,
                   min_len=1,
                   max_len=3,
                   seed=None
                   ):
    
    rnd = random.Random(seed)
    chunks = []
    i, n = 0, len(lst)

    while i < n:
        size = rnd.randint(min_len, max_len)
        size = min(size, n - i)
        chunks.append(lst[i : i + size])
        i += size

    rnd.shuffle(chunks)

    flattened = []
    for chunk in chunks:
        flattened.extend(chunk)

    return flattened

###################################################################################

def convert_bytes_in_nested_list(lst, 
                                 encoding='utf-8', 
                                 errors='ignore',
                                 return_changed_events_count=False
                                ):
    
    new_list = []

    ce_count = 0
    
    for item in lst:
        if isinstance(item, list):
            new_list.append(convert_bytes_in_nested_list(item))
            
        elif isinstance(item, bytes):
            new_list.append(item.decode(encoding, errors=errors))
            ce_count += 1
            
        else:
            new_list.append(item)
            
    if return_changed_events_count:       
        return new_list, ce_count

    else:
        return new_list
    
###################################################################################
    
def find_deepest_midi_dirs(roots,
                           marker_file="midi_score.mid",
                           suffixes=None,
                           randomize=False,
                           seed=None,
                           verbose=False
                          ):
    
    try:
        iter(roots)
        if isinstance(roots, (str, Path)):
            root_list = [roots]
        else:
            root_list = list(roots)
            
    except TypeError:
        root_list = [roots]

    if isinstance(marker_file, (list, tuple)):
        patterns = [p.lower() for p in marker_file if p]
        
    else:
        patterns = [marker_file.lower()] if marker_file else []

    allowed = {s.lower() for s in (suffixes or ['.mid', '.midi', '.kar'])}

    if verbose:
        print("Settings:")
        print("  Roots:", [str(r) for r in root_list])
        print("  Marker patterns:", patterns or "<no marker filter>")
        print("  Allowed suffixes:", allowed)
        print(f"  Randomize={randomize}, Seed={seed}")

    results = defaultdict(list)
    rng = random.Random(seed)

    for root in root_list:

        root_path = Path(root)
        
        if not root_path.is_dir():
            print(f"Warning: '{root_path}' is not a valid directory, skipping.")
            continue

        if verbose:
            print(f"\nScanning root: {str(root_path)}")

        all_dirs = list(root_path.rglob("*"))
        dirs_iter = tqdm.tqdm(all_dirs, desc=f"Dirs in {root_path.name}", disable=not verbose)

        for dirpath in dirs_iter:
            if not dirpath.is_dir():
                continue

            children = list(dirpath.iterdir())
            if any(child.is_dir() for child in children):
                if verbose:
                    print(f"Skipping non-leaf: {str(dirpath)}")
                continue

            files = [f for f in children if f.is_file()]
            names = [f.name.lower() for f in files]

            if patterns:
                matched = any(fnmatch(name, pat) for name in names for pat in patterns)
                if not matched:
                    if verbose:
                        print(f"No marker in: {str(dirpath)}")
                    continue
                    
                if verbose:
                    print(f"Marker found in: {str(dirpath)}")
                    
            else:
                if verbose:
                    print(f"Including leaf (no marker): {str(dirpath)}")

            for f in files:
                if f.suffix.lower() in allowed:
                    results[str(dirpath)].append(str(f))
                    
                    if verbose:
                        print(f"  Collected: {f.name}")

    all_leaves = list(results.keys())
    if randomize:
        if verbose:
            print("\nShuffling leaf directories")
            
        rng.shuffle(all_leaves)
        
    else:
        all_leaves.sort()

    final_dict = {}
    
    for leaf in all_leaves:
        file_list = results[leaf][:]
        if randomize:
            if verbose:
                print(f"Shuffling files in: {leaf}")
                
            rng.shuffle(file_list)
            
        else:
            file_list.sort()
            
        final_dict[leaf] = file_list

    if verbose:
        print("\nScan complete. Found directories:")
        for d, fl in final_dict.items():
            print(f"  {d} -> {len(fl)} files")

    return final_dict

###################################################################################

PERCUSSION_GROUPS = {
    
    1: {  # Bass Drums
        35: 'Acoustic Bass Drum',
        36: 'Bass Drum 1',
    },
    2: {  # Stick
        37: 'Side Stick',
    },
    3: {  # Snares
        38: 'Acoustic Snare',
        40: 'Electric Snare',
    },
    4: {  # Claps
        39: 'Hand Clap',
    },
    5: {  # Floor Toms
        41: 'Low Floor Tom',
        43: 'High Floor Tom',
    },
    6: {  # Hi-Hats
        42: 'Closed Hi-Hat',
        44: 'Pedal Hi-Hat',
        46: 'Open Hi-Hat',
    },
    7: {  # Toms
        45: 'Low Tom',
        47: 'Low-Mid Tom',
        48: 'Hi-Mid Tom',
        50: 'High Tom',
    },
    8: {  # Cymbals
        49: 'Crash Cymbal 1',
        51: 'Ride Cymbal 1',
        52: 'Chinese Cymbal',
        55: 'Splash Cymbal',
        57: 'Crash Cymbal 2',
        59: 'Ride Cymbal 2',
    },
    9: {  # Bells
        53: 'Ride Bell',
    },
    10: {  # Tambourine
        54: 'Tambourine',
    },
    11: {  # Cowbell
        56: 'Cowbell',
    },
    12: {  # Vibraslap
        58: 'Vibraslap',
    },
    13: {  # Bongos
        60: 'Hi Bongo',
        61: 'Low Bongo',
    },
    14: {  # Congas
        62: 'Mute Hi Conga',
        63: 'Open Hi Conga',
        64: 'Low Conga',
    },
    15: {  # Timbales
        65: 'High Timbale',
        66: 'Low Timbale',
    },
    16: {  # Agogô
        67: 'High Agogo',
        68: 'Low Agogo',
    },
    17: {  # Cabasa
        69: 'Cabasa',
    },
    18: {  # Maracas
        70: 'Maracas',
    },
    19: {  # Whistles
        71: 'Short Whistle',
        72: 'Long Whistle',
    },
    20: {  # Guiros
        73: 'Short Guiro',
        74: 'Long Guiro',
    },
    21: {  # Claves
        75: 'Claves',
    },
    22: {  # Wood Blocks
        76: 'Hi Wood Block',
        77: 'Low Wood Block',
    },
    23: {  # Cuica
        78: 'Mute Cuica',
        79: 'Open Cuica',
    },
    24: {  # Triangles
        80: 'Mute Triangle',
        81: 'Open Triangle',
    },
}

###################################################################################

def escore_notes_to_expanded_binary_matrix(escore_notes, 
                                           channel=0, 
                                           patch=0,
                                           flip_matrix=False,
                                           reverse_matrix=False,
                                           encode_velocities=True
                                          ):

  escore = [e for e in escore_notes if e[3] == channel and e[6] == patch]

  if escore:
    last_time = escore[-1][1]
    last_notes = [e for e in escore if e[1] == last_time]
    max_last_dur = max([e[2] for e in last_notes])

    time_range = last_time+max_last_dur

    escore_matrix = []

    escore_matrix = [[(0, 0)] * 128 for _ in range(time_range)]

    for note in escore:

        etype, time, duration, chan, pitch, velocity, pat = note

        time = max(0, time)
        duration = max(1, duration)
        chan = max(0, min(15, chan))
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))
        pat = max(0, min(128, pat))

        if channel == chan and patch == pat:

          count = 0
          
          for t in range(time, min(time + duration, time_range)):
            if encode_velocities:
                escore_matrix[t][pitch] = velocity, count

            else:
                escore_matrix[t][pitch] = 1, count
            count += 1

    if flip_matrix:

      temp_matrix = []

      for m in escore_matrix:
        temp_matrix.append(m[::-1])

      escore_matrix = temp_matrix

    if reverse_matrix:
      escore_matrix = escore_matrix[::-1]

    return escore_matrix

  else:
    return None

###################################################################################

def transpose_list(lst):
    return [list(row) for row in zip(*lst)]

###################################################################################

def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

###################################################################################

def flip_list_rows(lst):
    return [row[::-1] for row in lst]

###################################################################################

def flip_list_columns(lst):
    return lst[::-1]

###################################################################################

def exists(sub, lst):
    sub_len = len(sub)
    return any(lst[i:i + sub_len] == sub for i in range(len(lst) - sub_len + 1))

###################################################################################

def exists_noncontig(sub, lst):
    it = iter(lst)
    return all(x in it for x in sub)

###################################################################################

def exists_ratio(sub, lst, ratio):
    matches = sum(x in set(lst) for x in sub)
    return matches / len(sub) >= ratio
    
###################################################################################

def top_k_list_value(lst, k, reverse=True):
    return sorted(lst, reverse=reverse)[k]

###################################################################################

def top_k_list_values(lst, k, reverse=True):
    return sorted(lst, reverse=reverse)[:k]

###################################################################################

def concat_rows(lst_A, lst_B):
    return [a + b for a, b in zip(lst_A, lst_B)]

###################################################################################

def concat_cols(lst_A, lst_B):
    return [[ra + rb for ra, rb in zip(a, b)] for a, b in zip(lst_A, lst_B)]

###################################################################################

def chunk_by_threshold_mode(nums, threshold=0, normalize=False):

    if not nums:
        return []

    chunks = []
    chunk = []
    freq = defaultdict(int)
    max_freq = 0
    mode_val = None

    def try_add_and_validate(value):

        nonlocal max_freq, mode_val

        chunk.append(value)
        freq[value] += 1
        new_max_freq = max_freq
        candidate_mode = mode_val

        if freq[value] > new_max_freq:
            new_max_freq = freq[value]
            candidate_mode = value

        mode = candidate_mode
        valid = True

        for v in chunk:
            if abs(v - mode) > threshold:
                valid = False
                break

        if not valid:
            
            chunk.pop()
            freq[value] -= 1
            if freq[value] == 0:
                del freq[value]
                
            return False

        max_freq = new_max_freq
        mode_val = mode
        return True

    for num in nums:
        if not chunk:
            chunk.append(num)
            freq[num] = 1
            mode_val = num
            max_freq = 1
            
        else:
            if not try_add_and_validate(num):
                if normalize:
                    normalized_chunk = [mode_val] * len(chunk)
                    chunks.append(normalized_chunk)
                
                else:
                    chunks.append(chunk[:])

                chunk.clear()
                freq.clear()
                
                chunk.append(num)
                freq[num] = 1
                mode_val = num
                max_freq = 1

    if chunk:
        if normalize:
            normalized_chunk = [mode_val] * len(chunk)
            chunks.append(normalized_chunk)
            
        else:
            chunks.append(chunk)

    return chunks

###################################################################################

def proportional_adjust(values, target_sum, threshold):

    n = len(values)
    if n == 0:
        return []

    locked_idx = [i for i, v in enumerate(values) if v < threshold]
    adj_idx    = [i for i in range(n) if i not in locked_idx]

    locked_sum       = sum(values[i] for i in locked_idx)
    adj_original_sum = sum(values[i] for i in adj_idx)
    adj_target_sum   = target_sum - locked_sum

    def _proportional_scale(idxs, original, target):

        scaled_vals = {i: original[i] * (target / sum(original[i] for i in idxs))
                       if sum(original[i] for i in idxs) > 0 else 0
                       for i in idxs}
        
        floored = {i: math.floor(scaled_vals[i]) for i in idxs}
        rem = target - sum(floored.values())

        fracs = sorted(
            ((scaled_vals[i] - floored[i], i) for i in idxs),
            key=lambda x: (x[0], -x[1]),
            reverse=True
        )
        
        for _, idx in fracs[:rem]:
            floored[idx] += 1
            
        result = original.copy()
        
        for i in idxs:
            result[i] = floored[i]
            
        return result

    if not adj_idx:
        if locked_sum == target_sum:
            return values.copy()

        return _proportional_scale(locked_idx, values, target_sum)

    if adj_target_sum < 0:
        return _proportional_scale(range(n), values, target_sum)

    if adj_original_sum == 0:
        base = adj_target_sum // len(adj_idx)
        rem  = adj_target_sum - base * len(adj_idx)
        result = values.copy()
        
        for j, idx in enumerate(sorted(adj_idx)):
            increment = base + (1 if j < rem else 0)
            result[idx] = values[idx] + increment
            
        return result

    result = values.copy()
    scaled = {i: values[i] * (adj_target_sum / adj_original_sum) for i in adj_idx}
    floored = {i: math.floor(scaled[i]) for i in adj_idx}
    floor_sum = sum(floored.values())
    rem = adj_target_sum - floor_sum

    fracs = sorted(
        ((scaled[i] - floored[i], i) for i in adj_idx),
        key=lambda x: (x[0], -x[1]),
        reverse=True
    )
    
    for _, idx in fracs[:rem]:
        floored[idx] += 1

    for i in adj_idx:
        result[i] = floored[i]

    return result

###################################################################################

def advanced_align_escore_notes_to_bars(escore_notes, 
                                        bar_dtime=200,
                                        dtimes_adj_thresh=4,
                                        min_dur_gap=0
                                       ):

    #========================================================

    escore_notes = recalculate_score_timings(escore_notes)

    cscore = chordify_score([1000, escore_notes])

    #========================================================

    dtimes = [0] + [min(199, b[1]-a[1]) for a, b in zip(escore_notes[:-1], escore_notes[1:]) if b[1]-a[1] != 0]

    score_times = sorted(set([e[1] for e in escore_notes]))

    #========================================================

    dtimes_chunks = []
    
    time = 0
    dtime = []
    
    for i, dt in enumerate(dtimes):
        time += dt
        dtime.append(dt)

        if time >= bar_dtime:
            dtimes_chunks.append(dtime)
            
            time = 0
            dtime = []
    
    dtimes_chunks.append(dtime)

    #========================================================

    fixed_times = []
    
    time = 0
    
    for i, dt in enumerate(dtimes_chunks):
    
        adj_dt = proportional_adjust(dt, 
                                     bar_dtime, 
                                     dtimes_adj_thresh
                                    )
    
        for t in adj_dt:
    
            time += t
    
            fixed_times.append(time)

    #========================================================

    output_score = []
    
    for i, c in enumerate(cscore):
        
        cc = copy.deepcopy(c)
        time = fixed_times[i]
    
        for e in cc:
            e[1] = time
    
            output_score.append(e)

    #========================================================

    output_score = fix_escore_notes_durations(output_score, 
                                              min_notes_gap=min_dur_gap
                                             )

    #========================================================

    return output_score

###################################################################################

def check_monophonic_melody(escore_notes, 
                            times_idx=1, 
                            durs_idx=2
                           ):

    bcount = 0
    
    for i in range(len(escore_notes)-1):
        if escore_notes[i][times_idx]+escore_notes[i][durs_idx] > escore_notes[i+1][times_idx]:
            bcount += 1

    return bcount / len(escore_notes)

###################################################################################

def longest_common_chunk(list1, list2):
    
    base, mod = 257, 10**9 + 7
    max_len = min(len(list1), len(list2))
    
    def get_hashes(seq, size):

        h, power = 0, 1
        hashes = set()
        
        for i in range(size):
            h = (h * base + seq[i]) % mod
            power = (power * base) % mod
            
        hashes.add(h)
        
        for i in range(size, len(seq)):
            h = (h * base - seq[i - size] * power + seq[i]) % mod
            hashes.add(h)
            
        return hashes

    def find_match(size):

        hashes2 = get_hashes(list2, size)
        h, power = 0, 1
        
        for i in range(size):
            h = (h * base + list1[i]) % mod
            power = (power * base) % mod
            
        if h in hashes2:
            return list1[:size]
            
        for i in range(size, len(list1)):
            h = (h * base - list1[i - size] * power + list1[i]) % mod
            if h in hashes2:
                return list1[i - size + 1:i + 1]
                
        return []

    left, right = 0, max_len
    result = []
    
    while left <= right:
        mid = (left + right) // 2
        chunk = find_match(mid)
        
        if chunk:
            result = chunk
            left = mid + 1
        else:
            
            right = mid - 1
            
    return result

###################################################################################

def detect_plateaus(data, min_len=2, tol=0.0):

    plateaus = []
    n = len(data)
    if n < min_len:
        return plateaus

    min_deque = deque()
    max_deque = deque()

    start = 0
    idx = 0

    while idx < n:
        v = data[idx]

        if not isinstance(v, (int, float)) or math.isnan(v):

            if idx - start >= min_len:
                plateaus.append(data[start:idx])

            idx += 1
            start = idx
            min_deque.clear()
            max_deque.clear()
            
            continue

        while max_deque and data[max_deque[-1]] <= v:
            max_deque.pop()
            
        max_deque.append(idx)

        while min_deque and data[min_deque[-1]] >= v:
            min_deque.pop()
            
        min_deque.append(idx)

        if data[max_deque[0]] - data[min_deque[0]] > tol:

            if idx - start >= min_len:
                plateaus.append(data[start:idx])

            start = idx
            
            min_deque.clear()
            max_deque.clear()

            max_deque.append(idx)
            min_deque.append(idx)

        idx += 1

    if n - start >= min_len:
        plateaus.append(data[start:n])

    return plateaus

###################################################################################

def alpha_str_to_toks(s, shift=0, add_seos=False):

    tokens = []

    if add_seos:
        tokens = [53+shift]
    
    for char in s:
        if char == ' ':
            tokens.append(52+shift)
            
        elif char.isalpha():
            base = 0 if char.isupper() else 26
            offset = ord(char.upper()) - ord('A')
            token = (base + offset + shift) % 52  # wrap A–Z/a–z
            tokens.append(token)
            
    if add_seos:       
        tokens.append(53+shift)
        
    return tokens

###################################################################################

def toks_to_alpha_str(tokens, shift=0, sep=''):

    chars = []
    
    for token in tokens:
        if token == 53+shift:
            continue
            
        elif token == 52+shift:
            chars.append(' ')
            
        elif 0 <= token <= 25:
            original = (token - shift) % 52
            chars.append(chr(ord('A') + original))
            
        elif 26 <= token <= 51:
            original = (token - shift) % 52
            chars.append(chr(ord('a') + (original - 26)))

    return sep.join(chars)

###################################################################################

def insert_caps_newlines(text):

    if bool(re.search(r'\b[A-Z][a-z]+\b', text)):
        pattern = re.compile(r'\s+(?=[A-Z])')

        return pattern.sub('\n', text)

###################################################################################

def insert_newlines(text, every=4):

    count = 0
    result = []

    for char in text:
        result.append(char)
        
        if char == '\n':
            count += 1
            
            if count % every == 0:
                result.append('\n')

    return ''.join(result)

###################################################################################

def symmetric_match_ratio(list_a, list_b, threshold=0):

    a_sorted = sorted(list_a)
    b_sorted = sorted(list_b)

    i, j = 0, 0
    matches = 0
    
    used_a = set()
    used_b = set()

    while i < len(a_sorted) and j < len(b_sorted):
        diff = abs(a_sorted[i] - b_sorted[j])
        
        if diff <= threshold:
            matches += 1
            used_a.add(i)
            used_b.add(j)
            i += 1
            j += 1
            
        elif a_sorted[i] < b_sorted[j]:
            i += 1
            
        else:
            j += 1

    avg_len = (len(list_a) + len(list_b)) / 2
    
    return matches / avg_len if avg_len > 0 else 0.0

###################################################################################

def escore_notes_to_chords(escore_notes, 
                           use_full_chords=False,
                           repair_bad_chords=True,
                           skip_pitches=False,
                           convert_pitches=True,
                           shift_chords=False,
                           return_tones_chords=False
                          ):

    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL

    else:
        CHORDS = ALL_CHORDS_SORTED

    sp_score = solo_piano_escore_notes(escore_notes)

    cscore = chordify_score([1000, sp_score])

    chords = []

    for c in cscore:
        pitches = sorted(set([e[4] for e in c]))

        tones_chord = sorted(set([p % 12 for p in pitches]))

        if repair_bad_chords:
            if tones_chord not in CHORDS:
                tones_chord = check_and_fix_tones_chord(tones_chord, 
                                                        use_full_chords=use_full_chords
                                                       )
            
        if return_tones_chords:
            if convert_pitches:
                chords.append(tones_chord)

            else:
                if len(pitches) > 1:
                    chords.append(tones_chord)

                else:
                    chords.append([-pitches[0]])
                    
        else:
            if skip_pitches:
                if tones_chord in CHORDS:
                    cho_tok = CHORDS.index(tones_chord)

                else:
                    cho_tok = -1

                if len(pitches) > 1:
                    chords.append(cho_tok)

            else:
                if tones_chord in CHORDS:
                    cho_tok = CHORDS.index(tones_chord)

                else:
                    cho_tok = -1

                if cho_tok != -1:
                    if convert_pitches:
                        if shift_chords:
                            if len(pitches) > 1:
                                chords.append(cho_tok+12)
        
                            else:
                                chords.append(pitches[0] % 12)
                                
                        else:
                            chords.append(cho_tok)
        
                    else:
                        if len(pitches) > 1:
                            chords.append(cho_tok+128)
        
                        else:
                            chords.append(pitches[0])

    return chords

###################################################################################

def replace_chords_in_escore_notes(escore_notes,
                                   chords_list=[-1],
                                   use_full_chords=False,
                                   use_shifted_chords=False
                                  ):

    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL

    else:
        CHORDS = ALL_CHORDS_SORTED

    if use_shifted_chords:
        shift = 12

    else:
        shift = 0

    if min(chords_list) >= 0 and max(chords_list) <= len(CHORDS)+shift:

        chords_list_iter = cycle(chords_list)

        nd_score = [e for e in escore_notes if e[3] != 9]
        d_score = [e for e in escore_notes if e[3] == 9]
    
        cscore = chordify_score([1000, nd_score])
    
        new_score = []
    
        for i, c in enumerate(cscore):

            cur_chord = next(chords_list_iter)
    
            cc = copy.deepcopy(c)
    
            if use_shifted_chords:
                if cur_chord < 12:
                    sub_tones_chord = [cur_chord]
    
                else:
                    sub_tones_chord = CHORDS[cur_chord-12]
            else:
                sub_tones_chord = CHORDS[cur_chord]
                
            stcho = cycle(sub_tones_chord)

            if len(sub_tones_chord) > len(c):
                cc = [copy.deepcopy(e) for e in cc for _ in range(len(sub_tones_chord))]
                
            pseen = []
    
            for e in cc:
                st = next(stcho)
                new_pitch = ((e[4] // 12) * 12) + st

                if [new_pitch, e[6]] not in pseen:
                    e[4] = new_pitch
                    
                    new_score.append(e)
                    pseen.append([new_pitch, e[6]])
                    
        final_score = sorted(new_score+d_score, key=lambda x: x[1])

        return final_score

    else:
        return []

###################################################################################

class Cell:
    def __init__(self, cost, segments, gaps, prev_dir):
        self.cost = cost
        self.segments = segments
        self.gaps = gaps
        self.prev_dir = prev_dir

def align_integer_lists(seq1, seq2):

    n, m = len(seq1), len(seq2)
    
    if n == 0:
        return [None]*m, seq2.copy(), sum(abs(x) for x in seq2)
    if m == 0:
        return seq1.copy(), [None]*n, sum(abs(x) for x in seq1)

    priority = {'diag': 0, 'up': 1, 'left': 2}

    dp = [
        [Cell(cost=math.inf, segments=math.inf, gaps=math.inf, prev_dir='') for _ in range(m+1)]
        for _ in range(n+1)
    ]
    dp[0][0] = Cell(cost=0, segments=0, gaps=0, prev_dir='')

    for i in range(1, n+1):
        prev = dp[i-1][0]
        new_cost = prev.cost + abs(seq1[i-1])
        new_seg  = prev.segments + (1 if prev.prev_dir != 'up' else 0)
        new_gaps = prev.gaps + 1
        dp[i][0]  = Cell(new_cost, new_seg, new_gaps, 'up')

    for j in range(1, m+1):
        prev = dp[0][j-1]
        new_cost = prev.cost + abs(seq2[j-1])
        new_seg  = prev.segments + (1 if prev.prev_dir != 'left' else 0)
        new_gaps = prev.gaps + 1
        dp[0][j] = Cell(new_cost, new_seg, new_gaps, 'left')

    for i in range(1, n+1):
        for j in range(1, m+1):
            a, b = seq1[i-1], seq2[j-1]

            c0 = dp[i-1][j-1]
            cand_diag = Cell(
                cost     = c0.cost + abs(a - b),
                segments = c0.segments,
                gaps     = c0.gaps,
                prev_dir = 'diag'
            )

            c1 = dp[i-1][j]
            seg1 = c1.segments + (1 if c1.prev_dir != 'up' else 0)
            cand_up = Cell(
                cost     = c1.cost + abs(a),
                segments = seg1,
                gaps     = c1.gaps + 1,
                prev_dir = 'up'
            )

            c2 = dp[i][j-1]
            seg2 = c2.segments + (1 if c2.prev_dir != 'left' else 0)
            cand_left = Cell(
                cost     = c2.cost + abs(b),
                segments = seg2,
                gaps     = c2.gaps + 1,
                prev_dir = 'left'
            )

            best = min(
                (cand_diag, cand_up, cand_left),
                key=lambda c: (c.cost, c.segments, c.gaps, priority[c.prev_dir])
            )
            dp[i][j] = best

    aligned1 = []
    aligned2 = []
    i, j = n, m
    
    while i > 0 or j > 0:
        cell = dp[i][j]
        
        if cell.prev_dir == 'diag':
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i, j = i-1, j-1
            
        elif cell.prev_dir == 'up':
            aligned1.append(seq1[i-1])
            aligned2.append(None)
            i -= 1
            
        else:
            aligned1.append(None)
            aligned2.append(seq2[j-1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    
    total_cost = int(dp[n][m].cost)
    
    return aligned1, aligned2, total_cost

###################################################################################

def most_common_delta_time(escore_notes):
    
    dscore = delta_score_notes(escore_notes)
    
    dtimes = [t[1] for t in dscore if t[1] != 0]
    
    cdtime, count = Counter(dtimes).most_common(1)[0]

    return [cdtime, count / len(dtimes)]

###################################################################################

def delta_tones(escore_notes, 
                ptcs_idx=4
               ):
    
    pitches = [p[ptcs_idx] for p in escore_notes]
    tones = [p % 12 for p in pitches]

    return [b-a for a, b in zip(tones[:-1], tones[1:])]
    
###################################################################################

def find_divisors(val, 
                  reverse=False
                 ):
  
    if val == 0:
        return []

    n = abs(val)
    divisors = set()

    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)

    return sorted(divisors, reverse=reverse)

###################################################################################

def find_common_divisors(values, 
                         reverse=False
                        ):
   
    if not values:
        return []

    non_zero = [abs(v) for v in values if v != 0]
    if not non_zero:
        return []

    overall_gcd = reduce(gcd, non_zero)

    divisors = set()
    
    for i in range(1, int(overall_gcd**0.5) + 1):
        if overall_gcd % i == 0:
            divisors.add(i)
            divisors.add(overall_gcd // i)

    return sorted(divisors, reverse=reverse)

###################################################################################

def strings_dict(list_of_strings, 
                 verbose=False
                ):

    str_set = set()
    
    for st in tqdm.tqdm(list_of_strings, disable=not verbose):
        for cha in st:
            str_set.add(cha)

    str_lst = sorted(str_set)

    str_dic = dict(zip(str_lst, range(len(str_lst))))
    rev_str_dic = {v: k for k, v in str_dic.items()}

    return str_dic, rev_str_dic

###################################################################################

def chords_common_tones_chain(chords, 
                              use_full_chords=False
                             ):

    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL

    else:
        CHORDS = ALL_CHORDS_SORTED

    tones_chords = [CHORDS[c] for c in chords if 0 <= c < len(CHORDS)]

    n = len(tones_chords)
    
    if not tones_chords:
        return []
        
    if n < 2:
        return tones_chords

    result = []

    for i in range(n):
        if i == 0:
            common = set(tones_chords[0]) & set(tones_chords[1])
            
        elif i == n - 1:
            common = set(tones_chords[n - 2]) & set(tones_chords[n - 1])
            
        else:
            common = set(tones_chords[i - 1]) & set(tones_chords[i]) & set(tones_chords[i + 1])

        result.append(min(common) if common else -1)

    return result

###################################################################################

def tones_chord_to_int(tones_chord, 
                       reverse_bits=True
                      ):

    cbits = tones_chord_to_bits(tones_chord, 
                                reverse=reverse_bits
                               )

    cint = bits_to_int(cbits)
    
    return cint

###################################################################################

def int_to_tones_chord(integer, 
                       reverse_bits=True
                      ):

    integer = integer % 4096

    cbits = int_to_bits(integer)

    if reverse_bits:
        cbits.reverse()

    tones_chord = bits_to_tones_chord(cbits)

    return tones_chord

###################################################################################

def fix_bad_chords_in_escore_notes(escore_notes,
                                   use_full_chords=False,
                                   return_bad_chords_count=False
                                  ):

    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL

    else:
        CHORDS = ALL_CHORDS_SORTED

    bcount = 0

    if escore_notes:

        chords = chordify_score([1000, escore_notes])

        fixed_chords = []
    
        for c in chords:
            c.sort(key=lambda x: x[3])
    
            if len(c) > 1:
    
                groups = groupby(c, key=lambda x: x[3])
        
                for cha, gr in groups:

                    if cha != 9:
    
                        gr = list(gr)
                        
                        tones_chord = sorted(set([p[4] % 12 for p in gr]))
            
                        if tones_chord not in CHORDS:
                            tones_chord = check_and_fix_tones_chord(tones_chord, 
                                                                    use_full_chords=use_full_chords
                                                                   )
                    
                            bcount += 1
            
                        ngr = []
                        
                        for n in gr:
                            if n[4] % 12 in tones_chord:
                                ngr.append(n)
            
                        fixed_chords.extend(ngr)

                    else:
                        fixed_chords.extend(gr)
                        
    
            else:
                fixed_chords.extend(c)
                
        fixed_chords.sort(key=lambda x: (x[1], -x[4]))

        if return_bad_chords_count:
            return fixed_chords, bcount

        else:
            return fixed_chords
            
    else:
        if return_bad_chords_count:
            return escore_notes, bcount

        else:
            return escore_notes
        
###################################################################################
        
def remove_events_from_escore_notes(escore_notes,
                                    ele_idx=2,
                                    ele_vals=[1],
                                    chan_idx=3,
                                    skip_drums=True
                                    ):

    new_escore_notes = []
    
    for e in escore_notes:
        if skip_drums:
            if e[ele_idx] not in ele_vals or e[chan_idx] == 9:
                new_escore_notes.append(e)

        else:
            if e[ele_idx] not in ele_vals:
                new_escore_notes.append(e)

    return new_escore_notes
        
###################################################################################

def flatten_spikes(arr):
    
    if len(arr) < 3:
        return arr[:]

    result = arr[:]
    
    for i in range(1, len(arr) - 1):
        prev, curr, next_ = arr[i - 1], arr[i], arr[i + 1]

        if (prev <= next_ and (curr > prev and curr > next_)) or \
           (prev >= next_ and (curr < prev and curr < next_)):
            result[i] = max(min(prev, next_), min(max(prev, next_), curr))
            
    return result
        
###################################################################################

def flatten_spikes_advanced(arr, window=1):
    
    if len(arr) < 3:
        return arr[:]

    result = arr[:]
    n = len(arr)

    def is_spike(i):
        left = arr[i - window:i]
        right = arr[i + 1:i + 1 + window]
        
        if not left or not right:
            return False

        avg_left = sum(left) / len(left)
        avg_right = sum(right) / len(right)

        if arr[i] > avg_left and arr[i] > avg_right:
            return True

        if arr[i] < avg_left and arr[i] < avg_right:
            return True
        
        return False

    for i in range(window, n - window):
        if is_spike(i):
            neighbors = arr[i - window:i] + arr[i + 1:i + 1 + window]
            result[i] = int(sorted(neighbors)[len(neighbors) // 2])

    return result
        
###################################################################################

def add_smooth_melody_to_enhanced_score_notes(escore_notes,
                                              melody_channel=3,
                                              melody_patch=40,
                                              melody_start_chord=0,
                                              min_notes_gap=0,
                                              exclude_durs=[1],
                                              adv_flattening=True,
                                              extend_durs=True,
                                              max_mel_vels=127,
                                              max_acc_vels=80,
                                              return_melody=False
                                             ):

    escore_notes1 = remove_duplicate_pitches_from_escore_notes(escore_notes)
    
    escore_notes2 = fix_escore_notes_durations(escore_notes1, 
                                               min_notes_gap=min_notes_gap
                                              )
    
    escore_notes3 = fix_bad_chords_in_escore_notes(escore_notes2)
    
    escore_notes4 = remove_events_from_escore_notes(escore_notes3, 
                                                    ele_vals=exclude_durs
                                                   )
    
    escore_notes5 = add_expressive_melody_to_enhanced_score_notes(escore_notes4,
                                                                  melody_channel=melody_channel, 
                                                                  melody_patch=melody_patch, 
                                                                  melody_start_chord=melody_start_chord,
                                                                  return_melody=True,
                                                                 )
    
    mel_score = remove_events_from_escore_notes(escore_notes5,
                                                ele_vals=exclude_durs
                                               )
    
    pitches = [p[4] for p in mel_score]
    
    if adv_flattening:
        res = flatten_spikes_advanced(pitches)

    else:
        res = flatten_spikes(pitches)
    
    mel_score3 = copy.deepcopy(mel_score)
    
    for i, e in enumerate(mel_score3):
        e[4] = res[i]
    
    mel_score3 = fix_monophonic_score_durations(merge_melody_notes(mel_score3),
                                                extend_durs=extend_durs
                                               )

    adjust_score_velocities(mel_score3, max_mel_vels)
    adjust_score_velocities(escore_notes4, max_acc_vels)

    if return_melody:
        return sorted(mel_score3, key=lambda x: (x[1], -x[4]))

    else:
        return sorted(mel_score3 + escore_notes4, key=lambda x: (x[1], -x[4]))
    
###################################################################################

def sorted_chords_to_full_chords(chords):

    cchords = []
    
    for c in chords:
        tones_chord = ALL_CHORDS_SORTED[c]

        if tones_chord not in ALL_CHORDS_FULL:
            tones_chord = check_and_fix_tones_chord(tones_chord)

        cchords.append(ALL_CHORDS_FULL.index(tones_chord))

    return cchords

###################################################################################

def full_chords_to_sorted_chords(chords):

    cchords = []
    
    for c in chords:
        tones_chord = ALL_CHORDS_FULL[c]

        if tones_chord not in ALL_CHORDS_SORTED:
            tones_chord = check_and_fix_tones_chord(tones_chord, use_full_chords=False)

        cchords.append(ALL_CHORDS_SORTED.index(tones_chord))

    return cchords

###################################################################################

def chords_to_escore_notes(chords,
                           use_full_chords=False,
                           chords_dtime=500,
                           add_melody=True,
                           add_texture=True,
                          ):

    if use_full_chords:
        CHORDS = ALL_CHORDS_FULL

    else:
        CHORDS = ALL_CHORDS_SORTED

    score = []

    dtime = 0

    dur = chords_dtime

    for c in chords:

        if add_melody:
            score.append(['note', dtime, dur, 3, CHORDS[c][0]+72, 115+CHORDS[c][0], 40])

        for cc in CHORDS[c]:
            score.append(['note', dtime, dur, 0, cc+48, 30+cc+48, 0])

            if random.randint(0, 1) and add_texture:
                score.append(['note', dtime, dur, 0, cc+60, 20+cc+60, 0])    
            
        dtime += chords_dtime

    return score

###################################################################################

print('Module loaded!')
print('=' * 70)
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the TMIDI X Python module
###################################################################################