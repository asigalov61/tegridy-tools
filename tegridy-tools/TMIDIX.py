#! /usr/bin/python3


r'''###############################################################################
###################################################################################
#
#
#	Tegridy MIDI X Module (TMIDI X / tee-midi eks)
#	Version 1.0
#
#   NOTE: TMIDI X Module starts after the partial MIDI.py module @ line 1342
#
#	Based upon MIDI.py module v.6.7. by Peter Billam / pjb.com.au
#
#	Project Los Angeles
#
#	Tegridy Code 2021
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#       Copyright 2021 Project Los Angeles / Tegridy Code
#
#       Licensed under the Apache License, Version 2.0 (the "License");
#       you may not use this file except in compliance with the License.
#       You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
#       Unless required by applicable law or agreed to in writing, software
#       distributed under the License is distributed on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#       See the License for the specific language governing permissions and
#       limitations under the License.
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
###################################################################################'''

import sys, struct, copy
Version = '6.7'
VersionDate = '20201120'

_previous_warning = ''  # 5.4
_previous_times = 0     # 5.4
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
    if not len(ba):   # 6.7
        _warn('_unshift_ber_int: no integer found')
        return ((0, b""))
    byte = ba.pop(0)
    integer = 0
    while True:
        integer += (byte & 0x7F)
        if not (byte & 0x80):
            return ((integer, ba))
        if not len(ba):
            _warn('_unshift_ber_int: no end-of-integer found')
            return ((0, ba))
        byte = ba.pop(0)
        integer <<= 7

def _clean_up_warnings():  # 5.4
    # Call this before returning from any publicly callable function
    # whenever there's a possibility that a warning might have been printed
    # by the function, or by any private functions it might have called.
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
    global _previous_times
    global _previous_warning
    if s == _previous_warning:  # 5.4
        _previous_times = _previous_times + 1
    else:
        _clean_up_warnings()
        sys.stderr.write(str(s)+"\n")
        _previous_warning = s

def _some_text_event(which_kind=0x01, text=b'some_text', text_encoding='ISO-8859-1'):
    if str(type(text)).find("'str'") >= 0:   # 6.4 test for back-compatibility
        data = bytes(text, encoding=text_encoding)
    else:
        data = bytes(text)
    return b'\xFF'+bytes((which_kind,))+_ber_compressed_int(len(data))+data

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
    event_code = -1; # used for running status
    event_count = 0;
    events = []

    while(len(trackdata)):
        # loop while there's anything to analyze ...
        eot = False   # When True, the event registrar aborts this loop
        event_count += 1

        E = []
        # E for events - we'll feed it to the event registrar at the end.

        # Slice off the delta time code, and analyze it
        [time, remainder] = _unshift_ber_int(trackdata)

        # Now let's see what we can make of the command
        first_byte = trackdata.pop(0) & 0xFF

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

            if (command == 0xF6):  #  0-byte argument
                pass
            elif (command == 0xC0 or command == 0xD0):  #  1-byte argument
                parameter = trackdata.pop(0)  # could be B
            else: # 2-byte argument could be BB or 14-bit
                parameter = (trackdata.pop(0), trackdata.pop(0))

            #################################################################
            # MIDI events

            if (command      == 0x80):
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
                E = ['key_after_touch',time,channel,parameter[0],parameter[1]]
            elif (command == 0xB0):
                if 'control_change' in exclude:
                    continue
                E = ['control_change',time,channel,parameter[0],parameter[1]]
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
                 _read_14_bit(parameter)-0x2000]
            else:
                _warn("Shouldn't get here; command="+hex(command))

        elif (first_byte == 0xFF):  # It's a Meta-Event! ##################
            #[command, length, remainder] =
            #    unpack("xCwa*", substr(trackdata, $Pointer, 6));
            #Pointer += 6 - len(remainder);
            #    # Move past JUST the length-encoded.
            command = trackdata.pop(0) & 0xFF
            [length, trackdata] = _unshift_ber_int(trackdata)
            if (command      == 0x00):
                 if (length == 2):
                     E = ['set_sequence_number',time,_twobytes2int(trackdata)]
                 else:
                     _warn('set_sequence_number: length must be 2, not '+str(length))
                     E = ['set_sequence_number', time, 0]

            elif command >= 0x01 and command <= 0x0f:   # Text events
                # 6.2 take it in bytes; let the user get the right encoding.
                # text_str = trackdata[0:length].decode('ascii','ignore')
                # text_str = trackdata[0:length].decode('ISO-8859-1')
                # 6.4 take it in bytes; let the user get the right encoding.
                text_data = bytes(trackdata[0:length])   # 6.4
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
            elif (command == 0x51): # DTime, Microseconds/Crochet
                 if length != 3:
                     _warn('set_tempo event, but length='+str(length))
                 E = ['set_tempo', time,
                      struct.unpack(">I", b'\x00'+trackdata[0:3])[0]]
            elif (command == 0x54):
                 if length != 5:   # DTime, HR, MN, SE, FR, FF
                     _warn('smpte_offset event, but length='+str(length))
                 E = ['smpte_offset',time] + list(struct.unpack(">BBBBB",trackdata[0:5]))
            elif (command == 0x58):
                 if length != 4:   # DTime, NN, DD, CC, BB
                     _warn('time_signature event, but length='+str(length))
                 E = ['time_signature', time]+list(trackdata[0:4])
            elif (command == 0x59):
                 if length != 2:   # DTime, SF(signed), MI
                     _warn('key_signature event, but length='+str(length))
                 E = ['key_signature',time] + list(struct.unpack(">bB",trackdata[0:2]))
            elif (command == 0x7F):   # 6.4
                 E = ['sequencer_specific',time, bytes(trackdata[0:length])]
            else:
                 E = ['raw_meta_event', time, command,
                   bytes(trackdata[0:length])]   # 6.0
                 #"[uninterpretable meta-event command of length length]"
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
            #command = trackdata.pop(0)
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
        
        elif (first_byte == 0xF2):   # DTime, Beats
            #  <song position msg> ::=     F2 <data pair>
            E = ['song_position', time, _read_14_bit(trackdata[:2])]
            trackdata = trackdata[2:]

        elif (first_byte == 0xF3):   # <song select msg> ::= F3 <data singlet>
            # E = ['song_select', time, struct.unpack('>B',trackdata.pop(0))[0]]
            E = ['song_select', time, trackdata[0]]
            trackdata = trackdata[1:]
            # DTime, Thing (what?! song number?  whatever ...)

        elif (first_byte == 0xF6):   # DTime
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
            E = ['raw_data', time, trackdata[0]]   # 6.4 6.7
            trackdata = trackdata[1:]
        else:  # Fallthru.
            _warn("Aborting track.  Command-byte first_byte="+hex(first_byte))
            break
        # End of the big if-group


        ######################################################################
        #  THE EVENT REGISTRAR...
        if E and  (E[0] == 'end_track'):
            # This is the code for exceptional handling of the EOT event.
            eot = True
            if not no_eot_magic:
                if E[1] > 0:  # a null text-event to carry the delta-time
                    E = ['text_event', E[1], '']
                else:
                    E = []   # EOT with a delta-time of 0; ignore it.
        
        if E and not (E[0] in exclude):
            #if ( $exclusive_event_callback ):
            #    &{ $exclusive_event_callback }( @E );
            #else:
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
#	Version 1.0
#
#	Based upon and includes the amazing MIDI.py module v.6.7. by Peter Billam
#	pjb.com.au
#
#	Project Los Angeles
#	Tegridy Code 2021
# https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
###################################################################################

import os

import datetime

import copy

from datetime import datetime

import secrets

import random

import pickle

import csv

import tqdm

from itertools import zip_longest
from itertools import groupby
from collections import Counter

from operator import itemgetter

import sys

from abc import ABC, abstractmethod

from difflib import SequenceMatcher as SM

import statistics
import math

import matplotlib.pyplot as plt

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

def adjust_score_velocities(score, max_velocity):

    min_velocity = min([c[5] for c in score])
    max_velocity_all_channels = max([c[5] for c in score])
    min_velocity_ratio = min_velocity / max_velocity_all_channels

    max_channel_velocity = max([c[5] for c in score])
    if max_channel_velocity < min_velocity:
        factor = max_velocity / min_velocity
    else:
        factor = max_velocity / max_channel_velocity
    for i in range(len(score)):
        score[i][5] = int(score[i][5] * factor)

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

def fix_monophonic_score_durations(monophonic_score):
  
    fixed_score = []

    if monophonic_score[0][0] == 'note':

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][1]

        if note[1]+note[2] >= nmt:
          note_dur = nmt-note[1]-1
        else:
          note_dur = note[2]

        new_note = [note[0], note[1], note_dur] + note[3:]

        fixed_score.append(new_note)

      fixed_score.append(monophonic_score[-1])

    elif type(monophonic_score[0][0]) == int:

      for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]

        nmt = monophonic_score[i+1][0]

        if note[0]+note[1] >= nmt:
          note_dur = nmt-note[0]-1
        else:
          note_dur = note[1]

        new_note = [note[0], note_dur] + note[2:]

        fixed_score.append(new_note)

      fixed_score.append(monophonic_score[-1]) 

    return fixed_score

###################################################################################

from itertools import product

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
                              return_text_and_lyric_events=False
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
                                  legacy_timings=True
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

    if full_sorting:

      # Sorting by patch, reverse pitch and start-time
      esn.sort(key=lambda x: x[6])
      esn.sort(key=lambda x: x[4], reverse=True)
      esn.sort(key=lambda x: x[1])

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

def patch_enhanced_score_notes(enhanced_score_notes, 
                                default_patch=0, 
                                drums_patch=9,
                                verbose=False
                                ):
  
    #===========================================================================    
  
    enhanced_score_notes_with_patch_changes = []

    patches = [-1] * 16

    overflow_idx = -1

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
                        overflow_idx = idx
                        break

      enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    overflow_patches = []

    if overflow_idx != -1:
      for idx, e in enumerate(enhanced_score_notes[overflow_idx:]):
        if e[0] == 'note':
          if e[3] != 9:
            if e[6] not in patches:
              if e[6] not in overflow_patches:
                overflow_patches.append(e[6])
                enhanced_score_notes_with_patch_changes.append(['patch_change', e[1], e[3], e[6]])
            else:
              e[3] = patches.index(e[6])

          enhanced_score_notes_with_patch_changes.append(e)

    #===========================================================================

    patches = [p if p != -1 else default_patch for p in patches]

    patches[9] = drums_patch

    #===========================================================================

    if verbose:
      print('=' * 70)
      print('Composition patches')
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

    if dtime > max(durs):
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
                               split_durations=False
                               ):

  #=============================================================================

  aligned_escore_notes = copy.deepcopy(escore_notes)

  abs_time = 0
  nidx = 0
  delta = 0
  bcount = 0
  next_bar = [0]

  #=============================================================================

  while next_bar:

    next_bar = find_next_bar(escore_notes, bar_time, nidx, bcount)

    if next_bar:

      gescore_notes = escore_notes[nidx:next_bar[1]]
    else:
      gescore_notes = escore_notes[nidx:]

    original_timings = [delta] + [(b[1]-a[1]) for a, b in zip(gescore_notes[:-1], gescore_notes[1:])]
    adj_timings = adjust_numbers_to_sum(original_timings, bar_time)

    for t in adj_timings:

      abs_time += t

      aligned_escore_notes[nidx][1] = abs_time
      aligned_escore_notes[nidx][2] -= int(bar_time // 200)

      nidx += 1

    if next_bar:
      delta = escore_notes[next_bar[1]][1]-escore_notes[next_bar[1]-1][1]
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
                                  reverse_matrix=False
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
        velocity = max(0, min(127, velocity))
        pat = max(0, min(128, pat))

        if channel == chan and patch == pat:

          for t in range(time, min(time + duration, time_range)):

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
                                           velocity=-1
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
    
    if velocity == -1:
      vel = max(40, r[2])

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

  if return_ptcs_and_vels:
    if average_drums:
      ptcs = [e[ptcs_index] for e in escore_notes]
      vels = [e[vels_index] for e in escore_notes]
    else:
      ptcs = [e[ptcs_index] for e in escore_notes if e[chans_index] != 9]
      vels = [e[vels_index] for e in escore_notes if e[chans_index] != 9]      

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

  cscore = chordify_score([1000, escore_notes])

  sp_escore_notes = []

  for c in cscore:

    seen = []
    chord = []

    for cc in c:
      if cc[pitches_index] not in seen:

          if cc[channels_index] != 9:
            cc[channels_index] = 0
            cc[patches_index] = 0
            
            chord.append(cc)
            seen.append(cc[pitches_index])
          
          else:
            if keep_drums:
              chord.append(cc)
              seen.append(cc[pitches_index])

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
#  
# This is the end of the TMIDI X Python module
#
###################################################################################