#! /usr/bin/python3

###################################################################################

###################################################################################
#
#
#	Tegridy MIDI Module (TMIDI / tee-midi)
#	Version 2.3
#
#       NOTE: TMIDI Module starts after MIDI.py module @ line 1780
#
#	Based upon and includes the amazing MIDI.py module v.6.7. by Peter Billam
#	pjb.com.au
#
#	Project Los Angeles
#
#	Tegridy Code 2021
#
#       https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
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
#
#	MIDI.py Module v.6.7. by Peter Billam
#
#	https://pjb.com.au/
#	
#	Copyright 2020 Peter Billam
#
###################################################################################

###################################################################################

# unsupported 20091104 ...
#     ['set_sequence_number', dtime, sequence]
#     ['raw_data', dtime, raw]

# 20150914   jimbo1qaz   MIDI.py str/bytes bug report
# I found a MIDI file which had Shift-JIS titles. When midi.py decodes it as
# latin-1, it produces a string which cannot even be accessed without raising
# a UnicodeDecodeError.  Maybe, when converting raw byte strings from MIDI,
# you should keep them as bytes, not improperly decode them.  However, this
# would change the API.  (ie: text = a "string" ? of 0 or more bytes).  It
# could break compatiblity, but there's not much else you can do to fix the bug
# https://en.wikipedia.org/wiki/Shift_JIS

r'''
This module offers functions:  concatenate_scores(), grep(),
merge_scores(), mix_scores(), midi2opus(), midi2score(), opus2midi(),
opus2score(), play_score(), score2midi(), score2opus(), score2stats(),
score_type(), segment(), timeshift() and to_millisecs(),
where "midi" means the MIDI-file bytes (as can be put in a .mid file,
or piped into aplaymidi), and "opus" and "score" are list-structures
as inspired by Sean Burke's MIDI-Perl CPAN module.

Warning: Version 6.4 is not necessarily backward-compatible with
previous versions, in that text-data is now bytes, not strings.
This reflects the fact that many MIDI files have text data in
encodings other that ISO-8859-1, for example in Shift-JIS.

Download MIDI.py from   http://www.pjb.com.au/midi/free/MIDI.py
and put it in your PYTHONPATH.  MIDI.py depends on Python3.

There is also a call-compatible translation into Lua of this
module: see http://www.pjb.com.au/comp/lua/MIDI.html

The "opus" is a direct translation of the midi-file-events, where
the times are delta-times, in ticks, since the previous event.

The "score" is more human-centric; it uses absolute times, and
combines the separate note_on and note_off events into one "note"
event, with a duration:
 ['note', start_time, duration, channel, note, velocity] # in a "score"

  EVENTS (in an "opus" structure)
     ['note_off', dtime, channel, note, velocity]       # in an "opus"
     ['note_on', dtime, channel, note, velocity]        # in an "opus"
     ['key_after_touch', dtime, channel, note, velocity]
     ['control_change', dtime, channel, controller(0-127), value(0-127)]
     ['patch_change', dtime, channel, patch]
     ['channel_after_touch', dtime, channel, velocity]
     ['pitch_wheel_change', dtime, channel, pitch_wheel]
     ['text_event', dtime, text]
     ['copyright_text_event', dtime, text]
     ['track_name', dtime, text]
     ['instrument_name', dtime, text]
     ['lyric', dtime, text]
     ['marker', dtime, text]
     ['cue_point', dtime, text]
     ['text_event_08', dtime, text]
     ['text_event_09', dtime, text]
     ['text_event_0a', dtime, text]
     ['text_event_0b', dtime, text]
     ['text_event_0c', dtime, text]
     ['text_event_0d', dtime, text]
     ['text_event_0e', dtime, text]
     ['text_event_0f', dtime, text]
     ['end_track', dtime]
     ['set_tempo', dtime, tempo]
     ['smpte_offset', dtime, hr, mn, se, fr, ff]
     ['time_signature', dtime, nn, dd, cc, bb]
     ['key_signature', dtime, sf, mi]
     ['sequencer_specific', dtime, raw]
     ['raw_meta_event', dtime, command(0-255), raw]
     ['sysex_f0', dtime, raw]
     ['sysex_f7', dtime, raw]
     ['song_position', dtime, song_pos]
     ['song_select', dtime, song_number]
     ['tune_request', dtime]

  DATA TYPES
     channel = a value 0 to 15
     controller = 0 to 127 (see http://www.pjb.com.au/muscript/gm.html#cc )
     dtime = time measured in "ticks", 0 to 268435455
     velocity = a value 0 (soft) to 127 (loud)
     note = a value 0 to 127  (middle-C is 60)
     patch = 0 to 127 (see http://www.pjb.com.au/muscript/gm.html )
     pitch_wheel = a value -8192 to 8191 (0x1FFF)
     raw = bytes, of length 0 or more  (for sysex events see below)
     sequence_number = a value 0 to 65,535 (0xFFFF)
     song_pos = a value 0 to 16,383 (0x3FFF)
     song_number = a value 0 to 127
     tempo = microseconds per crochet (quarter-note), 0 to 16777215
     text = bytes, of length 0 or more
     ticks = the number of ticks per crochet (quarter-note)

   In sysex_f0 events, the raw data must not start with a \xF0 byte,
   since this gets added automatically;
   but it must end with an explicit \xF7 byte!
   In the very unlikely case that you ever need to split sysex data
   into one sysex_f0 followed by one or more sysex_f7s, then only the
   last of those sysex_f7 events must end with the explicit \xF7 byte
   (again, the raw data of individual sysex_f7 events must not start
   with any \xF7 byte, since this gets added automatically).

   Since version 6.4, text data is in bytes, not in a ISO-8859-1 string.


  GOING THROUGH A SCORE WITHIN A PYTHON PROGRAM
    channels = {2,3,5,8,13}
    itrack = 1   # skip 1st element which is ticks
    while itrack < len(score):
        for event in score[itrack]:
            if event[0] == 'note':   # for example,
                pass  # do something to all notes
            # or, to work on events in only particular channels...
            channel_index = MIDI.Event2channelindex.get(event[0], False)
            if channel_index and (event[channel_index] in channels):
                pass  # do something to channels 2,3,5,8 and 13
        itrack += 1

'''

import sys, struct, copy
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb')
Version = '6.7'
VersionDate = '20201120'
# 20201120 6.7 call to bytest() removed, and protect _unshift_ber_int
# 20160702 6.6 to_millisecs() now handles set_tempo across multiple Tracks
# 20150921 6.5 segment restores controllers as well as patch and tempo
# 20150914 6.4 text data is bytes or bytearray, not ISO-8859-1 strings
# 20150628 6.3 absent any set_tempo, default is 120bpm (see MIDI file spec 1.1)
# 20150101 6.2 all text events can be 8-bit; let user get the right encoding
# 20141231 6.1 fix _some_text_event; sequencer_specific data can be 8-bit
# 20141230 6.0 synth_specific data can be 8-bit
# 20120504 5.9 add the contents of mid_opus_tracks()
# 20120208 5.8 fix num_notes_by_channel() ; should be a dict
# 20120129 5.7 _encode handles empty tracks; score2stats num_notes_by_channel
# 20111111 5.6 fix patch 45 and 46 in Number2patch, should be Harp
# 20110129 5.5 add mix_opus_tracks() and event2alsaseq()
# 20110126 5.4 "previous message repeated N times" to save space on stderr
# 20110125 5.2 opus2score terminates unended notes at the end of the track
# 20110124 5.1 the warnings in midi2opus display track_num
# 21110122 5.0 if garbage, midi2opus returns the opus so far
# 21110119 4.9 non-ascii chars stripped out of the text_events
# 21110110 4.8 note_on with velocity=0 treated as a note-off
# 21110108 4.6 unknown F-series event correctly eats just one byte
# 21011010 4.2 segment() uses start_time, end_time named params
# 21011005 4.1 timeshift() must not pad the set_tempo command
# 21011003 4.0 pitch2note_event must be chapitch2note_event
# 21010918 3.9 set_sequence_number supported, FWIW
# 20100913 3.7 many small bugfixes; passes all tests
# 20100910 3.6 concatenate_scores enforce ticks=1000, just like merge_scores
# 20100908 3.5 minor bugs fixed in score2stats
# 20091104 3.4 tune_request now supported
# 20091104 3.3 fixed bug in decoding song_position and song_select
# 20091104 3.2 unsupported: set_sequence_number tune_request raw_data
# 20091101 3.1 document how to traverse a score within Python
# 20091021 3.0 fixed bug in score2stats detecting GM-mode = 0
# 20091020 2.9 score2stats reports GM-mode and bank msb,lsb events
# 20091019 2.8 in merge_scores, channel 9 must remain channel 9 (in GM)
# 20091018 2.7 handles empty tracks gracefully
# 20091015 2.6 grep() selects channels
# 20091010 2.5 merge_scores reassigns channels to avoid conflicts
# 20091010 2.4 fixed bug in to_millisecs which now only does opusses
# 20091010 2.3 score2stats returns channels & patch_changes, by_track & total
# 20091010 2.2 score2stats() returns also pitches and percussion dicts
# 20091010 2.1 bugs: >= not > in segment, to notice patch_change at time 0
# 20091010 2.0 bugs: spurious pop(0) ( in _decode sysex
# 20091008 1.9 bugs: ISO decoding in sysex; str( not int( in note-off warning
# 20091008 1.8 add concatenate_scores()
# 20091006 1.7 score2stats() measures nticks and ticks_per_quarter
# 20091004 1.6 first mix_scores() and merge_scores()
# 20090424 1.5 timeshift() bugfix: earliest only sees events after from_time
# 20090330 1.4 timeshift() has also a from_time argument
# 20090322 1.3 timeshift() has also a start_time argument
# 20090319 1.2 add segment() and timeshift()
# 20090301 1.1 add to_millisecs()

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

def midi2opus(midi=b''):
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

def midi2score(midi=b''):
    r'''
Translates MIDI into a "score", using midi2opus() then opus2score()
'''
    return opus2score(midi2opus(midi))

def midi2ms_score(midi=b''):
    r'''
Translates MIDI into a "score" with one beat per second and one
tick per millisecond, using midi2opus() then to_millisecs()
then opus2score()
'''
    return opus2score(to_millisecs(midi2opus(midi)))

#------------------------ Other Transformations ---------------------

def to_millisecs(old_opus=None, desired_time_in_ms=1):
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

def play_score(score=None):
    r'''Converts the "score" to midi, and feeds it into 'aplaymidi -'
'''
    if score == None:
        return
    import subprocess
    pipe = subprocess.Popen(['aplaymidi','-'], stdin=subprocess.PIPE)
    if score_type(score) == 'opus':
        pipe.stdin.write(opus2midi(score))
    else:
        pipe.stdin.write(score2midi(score))
    pipe.stdin.close()

def timeshift(score=None, shift=None, start_time=None, from_time=0, tracks={0,1,2,3,4,5,6,7,8,10,12,13,14,15}):
    r'''Returns a "score" shifted in time by "shift" ticks, or shifted
so that the first event starts at "start_time" ticks.

If "from_time" is specified, only those events in the score
that begin after it are shifted. If "start_time" is less than
"from_time" (or "shift" is negative), then the intermediate
notes are deleted, though patch-change events are preserved.

If "tracks" are specified, then only those tracks get shifted.
"tracks" can be a list, tuple or set; it gets converted to set
internally.

It is deprecated to specify both "shift" and "start_time".
If this does happen, timeshift() will print a warning to
stderr and ignore the "shift" argument.

If "shift" is negative and sufficiently large that it would
leave some event with a negative tick-value, then the score
is shifted so that the first event occurs at time 0. This
also occurs if "start_time" is negative, and is also the
default if neither "shift" nor "start_time" are specified.
'''
    #_warn('tracks='+str(tracks))
    if score == None or len(score) < 2:
        return [1000, [],]
    new_score = [score[0],]
    my_type = score_type(score)
    if my_type == '':
        return new_score
    if my_type == 'opus':
        _warn("timeshift: opus format is not supported\n")
        # _clean_up_scores()  6.2; doesn't exist! what was it supposed to do?
        return new_score
    if not (shift == None) and not (start_time == None):
        _warn("timeshift: shift and start_time specified: ignoring shift\n")
        shift = None
    if shift == None:
        if (start_time == None) or (start_time < 0):
            start_time = 0
        # shift = start_time - from_time

    i = 1   # ignore first element (ticks)
    tracks = set(tracks)  # defend against tuples and lists
    earliest = 1000000000
    if not (start_time == None) or shift < 0:  # first find the earliest event
        while i < len(score):
            if len(tracks) and not ((i-1) in tracks):
                i += 1
                continue
            for event in score[i]:
                 if event[1] < from_time:
                     continue  # just inspect the to_be_shifted events
                 if event[1] < earliest:
                     earliest = event[1]
            i += 1
    if earliest > 999999999:
        earliest = 0
    if shift == None:
        shift = start_time - earliest
    elif (earliest + shift) < 0:
        start_time = 0
        shift = 0 - earliest

    i = 1   # ignore first element (ticks)
    while i < len(score):
        if len(tracks) == 0 or not ((i-1) in tracks):  # 3.8
            new_score.append(score[i])
            i += 1
            continue
        new_track = []
        for event in score[i]:
            new_event = list(event)
            #if new_event[1] == 0 and shift > 0 and new_event[0] != 'note':
            #    pass
            #elif new_event[1] >= from_time:
            if new_event[1] >= from_time:
                # 4.1 must not rightshift set_tempo
                if new_event[0] != 'set_tempo' or shift<0:
                    new_event[1] += shift
            elif (shift < 0) and (new_event[1] >= (from_time+shift)):
                continue
            new_track.append(new_event)
        if len(new_track) > 0:
            new_score.append(new_track)
        i += 1
    _clean_up_warnings()
    return new_score

def segment(score=None, start_time=None, end_time=None, start=0, end=100000000,
 tracks={0,1,2,3,4,5,6,7,8,10,11,12,13,14,15}):
    r'''Returns a "score" which is a segment of the one supplied
as the argument, beginning at "start_time" ticks and ending
at "end_time" ticks (or at the end if "end_time" is not supplied).
If the set "tracks" is specified, only those tracks will
be returned.
'''
    if score == None or len(score) < 2:
        return [1000, [],]
    if start_time == None:  # as of 4.2 start_time is recommended
        start_time = start  # start is legacy usage
    if end_time == None:    # likewise
        end_time = end
    new_score = [score[0],]
    my_type = score_type(score)
    if my_type == '':
        return new_score
    if my_type == 'opus':
        # more difficult (disconnecting note_on's from their note_off's)...
        _warn("segment: opus format is not supported\n")
        _clean_up_warnings()
        return new_score
    i = 1   # ignore first element (ticks); we count in ticks anyway
    tracks = set(tracks)  # defend against tuples and lists
    while i < len(score):
        if len(tracks) and not ((i-1) in tracks):
            i += 1
            continue
        new_track = []
        channel2cc_num  = {}     # most recent controller change before start
        channel2cc_val  = {}
        channel2cc_time = {}
        channel2patch_num  = {}  # keep most recent patch change before start
        channel2patch_time = {}
        set_tempo_num  = 500000 # most recent tempo change before start 6.3
        set_tempo_time = 0
        earliest_note_time = end_time
        for event in score[i]:
            if event[0] == 'control_change':  # 6.5
                cc_time = channel2cc_time.get(event[2]) or 0
                if (event[1] <= start_time) and (event[1] >= cc_time):
                    channel2cc_num[event[2]]  = event[3]
                    channel2cc_val[event[2]]  = event[4]
                    channel2cc_time[event[2]] = event[1]
            elif event[0] == 'patch_change':
                patch_time = channel2patch_time.get(event[2]) or 0
                if (event[1]<=start_time) and (event[1] >= patch_time):  # 2.0
                    channel2patch_num[event[2]]  = event[3]
                    channel2patch_time[event[2]] = event[1]
            elif event[0] == 'set_tempo':
                if (event[1]<=start_time) and (event[1]>=set_tempo_time): #6.4
                    set_tempo_num  = event[2]
                    set_tempo_time = event[1]
            if (event[1] >= start_time) and (event[1] <= end_time):
                new_track.append(event)
                if (event[0] == 'note') and (event[1] < earliest_note_time):
                    earliest_note_time = event[1]
        if len(new_track) > 0:
            new_track.append(['set_tempo', start_time, set_tempo_num])
            for c in channel2patch_num:
                new_track.append(['patch_change',start_time,c,channel2patch_num[c]],)
            for c in channel2cc_num:   # 6.5
                new_track.append(['control_change',start_time,c,channel2cc_num[c],channel2cc_val[c]])
            new_score.append(new_track)
        i += 1
    _clean_up_warnings()
    return new_score

def score_type(opus_or_score=None):
    r'''Returns a string, either 'opus' or 'score' or ''
'''
    if opus_or_score == None or str(type(opus_or_score)).find('list')<0 or len(opus_or_score) < 2:
        return ''
    i = 1   # ignore first element
    while i < len(opus_or_score):
        for event in opus_or_score[i]:
            if event[0] == 'note':
                return 'score'
            elif event[0] == 'note_on':
                return 'opus'
        i += 1
    return ''

def concatenate_scores(scores):
    r'''Concatenates a list of scores into one score.
If the scores differ in their "ticks" parameter,
they will all get converted to millisecond-tick format.
'''
    # the deepcopys are needed if the input_score's are refs to the same obj
    # e.g. if invoked by midisox's repeat()
    input_scores = _consistentise_ticks(scores)  # 3.7
    output_score = copy.deepcopy(input_scores[0])
    for input_score in input_scores[1:]:
        output_stats = score2stats(output_score)
        delta_ticks = output_stats['nticks']
        itrack = 1
        while itrack < len(input_score):
            if itrack >= len(output_score): # new output track if doesn't exist
                output_score.append([])
            for event in input_score[itrack]:
                output_score[itrack].append(copy.deepcopy(event))
                output_score[itrack][-1][1] += delta_ticks
            itrack += 1
    return output_score

def merge_scores(scores):
    r'''Merges a list of scores into one score.  A merged score comprises
all of the tracks from all of the input scores; un-merging is possible
by selecting just some of the tracks.  If the scores differ in their
"ticks" parameter, they will all get converted to millisecond-tick
format.  merge_scores attempts to resolve channel-conflicts,
but there are of course only 15 available channels...
'''
    input_scores = _consistentise_ticks(scores)  # 3.6
    output_score = [1000]
    channels_so_far = set()
    all_channels = {0,1,2,3,4,5,6,7,8,10,11,12,13,14,15}
    global Event2channelindex
    for input_score in input_scores:
        new_channels = set(score2stats(input_score).get('channels_total', []))
        new_channels.discard(9)  # 2.8 cha9 must remain cha9 (in GM)
        for channel in channels_so_far & new_channels:
            # consistently choose lowest avaiable, to ease testing
            free_channels = list(all_channels - (channels_so_far|new_channels))
            if len(free_channels) > 0:
                free_channels.sort()
                free_channel = free_channels[0]
            else:
                free_channel = None
                break
            itrack = 1
            while itrack < len(input_score):
                for input_event in input_score[itrack]:
                    channel_index=Event2channelindex.get(input_event[0],False)
                    if channel_index and input_event[channel_index]==channel:
                        input_event[channel_index] = free_channel
                itrack += 1
            channels_so_far.add(free_channel)

        channels_so_far |= new_channels
        output_score.extend(input_score[1:])
    return output_score

def _ticks(event):
    return event[1]
def mix_opus_tracks(input_tracks):   # 5.5
    r'''Mixes an array of tracks into one track.  A mixed track
cannot be un-mixed.  It is assumed that the tracks share the same
ticks parameter and the same tempo.
Mixing score-tracks is trivial (just insert all events into one array).
Mixing opus-tracks is only slightly harder, but it's common enough
that a dedicated function is useful.
'''
    output_score = [1000, []]
    for input_track in input_tracks:   # 5.8
        input_score = opus2score([1000, input_track])
        for event in input_score[1]:
            output_score[1].append(event)
    output_score[1].sort(key=_ticks) 
    output_opus = score2opus(output_score)
    return output_opus[1]

def mix_scores(scores):
    r'''Mixes a list of scores into one one-track score.
A mixed score cannot be un-mixed.  Hopefully the scores
have no undesirable channel-conflicts between them.
If the scores differ in their "ticks" parameter,
they will all get converted to millisecond-tick format.
'''
    input_scores = _consistentise_ticks(scores)  # 3.6
    output_score = [1000, []]
    for input_score in input_scores:
        for input_track in input_score[1:]:
            output_score[1].extend(input_track)
    return output_score

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
#	Tegridy MIDI Module (TMIDI / tee-midi)
#	Version 1.4
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

from operator import itemgetter

import sys

from abc import ABC, abstractmethod

# from collections import defaultdict

###################################################################################

def Tegridy_MIDI_Processor(MIDI_file, 
                          MIDI_channel=0, 
                          time_denominator=1, 
                          transpose_all_notes_by_this_many_pitches = 0,
                          flip_notes=0,
                          randomize_notes=0,
                          randremove_notes=0,
                          MIDI_patch=[0, 24, 32, 40, 42, 46, 56, 71, 73],
                          voble=0):

    '''Tegridy MIDI Processor

    Input: A single MIDI file.
           Desired MIDI channel to process. Def. = 0. All but drums = -1 and all channels = 16
           Notes/Chords timings divider (denominator).

    Output: A list of MIDI chords and a list of melody notes.
    MIDI Chords: Sorted by pitch (chord[0] == highest pitch).
    Melody Notes: Sorted by start time.

    Format: MIDI.py Score Events format.

    Default precision: 1 ms per note/chord.

    Enjoy! :)

    Project Los Angeles
    Tegridy Code 2020'''

###########

    minimum_number_of_notes_per_chord = 2 # Must be 2 or more...

    debug = False

###########

    average_note_pitch = 0
    min_note = 127
    max_note = 0

    files_count = 0

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
    if debug: print('Processing File:', file_address)
    
    try:
      opus = midi2opus(midi_file.read())
    
    except:
      print('Bad file. Skipping...')
      print('File name:', MIDI_file)
      midi_file.close()
      return chords_list, melody
         
    midi_file.close()

    score1 = to_millisecs(opus)
    score2 = opus2score(score1)
    
    if MIDI_channel == 16: # Process all MIDI channels
      score = score2
    
    if MIDI_channel >= 0 and MIDI_channel <= 15: # Process only a selected single MIDI channel
      score = grep(score2, [MIDI_channel])
    
    if MIDI_channel == -1: # Process all channels except drums (except channel 9)
      score = grep(score2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15])   
    
    #print('Reading all MIDI events from the MIDI file...')
    while itrack < len(score):
      chords_list_track = []
      for event in score[itrack]:
        chords_list = []
        if event[0] == 'patch_change':
          patch = event[3] 
        if event[0] == 'note' and patch in MIDI_patch:
          if len(event) == 6: # Checking for bad notes...
              rec_event = event
              rec_event[1] = int((event[1] + voble) / time_denominator)
              rec_event[2] = int((event[2] + voble) / time_denominator)
              rec_event[5] = int(event[5] + voble)

              if transpose_all_notes_by_this_many_pitches != 0:
                rec_event[4] = abs(int(event[4] + transpose_all_notes_by_this_many_pitches))

              if flip_notes !=0:
                if flip_notes == 1:
                  rec_event[4] = abs(int(127 - event[4]))
                
                if flip_notes == -1:
                  rec_event[4] = abs(int(0 - event[4]))

              if randomize_notes !=0:
                rec_event[4] = abs(int(event[4] + randomize_notes))

              if randremove_notes !=0:
                rec_event[4] = abs(int(event[4] * randremove_notes * secrets.choice([0-randremove_notes, 0, randremove_notes])))

              events_matrix.append(rec_event)

              min_note = int(min(min_note, rec_event[4]))
              max_note = int(max(max_note, rec_event[4]))          
              
              ev += 1
      
      itrack +=1 # Going to next track...
  
    #print('Doing some heavy pythonic sorting...Please stand by...')

    #print('Removing zero pitches and zero velocities events')
    events_matrix1 = [i for i in events_matrix if i[4] > 0 and i[5] > 0] # removing zero pitches and zero velocities events
    events_matrix = events_matrix1
    events_matrix1 = []

    for event in events_matrix:
      seen = set()
      event1 = [x for x in event if x not in seen and not seen.add(x)]
      events_matrix1.append(event1)

    events_matrix = []
    events_matrix = events_matrix1 

    #print('Sorting input by start time...')
    events_matrix.sort(key=lambda x: x[1]) # Sorting input by start time

    #print('Grouping by start time. This will take a while...')
    values = set(map(lambda x:x[1], events_matrix)) # Non-multithreaded function version just in case

    groups = [[y for y in events_matrix if y[1]==x and len(y) == 6] for x in values] # Grouping notes into chords while discarting bad notes...
    
    chords_list1 = []
    #print('Removing single note/excessive events, sorting events by pitch, and creating a chords list...')
    for items in groups: 
      if len(items) >= minimum_number_of_notes_per_chord: # Removing single note events
        items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch
        chords_list1.append(items) # Creating final chords list

    #print('Removing duplicate pitches from chords and creating a final chords list...')
    chords_list = []
    chord = []
    chord1 = []
    chord2 = []

    for chord in chords_list1:
      seen = set()
      chord1 = [x for x in chord if x[4] not in seen and not seen.add(x[4])]
      chord2 = [x for x in chord1 if len(x) == 6] # Removing bad note events from chords
      chords_list.append(chord2)

    chords_list_track = [i for i in chords_list if i != []]

    chords_list = []
    chords_list.extend(chords_list_track)

    #print('Extracting melody...')
    melody_list = []
    
    #print('Sorting events...')
    for items in groups:
        items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch 
        melody_list.append(items) # Creating final chords list
    
    #print('Removing duplicates if any...')
    for item in melody_list:
        seen = set()
        mel = [x for x in item if x[1] not in seen and not seen.add(x[1])]
        melody1.extend(mel)
    
    #print('Removing bad notes if any...')    
    for item in melody1:
        if len(item) == 6:
          melody.append(item)
    
    #print('Final sorting by start time...')      
    melody.sort(reverse=False, key=lambda x: x[1]) # Sorting events by start time
      
    return chords_list, melody

###################################################################################

def Tegridy_Chords_Converter(chords_list, melody_list, song_name, melody_notes_in_chords=True):
  '''Tegridy Chords Coverter

  Inputs: Tegridy MIDI chords_list (as is)

          Tegridy MIDI melody_list (as is)

          Name of the song as plain string

          Include or exclude melody notes in each chord. Def. is to include.


  Outputs: Converted chords_list with melody_notes and song name

           Converted melody_list with song name

  Project Los Angeles
  Tegridy Code 2020'''

  temp_chords_list = []
  chords_list_final = []
  melody_list_final = []

  temp_chords_list = [[song_name, 0, 0, 0, 0, 0]]
  melody_list_final = [song_name, 0, 0, 0, 0, 0]

  debug = False

  for notez in melody_list:
    if melody_notes_in_chords:
      temp_chords_list.append([notez])
    melody_list_final.append(notez)
    for chord in chords_list:
      if notez[1] == chord[0][1]:
        temp_chords_list.append(chord[1:])
  
  '''# Gonna use a dic here to join chords by start-time :)
  record_dict = defaultdict(list)

  for chords in temp_chords_list:
    if len(chords) > 0:
      record_dict[chords[0][1]].extend(chords)

  temp_chords_list = list(record_dict.values())'''

  chords_list_final = []
  #print('Sorting chords notes by pitch/removing empty chords if any...')
  chords_list_final.append(temp_chords_list[0])
  for chordz in temp_chords_list[1:]:
    if len(chordz) > 0:
      if debug: print(chordz)
      chordz.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch 
      chords_list_final.append(chordz) # Creating final chords list      

  chords_list_final[0] = [[song_name + '_with_' + str(len(chords_list_final)-1) + '_Chords', 0, 0, len(chords_list_final)-1, 0, 0]]
  melody_list_final[0] = [song_name + '_with_' + str(len(melody_list_final)-1) + '_Notes', 0, 0, len(melody_list_final)-1, 0, 0]
  chords_list_final.append([['song_end', chords_list_final[:-1][1], 0, len(chords_list_final)-1, 0, 1]])
  melody_list_final.append(['song_end', melody_list_final[:-1][1], 0, len(melody_list_final)-1, 0, 1])     
  first_song = False
    
  return chords_list_final, melody_list_final

###################################################################################

def Tegridy_MIDI_TXT_Processor(dataset_name,
                              converted_chords_list,
                              converted_melody_list,
                              simulate_velocity=False,
                              line_by_line_output=False,
                              represent_every_number_of_chords = 0,
                              chords_duration_multiplier = 1,
                              pad_chords_with_stops=False,
                              chords_beat_divider = 100):

    '''Tegridy MIDI to TXT Processor
     
     Input: Dataset name
            Tegridy MIDI chords_list and melody_list (as is)
            Simulate velocity or not
            Line-by-line switch (useful for the AI models tokenizers and other specific purposes)
            Represent events every so many steps. Def. is 0. == do not represent.
            Chords durations multiplier. Def. = 1
            Pad chords with timed rests or not. Helps with some NLP implementations
            Chords beat divider/denominator. This essentially creates a beat for AI models to keep in mind. Default is 100 = 10 beats per second.

     Output: TXT encoded MIDI events as plain txt/plain str
             Number of processed chords
             Number of bad/skipped chords (for whatever reason)

     Project Los Angeles
     Tegridy Code 2020'''

    debug = False

    song_chords_count = 0
    number_of_chords_recorded = 0
    number_of_bad_chords_recorded = 0
    chord_start_time = 0
    first_song = True

    rpz = 1

    previous_start_time = 0

    beat = 0

    if dataset_name != '':
      TXT_string = str(dataset_name)
    else:
      TXT_string = ''
    
    if line_by_line_output:
      TXT_string += '\n'
    else:
      TXT_string += ' '  

    for chord in tqdm.auto.tqdm(converted_chords_list):
      try:
        if chord[0][3] > 15:
          song_dur = int(chord[0][3])
        
        if len(chord) > 1:
          durs_chord = int(max(list(zip(*chord))[2]) * chords_duration_multiplier)
          chord_duration = durs_chord
        
        else:      
          chord_duration = int(chord[0][2] * chords_duration_multiplier)
          
        if simulate_velocity:
          chord_velocity = chord[0][4]
        else:  
          chord_velocity = chord[0][5]
        chord_start_time = chord[0][1] 
        if chord_duration == 0 and chord_velocity == 0:
          if not str(chord[0][0]) == 'song_end':
            if not first_song:
              TXT_string += 'SONG=END_' + str(song_chords_count) + '_Chords'
              if line_by_line_output:
                TXT_string += '\n'
              else:  
                TXT_string += ' '

              TXT_string += 'SONG=' + str(chord[0][0])
              if line_by_line_output:
                TXT_string += '\n'
              else:  
                TXT_string += ' '
              song_chords_count = 1 
                          
            else:  

              TXT_string += 'SONG=' + str(chord[0][0])
              if line_by_line_output:
                TXT_string += '\n'
              else:  
                TXT_string += ' '
              song_chords_count = 1  
            
          else:
              TXT_string += 'SONG=END_' + str(song_chords_count-1) + '_Chords'
              if line_by_line_output:
                TXT_string += '\n'
              else:  
                TXT_string += ' '             

        else:
          
          beat = int((abs(int(chord_start_time - previous_start_time))) / chords_beat_divider)

          if pad_chords_with_stops:
            if (chord_start_time - previous_start_time - 1) > 0:
              TXT_string += str(abs(int(chord_start_time - previous_start_time) - 1)) + '-' + str(0) + '-' + str(0) + '-' +str(0) + '-' + str(beat) + '-' + str(str(0) + '/' + str(0))
              if line_by_line_output:
                TXT_string += '\n'
              else:  
                TXT_string += ' '

          TXT_string += str(abs(int(chord_start_time - previous_start_time))) + '-' + str(chord_duration) + '-' + str(chord[0][3]) + '-' + str(chord_velocity) + '-' + str(beat)
          
          
          previous_start_time = chord_start_time
          
          for note in chord:
            TXT_string += '-' + str(note[4]) + '/' + str(chord_duration - int(note[2] * chords_duration_multiplier))
          
          # Representation of events
          if represent_every_number_of_chords > 0:
            if rpz == represent_every_number_of_chords:
              TXT_string += '#' + str(song_dur)
              rpz = 0    
          
          if line_by_line_output:
            TXT_string += '\n'
          else:  
            TXT_string += ' '
          
          if debug: print(chord)

          song_chords_count += 1
          number_of_chords_recorded += 1
          rpz += 1
          
      except:
        if debug: print('Bad chord. Skipping...')
        number_of_bad_chords_recorded += 1
        continue            

    return TXT_string, number_of_chords_recorded, number_of_bad_chords_recorded

###################################################################################

def Tegridy_TXT_MIDI_Processor(input_string, 
                              line_by_line_dataset = False,
                              dataset_MIDI_events_time_denominator = 10,
                              number_of_ticks_per_quarter = 425,
                              start_from_this_generated_event = 0,
                              remove_generated_silence_if_needed = False,
                              silence_offset_from_start = 75000,
                              simulate_velocity = False,
                              output_signature = 'TMIDI-TXT-MIDI',
                              list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0]):

    '''Tegridy TXT to MIDI Processor
     
     Input: Input TXT string in the TMIDI-TXT format
            Input is line-by-line or one-line
            Used dataset time denominator
            Number of ticks per quater for output MIDI
            Start from this input event (skip this many from start)
            Is there a generated silence or not
            Silence offset in MIDI ticks from start
            Simulate velocity (V = max(Pitch))
            Output MIDI signature
            List of 16 desired MIDI patch numbers for the output MIDI. Def. is MuseNet compatible patch list.

     Output: NOTE: For now only 1st recorded TXT performance converted to MIDI.
             Raw/binary MIDI data that can be recorded to a file with standard python functions.
             Detected number of input notes
             Recorded number of output notes
             Detailed created MIDI stats in the MIDI.py module format (MIDI.score2stats)

     Project Los Angeles
     Tegridy Code 2020'''

    debug = False

    if line_by_line_dataset:
      input_string = input_string.split() # for new datasets
    else:
      input_string = input_string.split(' ') # for compatibility/legacy datasets
    if debug: print(input_string)


    i=0
    z=1

    notes_specs = []
    song_name = ''
    previous_chord_start_time = 0
    number_of_notes_recorded = 0
    zero_marker = True
    song_header = []
    song_score = []

    start_time = 0
    duration = 0

    print('Converting TXT to MIDI. Please wait...')
    for i in range(len(input_string)):
      if input_string[i].split('=END_')[0] == 'SONG':
        break
        #TODO Record all perfomances (even incomplete ones) as separate tracks in output MIDI

      if input_string[i].split('=')[0] == 'SONG':
        try:
          song_name = input_string[i].split('=')[1]
          song_header.append(['track_name', 0, song_name])
          duration = 1
          continue
        except:
          print('Unknown Song name format', song_name)
          duration = 1
          continue
        
      if duration != 0: #Skipping rests

        try:
          start_time += int(input_string[i].split('-')[0]) * dataset_MIDI_events_time_denominator
          duration = int(input_string[i].split('-')[1]) * dataset_MIDI_events_time_denominator 
          channel = int(input_string[i].split('-')[2])
          velocity = int(input_string[i].split('-')[3])
        except:
          print('Unknown Chord:', input_string[i])
        
        try:
          for x in range(len(str(input_string[i]).split('-')[5:])):
            notes_specs, dur = str(input_string[i].split('-')[5:][x]).split('/')
            duration = duration - int(dur)
            simulated_velocity = int(notes_specs)
            if simulate_velocity:
              song_score.append(['note', 
                                  int(start_time), #Start Time
                                  int(duration), #Duration
                                  int(channel), #Channel
                                  int(notes_specs), #Note
                                  int(simulated_velocity)]) #Velocity              
              number_of_notes_recorded += 1
            else:  
              song_score.append(['note', 
                                  int(start_time), #Start Time
                                  int(duration), #Duration
                                  int(channel), #Channel
                                  int(notes_specs), #Note
                                  int(velocity)]) #Velocity              
              number_of_notes_recorded += 1
        except:
          print("Unknown Notes: " + input_string[i])
          continue

    if remove_generated_silence_if_needed:
      song_score1 = []
      for note in song_score[start_from_this_generated_event:]:
        note1 = note
        note1[1] = note[1] - silence_offset_from_start
        song_score1.append(note1)
      song_score = song_score1  

    output_header = [number_of_ticks_per_quarter, [['track_name', 0, bytes(output_signature, 'utf-8')]]] 
                                                  
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
                ['patch_change', 0, 15, list_of_MIDI_patches[15]]]

    output_song = song_header + song_score[start_from_this_generated_event:]
    output = output_header + [patch_list + output_song]

    midi_data = score2midi(output)
    detailed_MIDI_stats = score2stats(output)

    return midi_data, len(input_string), number_of_notes_recorded, detailed_MIDI_stats

###################################################################################

def Tegridy_TXT_to_INT_Processor(input_TXT_string):

    '''Tegridy TXT to Intergers Processor
     
    Input: Input TXT string in the TMIDI-TXT format

    Output: List of intergers
            Decoding dictionary


    Project Los Angeles
    Tegridy Code 2020'''

    # get vocabulary set
    words = sorted(tuple(set(input_TXT_string.split('\n'))))
    n = len(words)

    # create word-integer encoder/decoder
    word2int = dict(zip(words, list(range(n))))
    decoding_dictionary = dict(zip(list(range(n)), words))

    # encode all words in the string into integers
    output_INT_list = [word2int[word] for word in input_TXT_string.split('\n')]
    
    return output_INT_list, decoding_dictionary

###################################################################################

def Tegridy_INT_to_TXT_Processor(input_INT_list, decoding_dictionary):

    '''Tegridy Intergers to TXT Processor
     
    Input: List of intergers in TMIDI-TXT-INT format
          Decoding dictionary in TMIDI-TXT-INT format

    Output: Decoded TXT string in TMIDI-TXT format

    Project Los Angeles
    Tegridy Code 2020'''

    output_TXT_string = '\n'.join(decoding_dictionary[int_] for int_ in input_INT_list)
    
    return output_TXT_string

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

def Tegridy_TXT_Reducer(input_string, 
                        line_by_line_input_dataset = True,
                        line_by_line_output_dataset = True,
                        include_MIDI_channels=True,
                        include_notes_velocities=True,
                        char_encoding_offset = 30,
                        include_beat = False):

    '''Tegridy TXT Reducer
     
    Input: Input TXT string in the TMIDI-TXT format
           Input dataset type
           Output dataset type
           Dataset MIDI events time divider/denominator
           Reduce MIDI channels or not (False = savings on AI token memory)
           Reduce Note's velocities or not (False = savings on AI token memory)
           Char encoding offset. This is to prevent ambiguity with sys chars like \n.

    Output: Reduced TXT string in UTF-8 format
            Number of recorded notes

    Project Los Angeles
    Tegridy Code 2020'''

    debug = False

    if line_by_line_input_dataset:
      input_string = input_string.split() # for general use
    else:
      input_string = input_string.split(' ') # for some specific purposes
    if debug: print(input_string)


    i=0
    z=1

    notes_specs = []
    song_name = ''
    previous_chord_start_time = 0
    number_of_notes_recorded = 0
    zero_marker = True
    song_header = []
    song_score = []

    start_time = 0

    Output_TXT_string = ''

    print('Reducing TXT. Please wait...')
    for i in tqdm.auto.tqdm(range(len(input_string))):
      
      if input_string[i].split('=')[0] == 'DATASET':
        Output_TXT_string += input_string[i] + '\n'
        continue

      if input_string[i].split('=')[0] == 'SONG':
        if input_string[i].split('=')[1][0:4] != 'END_':
          if line_by_line_output_dataset:
            Output_TXT_string += input_string[i] + '\n'
            continue
          else:
            Output_TXT_string += input_string[i] + ' '  
            continue
        else:
          Output_TXT_string += input_string[i] + '\n'
          continue
        
      try:
        start_time = int(input_string[i].split('-')[0])
        duration = int(input_string[i].split('-')[1])
        channel = int(input_string[i].split('-')[2])
        velocity = int(input_string[i].split('-')[3])
        beat = int(input_string[i].split('-')[4])
      except:
        print('Unknown Chord:', input_string[i])
      
      try:
        chars = ''
        for x in range(len(str(input_string[i]).split('-')[5:])):
          notes_specs, dur = str(input_string[i].split('-')[5:][x]).split('/')
          dura = duration - int(dur)
          
          chars += chr(int(start_time)+char_encoding_offset) #Start Time
          chars += chr(int(dura)+char_encoding_offset) #Duration
          
          if include_beat == True and include_MIDI_channels == False:
            chars += chr(int(beat)+char_encoding_offset) #Beat

          if include_MIDI_channels == True and include_beat == False:
            chars += chr(int(channel)+char_encoding_offset) #Channel

          chars += chr(int(notes_specs)+char_encoding_offset) #Note

          if include_notes_velocities:
            chars += chr(int(velocity)+char_encoding_offset) #Velocity
 
          number_of_notes_recorded += 1
      except:
        print("Unknown Notes: " + input_string[i])
        continue
      if len(chars) > 0:
        Output_TXT_string += chars
        if line_by_line_output_dataset:
          Output_TXT_string += '\n'
        else:
          Output_TXT_string += ' '  

    print('Task complete! Enjoy :)')
       
    return Output_TXT_string, number_of_notes_recorded

###################################################################################

def Tegridy_Reduced_TXT_to_Notes_Converter(Reduced_TXT_String,
                                          line_by_line_dataset = True,
                                          has_MIDI_channels = True,
                                          has_velocities = True,
                                          dataset_MIDI_events_time_denominator = 10,
                                          char_encoding_offset = 30,
                                          save_only_first_composition = True,
                                          dataset_includes_beat = False):
                                                            
    '''Tegridy Reduced TXT to Notes Converter
     
    Input: Input TXT string in the Reduced TMIDI-TXT format
           Input dataset type
           Dataset was encoded with MIDI channels info or not
           Dataset was encoded with note's velocities info or not
           Used dataset time denominator/divider. It must match or the timings will be off.
           Char encoding offset. This is to prevent ambiguity with sys chars like \n.

    Output: List of notes in MIDI.py Score format (TMIDI SONG format)
            First SONG= occurence (song name usually)

    Project Los Angeles
    Tegridy Code 2020'''

    print('Tegridy Reduced TXT to Notes Converter')
    print('Converting Reduced TXT to Notes list...Please wait...')

    song_name = ''

    if line_by_line_dataset:
      input_string = Reduced_TXT_String.split('\n')
    else:
      input_string = Reduced_TXT_String.split(' ')
    
    if line_by_line_dataset:
      name_string = Reduced_TXT_String.split('\n')[0].split('=')
    else:
      name_string = Reduced_TXT_String.split(' ')[0].split('=')

    if name_string[0] == 'SONG':
      song_name = name_string[1]

    output_list = []
    st = 0

    for i in range(2, len(input_string)-1):
      
      if save_only_first_composition:
        if input_string[i].split('=')[0] == 'SONG':
          if input_string[i].split('=')[1][0:4] != 'END_' :
            song_name = name_string[1]
            continue
          else:
            break

      try:
        istring = input_string[i]

        if has_MIDI_channels==False and has_velocities==False:
          step = 3

        if has_MIDI_channels==True and has_velocities==False:
          step = 4          
        
        if has_MIDI_channels==False and has_velocities==True:
          step = 4  
        
        if has_MIDI_channels==True and has_velocities==True:
          step = 5

        if dataset_includes_beat:
          step = step + 1

        dur = (ord(istring[1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator


        if dur != 0 and int(ord(istring[3]) - char_encoding_offset) != 0: #Check for rests/zero notes (we are skipping them)

          st += int(ord(istring[0]) - char_encoding_offset) * dataset_MIDI_events_time_denominator

          for s in range(0, len(istring), step):
              if has_MIDI_channels==True and has_velocities==True:
                if step >= 4 and len(istring) > 3:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time
                      out.append((ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(ord(istring[s+2]) - char_encoding_offset) # Channel
                      out.append(ord(istring[s+3]) - char_encoding_offset) # Pitch
                      out.append(ord(istring[s+4]) - char_encoding_offset) # Velocity

                      output_list.append(out)
              
              if has_MIDI_channels==True and has_velocities==False:
                if step >= 3 and len(istring) > 2:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time
                      out.append((ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(ord(istring[s+2]) - char_encoding_offset) # Channel
                      out.append(ord(istring[s+3]) - char_encoding_offset) # Pitch
                      if s == 0:
                        sim_vel = ord(istring[s+3]) - char_encoding_offset
                      out.append(sim_vel) # Simulated Velocity (= highest note's pitch)
                      output_list.append(out)

              if has_velocities==True and has_MIDI_channels==False:
                if step >= 3 and len(istring) > 2:
                      out = []       
                      out.append('note')
                      
                      out.append(st) # Start time
                      out.append((ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(int(0)) # Simulated Channel (Defaulting to Channel 0)
                      out.append(ord(istring[s+3]) - char_encoding_offset) # Pitch
                      out.append(ord(istring[s+4]) - char_encoding_offset) # Velocity

                      output_list.append(out)

              if has_MIDI_channels==False and has_velocities==False:
                if step >= 2 and len(istring) > 1:
                      out = []       
                      out.append('note')

                      out.append(st) # Start time
                      out.append((ord(istring[s+1]) - char_encoding_offset) * dataset_MIDI_events_time_denominator) # Duration
                      out.append(int(0)) # Simulated Channel (Defaulting to Channel 0)
                      out.append(ord(istring[s+3]) - char_encoding_offset) # Pitch
                      if s == 0:
                        sim_vel = ord(istring[s+3]) - char_encoding_offset
                      out.append(sim_vel) # Simulated Velocity (= highest note's pitch)
                      output_list.append(out)
      except:
        print('Bad note string:', istring)
        continue

    print('Task complete! Enjoy! :)')

    return output_list, song_name

###################################################################################

def Tegridy_SONG_to_MIDI_Converter(SONG,
                                  output_signature = 'Tegridy TMIDI Module', 
                                  track_name = 'Composition Track',
                                  number_of_ticks_per_quarter = 425,
                                  list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                  output_file_name = 'TMIDI-Composition',
                                  text_encoding='ISO-8859-1'):

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
    
    print('Done! Enjoy! :)')
    
    return detailed_MIDI_stats

###################################################################################

def Tegridy_Karaoke_MIDI_to_Reduced_TXT_Processor(Karaoke_MIDI_file, 
                                                  karaoke_language_encoding = 'ISO-8859-1',
                                                  char_encoding_offset = 30):

    '''Tegridy Karaoke MIDI to Reduced TXT Processor
     
    Input: Karaoke MIDI file. Must be a Karaoke MIDI or the processor will not work properly.
           
           Karaoke language encoding. Please see official encoding list for your language. 
           https://docs.python.org/3/library/codecs.html#standard-encodings
           Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.
           
           Char encoding offset to prevent ambiguity with sys chars like \n. 
           This may need to be adjusted for languages other than English.

    Output: Line-by-line reduced TXT string
            Number of processed MIDI events from the Karaoke MIDI file
            Number of recorded Karaoke events in the TXT string
            All recorded Pitches/Words of the given KarMIDI file as a list
            All recorded words of the given KarMIDI file as a string

    Project Los Angeles
    Tegridy Code 2021'''

    events_list = []
    events_matrix = []

    MIDI_ev = 0
    KAR_ev = 0

    itrack = 1

    tst = 0

    output_string = ''

    midi_file = open(Karaoke_MIDI_file, 'rb')
    
    try:
      opus = midi2opus(midi_file.read())
    
    except:
      print('Problematic file. Skipping...')
      print('Skipped file name:', Karaoke_MIDI_file)
      midi_file.close()
      return output_string, MIDI_ev, KAR_ev
          
    midi_file.close()

    score1 = to_millisecs(opus)
    score = opus2score(score1)

    #print('Reading all MIDI events from the MIDI file...')
    while itrack < len(score):
      for event in score[itrack]:
        if event[0] == 'text_event':
          tst = 0
          txt = ''
          tst = event[1]
          txt = event[2]
          
        if event[0] == 'note' and event[1] == tst:
          evt = copy.deepcopy(event)
          evt.extend([''])
          evt[6] = txt
          events_list.append(evt)       
        
        MIDI_ev += 1
      
      itrack +=1 # Going to next track...

    evt = sorted(events_list, key=itemgetter(1)) # Sorting by start time
    groups = [list(g) for _,g in groupby(evt,itemgetter(1))] # Grouping by start time

    events_matrix.extend(groups)

    f_matrix = []
    final_matrix = []
    for items in events_matrix: 
      if len(items) > 0: # Removing single note events
        it = []
        
        it.extend(items)
        it.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch
        f_matrix.append(it[0]) 

    ptime = 0
    time = 0
    delta = 0
    output_song = []

    for n in range(len(f_matrix)-1):
      no = copy.deepcopy(f_matrix[n])

      no[1] = int(delta / 10)
      no[2] = int(no[2] / 10)
      no[5] = no[4]

      ptime = f_matrix[n][1]
      time = f_matrix[n+1][1]

      delta = abs(time-ptime)    

      output_song.append(no)

    output_string = ''
    all_words = ''
    pitches_words_list = []

    for note in output_song:
      if note[1] < 250 and note[2] < 250:
        if note[1] >= 0 and len(note[6]) > 0: 
          output_string += chr(note[1] + char_encoding_offset)
          output_string += chr(note[2] + char_encoding_offset)
          output_string += chr(note[4] + char_encoding_offset)
          output_string += '='
          word = str(note[6].decode(karaoke_language_encoding, 'replace')).replace('/', '').replace(' ', '')
          output_string += word
          output_string += '\n'
          all_words += word + ' '
          pitches_words_list.append([note[4], word])
          KAR_ev += 1

    return output_string, MIDI_ev, KAR_ev, pitches_words_list, all_words  

###################################################################################

def Tegridy_Karaoke_TXT_to_MIDI_Processor(Karaoke_TXT_String,
                                          text_encoding='ISO-8859-1',
                                          char_encoding_offset = 30):

    '''Tegridy Karaoke TXT to MIDI Processor
        
    Input: Karaoke Reduced TXT String in TMIDI Karaoke Reduced TXT format
            
           Karaoke language encoding. Please see official encoding list for your language. 
           https://docs.python.org/3/library/codecs.html#standard-encodings
           Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.
            
           Char encoding offset to prevent ambiguity with sys chars like \n. 
           This may need to be adjusted for languages other than English.

    Output: Inferred song name (from the first line of the input TXT string)
            Song (notes list in MIDI.py score format) that you can write to MIDI file with TMIDI Song to MIDI converter.
            All song's lyrics as one TXT string (this is for eval/display purposes mostly)
            Number of recorded Karaoke events in the output song

    Project Los Angeles
    Tegridy Code 2021'''    
    
    print('Tegridy Karaoke TXT to MIDI Processor')
    print('Converting Karaoke TXT to MIDI. Please wait...')

    o_str = Karaoke_TXT_String.split('\n')

    song_name = o_str[0]

    song = []

    lyrics_text = ''

    ptime = 0

    KAR_ev = 0

    for st in o_str:

      note = ['note', 0, 0, 0, 0, 0]
      text = ['text_event', 0, '']

      if len(st.split('=')[0]) == 3 and len(st) > 4:

        note[1] = ptime * 10
        note[2] = (ord(st.split('=')[0][1]) - char_encoding_offset) * 10
        note[4] = (ord(st.split('=')[0][2]) - char_encoding_offset)
        note[5] = (ord(st.split('=')[0][2]) - char_encoding_offset)

        text[1] = ptime * 10
        text[2] = str(st.split('=')[1])

        ptime += ord(st.split('=')[0][0]) - char_encoding_offset

        song.append(note)
        song.append(text)

        lyrics_text += str(st.split('=')[1]) + ' '

        KAR_ev += 1

    print('Task complete! Enjoy! :)')

    return song_name, song, lyrics_text, KAR_ev

###################################################################################
#
# Tegridy helper functions
#
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

def Tegridy_TXT_Dataset_File_Writer(input_file_name='TMIDI_TXT_Dataset', 
                                    ext = '', 
                                    TXT_String = ''):

  '''Tegridy TXT Dataset File Writer
     
  Input: Full path and file name without extention
         File extension
         Dataset as TXT string
          
  Output: Named TXT Dataset File

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy TXT Dataset File Writer')

  full_path_to_TXT_dataset = input_file_name + ext
  
  if os.path.exists(full_path_to_TXT_dataset):
    os.remove(full_path_to_TXT_dataset)
    print('Removing old Dataset...')
  else:
    print("Creating new Dataset file...")

  print('Writing dataset to a file...Please wait...')
  f = open(full_path_to_TXT_dataset, 'wb')
  f.write(TXT_String.encode('utf-8', 'replace'))
  f.close()
  print('Dataset was saved as:', full_path_to_TXT_dataset)
  print('Task complete! Enjoy :)')

###################################################################################

def Tegridy_Pickle_File_Writer(Data, input_file_name='TMIDI_Pickle_File'):

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

def Tegridy_Pickle_File_Loader(input_file_name='TMIDI_Pickle_File', ext='.pickle'):

  '''Tegridy Pickle File Loader
     
  Input: Full path and file name without extention
         File extension if different from default .pickle
       
  Output: Chords list in TMIDI MIDI Processor format
          Melody list in TMIDI MIDI Processor format

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Loader')
  print('Loading the pickle file. Please wait...')

  dataset = open(input_file_name + ext, "rb")

  chords_list_f, melody_list_f = pickle.load(dataset)

  dataset.close()

  print('Loading complete.')
  print('Number of MIDI chords recorded:', len(chords_list_f))
  print('The longest chord:', len(max(chords_list_f, key=len)), 'notes') 
  print(max(chords_list_f, key=len))
  print('Number of recorded melody events:', len(melody_list_f))
  print('First melody event:', melody_list_f[0], 'Last Melody event:', melody_list_f[-1])
  print('Total number of MIDI events recorded:', len(chords_list_f))
  print('Task complete. Enjoy! :)')

  return chords_list_f, melody_list_f

def Tegridy_Any_Pickle_File_Loader(input_file_name='TMIDI_Pickle_File', ext='.pickle'):

  '''Tegridy Pickle File Loader
     
  Input: Full path and file name without extention
         File extension if different from default .pickle
       
  Output: Standard Python 3 unpickled data object

  Project Los Angeles
  Tegridy Code 2021'''

  print('Tegridy Pickle File Loader')
  print('Loading the pickle file. Please wait...')

  with open(input_file_name + ext, 'rb') as pickle_file:
    content = pickle.load(pickle_file)

  return content

###################################################################################

def Tegridy_Karaoke_Pitches_Words_List_to_CSV_Writer(pitches_words_list, file_name='pitches_words.csv'):
    
    '''Tegridy Karaoke Pitches Words List to CSV Writer
      
    Input: Pitches/Words list in TMIDI Karaoke MIDI to TXT Converter format
           Desired full output CSV file name with extension
        
    Output: CSV file with all Pitches/Words that were in the input pitches/words list

    Project Los Angeles
    Tegridy Code 2021'''
    
    print('Tegridy Karaoke Pitches/Words CSV Writer')
    print('Starting up...')
    print('Grouping input pitches/words list...Please stand by...')
    values = set(map(lambda x:x[0], pitches_words_list))

    groups = [[y for y in pitches_words_list if y[0]==x] for x in values]
    
    print('Preparing final CSV list...')

    final_list = {}

    for g in groups:
      pitch = g[0][0]
      f_list = []
      for value in g:
        if value[1] not in f_list:
          f_list.append(value[1])
      final_list[pitch] = f_list

    print('Writing CSV file to disk...')
    with open(file_name,'w',newline='') as f:
      w = csv.writer(f)
      w.writerow(['pitch','words'])
      for key,items in final_list.items():
        w.writerow([key, ' '.join(sorted(items))])

    print('Task complete! Enjoy :)')

###################################################################################

# TMIDI 2.0 Code is below

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
                              perfect_timings=False):

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
    if debug: print('Processing File:', file_address)
    
    try:
      opus = midi2opus(midi_file.read())
    
    except:
      print('Bad file. Skipping...')
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
          karaokez.append(event)
        
        if event[0] == 'text_event' or event[0] == 'lyric':
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

    if not karaoke:
      previous_event = copy.deepcopy(events_matrix[0])
      for event in events_matrix:

        '''# Computing deltas
        start_time = int(event[1] - previous_event[1])
        duration = int(event[2] - previous_event[2])
        channel = int(event[3])
        pitch = int(event[4] - previous_event[4])
        velocity = int(event[5] - previous_event[5])'''

        # Computing events details
        start_time = int(event[1] - previous_event[1])
        
        duration = int(previous_event[2])

        channel = int(previous_event[3])

        pitch = int(previous_event[4] + transpose_by)
        if flip == True:
          pitch = 127 - int(previous_event[4] + transpose_by)

        velocity = int(previous_event[5])

        # Converting to TXT
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
      
      if not line_by_line_output:
        txt += chr(10)      
    
    chords.extend(events_matrix)
    #print(chords)

    #print('Extracting melody...')
    melody_list = []

    #print('Grouping by start time. This will take a while...')
    values = set(map(lambda x:x[1], events_matrix)) # Non-multithreaded function version just in case

    groups = [[y for y in events_matrix if y[1]==x and len(y) == 6] for x in values] # Grouping notes into chords while discarting bad notes...
  
    #print('Sorting events...')
    for items in groups:
        items.sort(reverse=True, key=lambda x: x[4]) # Sorting events by pitch
        items[0][3] = 0 # Melody should always bear MIDI Channel 0 for code to work
        melody_list.append(items[0]) # Creating final melody list
        melody_chords.append(items) # Creating final chords list
        bass_melody.append(items[-1]) # Creating final bass melody list
    
    #print('Final sorting by start time...')      
    melody_list.sort(reverse=False, key=lambda x: x[1]) # Sorting events by start time
    melody_chords.sort(reverse=False, key=lambda x: x[0][1]) # Sorting events by start time
    bass_melody.sort(reverse=False, key=lambda x: x[1]) # Sorting events by start time
    #print(txt, melody, chords)

    # [WIP] Melody-conditioned chords list
    if melody_conditioned_encoding == True:
      if not karaoke:
   
        previous_event = copy.deepcopy(melody_chords[0][0])

        for ev in melody_chords:
          hp = True
          ev.sort(reverse=False, key=lambda x: x[4]) # Sorting chord events by pitch
          for event in ev:
          
            # Computing events details
            start_time = int(event[1] - previous_event[1])
            
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

            # Converting to TXT
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
            
        if not line_by_line_output:
          txtc += chr(10)

        txt = txtc
        chords = melody_chords

    if karaoke:
      previous_event = copy.deepcopy(melody_list[0])
      for event in melody_list:

        # Computing events details
        start_time = int(event[1] - previous_event[1])
        
        duration = int(previous_event[2])

        channel = int(previous_event[3])

        pitch = int(previous_event[4] + transpose_by)

        velocity = int(previous_event[5])

        # Converting to TXT
        txt += str(chr(start_time + char_offset))
        txt += str(chr(duration + char_offset))
        txt += str(chr(pitch + char_offset))

        txt += str(chr(velocity + char_offset))      

        txt += '='
        for k in karaoke_events_matrix:
          if event[1] == k[1]:
            txt += str(k[2])          
            break

        if line_by_line_output:
          txt += chr(10)
        else:
          txt += chr(32) 
        
        previous_event = copy.deepcopy(event)
      
      if not line_by_line_output:
        txt += chr(10)

    # Helper aux/backup function for Karaoke
    karaokez.sort(reverse=False, key=lambda x: x[1])  

    return txt, melody_list, chords #, bass_melody # Bass melody aux output

###################################################################################

def Tegridy_Optimus_TXT_to_Notes_Converter(Optimus_TXT_String,
                                          line_by_line_dataset = True,
                                          has_velocities = True,
                                          has_MIDI_channels = True,
                                          dataset_MIDI_events_time_denominator = 1,
                                          char_encoding_offset = 30000,
                                          save_only_first_composition = True,
                                          simulate_velocity=True,
                                          karaoke=False):
                                                            
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

    if name_string[0] == 'SONG':
      song_name = name_string[1]

    output_list = []
    st = 0

    for i in range(2, len(input_string)-1):
      
      if save_only_first_composition:
        if input_string[i].split('=')[0] == 'SONG':

          song_name = name_string[1]
          break

      try:
        istring = input_string[i]
        #print(istring)

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
              out.append(0) # Channel
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
                out.append(istring[5:])
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
#
# TMIDI 2.0 Helper functions
#
###################################################################################

def Tegridy_MIDI_Zip_Notes_Summarizer(chords_list, max_number_of_unique_pitches = 128):

    '''Tegridy MIDI Zip Notes Summarizer
     
    Input: Flat chords list / SONG
           Max number of unique pitches allowed

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

      if o[4] not in pout:
        if len(pout) > max_number_of_unique_pitches:
          pout = []
        pout.append(o[4])
        out1.append(o)
        j += 1
      
      else:
        i += 1

    return out1, i

###################################################################################

def Tegridy_Score_Chords_Pairs_Generator(chords_list, shuffle_pairs = True):

    '''Tegridy Score Chords Pairs Generator
     
    Input: Flat chords list
           Shuffle pairs (recommended)

    Output: Score chords pairs list
            Number of created pairs

    Project Los Angeles
    Tegridy Code 2021'''

    chords = []
    cho = []

    i = 0

    chords_list.sort(reverse=False, key=lambda x: x[1])
    pcho = chords_list[0]
    for cc in chords_list[1:]:
      if cc[1] == pcho[1]:
        
        cho.append(cc)

      else:
        if cho != [] and pcho != []: chords.append([[pcho], cho])
        pcho = copy.deepcopy(cc)
        cho = []
        cho.append(cc)
        
        i += 1
      
    if cho != [] and pcho != []: 
      chords.append([[pcho], cho])
      
      i += 1
    
    if shuffle_pairs: random.shuffle(chords)

    return chords, i

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
                              fixed_start_time = 300, 
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
      if song[i][0] =='note':
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
        song1.append(song[i])

    average_delta_st = int(sum(delta) / len(delta))
    average_duration = int(sum([y[2] for y in song1]) / len([y[2] for y in song1]))

    song1.sort(reverse=False, key=lambda x: x[1])

    return song1, time, average_delta_st, average_duration

###################################################################################

def Tegridy_Score_Slicer(chords_list, number_of_miliseconds_per_slice=2000):

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

    i = 0

    chords_list1 = [x for x in chords_list if x]
    chords_list1.sort(reverse=False, key=lambda x: x[1])
    
    for cc in chords_list1:

      if cc[1] <= time:
        
        cho.append(cc)
      

      else:
        chords.append(cho)
        cho = []
        cho.append(cc)
        time += number_of_miliseconds_per_slice
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

def Tegridy_Optimus_TXT_to_INT_Converter(TXT_String, time_denominator=128):

  '''Project Los Angeles
     Tegridy Code 2021'''

  INT8_List = []

  for i in TXT_String:
    try:
      a = int((ord(i) // time_denominator) // 128)
      b = int(ord(i) % time_denominator)

      INT8_List.append(a)
      INT8_List.append(b)
    except:
      print('Bad TXT bits:', a, '/', b)
      continue

  return INT8_List

###################################################################################

def Tegridy_Optimus_INT_to_TXT_Converter(INT8_List, time_denominator=128):

  TXT_String = ''

  for i in range(0, len(INT8_List)-1, 2):
    try:
      a = INT8_List[i] * time_denominator * 128
      b = INT8_List[i+1]
    
      TXT_String += chr(a+b)
    except:
      print('Bad INT bits:', a, '/', b)
      continue

  return TXT_String

###################################################################################

def Tegridy_Optimus_Sum_Intro_Rand_End_Sampler(MIDI_file, number_of_notes_in_samples = 256):

  '''Project Los Angeles
     Tegridy Code 2021'''

  INTRO = []
  RAND = []
  END = []

  SUM = 0

  txt, melody_list, chords = Optimus_MIDI_TXT_Processor(MIDI_file, True, False, 1, False, False, -1, range(127))

  if len(chords) < number_of_notes_in_samples:
    number_of_notes_in_samples = len(chords)

  for i in chords[:number_of_notes_in_samples]:
    INTRO.append(i[4])

  r = secrets.randbelow(len(chords) - number_of_notes_in_samples)
  
  for i in chords[r:r+number_of_notes_in_samples]: 
    RAND.append(i[4])
  
  for i in chords[len(chords)-number_of_notes_in_samples:len(chords)]: 
    END.append(i[4])

  for i in chords:
    SUM += i[4]

  return SUM, INTRO, RAND, END

###################################################################################

def Tegridy_MIDI_Signature(melody, chords):

  '''Input: flat melody list and flat chords list
     
     Output: Melody signature and Chords signature
     
     Project Los Angeles
     Tegridy Code 2021'''

  # prepping data
  melody_list_f = []
  chords_list_f = []

  # melody
  m_st_sum = sum([y[1] for y in melody])
  m_st_avg = int(m_st_sum / len(melody))
  m_du_sum = sum([y[2] for y in melody])
  m_du_avg = int(m_du_sum / len(melody))
  m_ch_sum = sum([y[3] for y in melody])
  m_ch_avg = int(m_ch_sum / len(melody))
  m_pt_sum = sum([y[4] for y in melody])
  m_pt_avg = int(m_pt_sum / len(melody))
  m_vl_sum = sum([y[5] for y in melody])
  m_vl_avg = int(m_vl_sum / len(melody))
  
  melody_list_f.extend([m_st_sum, 
                        m_st_avg,
                        m_du_sum,
                        m_du_avg,
                        m_ch_sum,
                        m_ch_avg,
                        m_pt_sum,
                        m_pt_avg,
                        m_vl_sum,
                        m_vl_avg])

  # chords
  c_st_sum = sum([y[1] for y in chords])
  c_st_avg = int(c_st_sum / len(chords))
  c_du_sum = sum([y[2] for y in chords])
  c_du_avg = int(c_du_sum / len(chords))
  c_ch_sum = sum([y[3] for y in chords])
  c_ch_avg = int(c_ch_sum / len(chords))
  c_pt_sum = sum([y[4] for y in chords])
  c_pt_avg = int(c_pt_sum / len(chords))
  c_vl_sum = sum([y[5] for y in chords])
  c_vl_avg = int(c_vl_sum / len(chords))

  chords_list_f.extend([c_st_sum, 
                        c_st_avg,
                        c_du_sum,
                        c_du_avg,
                        c_ch_sum,
                        c_ch_avg,
                        c_pt_sum,
                        c_pt_avg,
                        c_vl_sum,
                        c_vl_avg])

  return melody_list_f, chords_list_f

###################################################################################

def Tegridy_List_Slicer(input_list, number_of_slices):

  '''Input: List to slice
            Number of desired slices
     
     Output: Sliced list of lists
     
     Project Los Angeles
     Tegridy Code 2021'''

  for i in range(0, len(input_list), number_of_slices):
     yield input_list[i:i + number_of_slices]

###################################################################################

# Alternative MIDI Processor (Google Magenta note-seq representation)
# Useful in some cases and for some Music AI implementations/architectures. 

# Based on the repo/code of Music X Lab at NYU Shanghai:
# https://github.com/music-x-lab/POP909-Dataset
# Please note that the license for the code below is MIT as requested by the dev.

###################################################################################

# Copyright (c) 2020 Music X Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###################################################################################

"""
Notes Representation Processor
============

These are core classes of representation processor.

Repr Processor: the basic representation processor
    - Event Processor
"""

class Tegridy_ReprProcessor(ABC):

    """Abstract base class severing as the representation processor.

    It provides the following abstract methods.
    - encode(self, note_seq): encode the note sequence into the representation sequence.
    - decode(self, repr_seq): decode the representation sequence into the note sequence.

    Notes
    -----
    The base representation processor class includes the convertion between the note sequence and the representation sequence.
    In general, we assume the input note sequence has already been quantized.
    In that, the smallest unit of the quantization is actually 1 tick no matter what resolution is.
    If you init "min_step" to be larger than 1, we assume you wish to compress all the base tick.
    e.g. min_step = 2, then the whole ticks will be convertd half.
    If you do this, the representation convertion may not be 100% correct.
    -----

    """

    def __init__(self, min_step: int = 1):
        self.min_step = min_step

    def _compress(self, note_seq=None):
        """Return the compressed note_seq based on the min_step > 1.

        Parameters
        ----------
        note_seq : Note Array.
        ----------

        WARNING: If you do this, the representation convertion may not be 100% correct.

        """
        new_note_seq = [
            Note(
                start=int(d.start / self.min_step),
                end=int(d.end / self.min_step),
                pitch=d.pitch,
                velocity=d.velocity,
            )
            for d in note_seq
        ]
        return new_note_seq

    def _expand(self, note_seq=None):
        """Return the expanded note_seq based on the min_step > 1.

        Parameters
        ----------
        note_seq : Note Array.
        ----------

        WARNING: If you do this, the representation convertion may not be 100% correct.

        """
        new_note_seq = [
            Note(
                start=int(d.start * self.min_step),
                end=int(d.end * self.min_step),
                pitch=d.pitch,
                velocity=d.velocity,
            )
            for d in note_seq
        ]
        return new_note_seq

    @abstractmethod
    def encode(self, note_seq=None):
        """encode the note sequence into the representation sequence.

        Parameters
        ----------
        note_seq= the input {Note} sequence

        Returns
        ----------
        repr_seq: the representation numpy sequence

        """

    @abstractmethod
    def decode(self, repr_seq=None):
        """decode the representation sequence into the note sequence.

        Parameters
        ----------
        repr_seq: the representation numpy sequence

        Returns
        ----------
        note_seq= the input {Note} sequence

        """

###################################################################################
# pretty_midi Note class
# Code is from original pretty_midi repo:
# https://github.com/craffel/pretty-midi/blob/master/pretty_midi/containers.py
###################################################################################
class Note(object):
    """A note event.
    Parameters
    ----------
    velocity : int
        Note velocity.
    pitch : int
        Note pitch, as a MIDI note number.
    start : float
        Note on time, absolute, in seconds.
    end : float
        Note off time, absolute, in seconds.
    """

    def __init__(self, velocity, pitch, start, end):
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def get_duration(self):
        """Get the duration of the note in seconds."""
        return self.end - self.start

    @property
    def duration(self):
        return self.get_duration()

    def __repr__(self):
        return 'Note(start={:f}, end={:f}, pitch={}, velocity={})'.format(
            self.start, self.end, self.pitch, self.velocity)

###################################################################################

class Tegridy_RPR_MidiEventProcessor(Tegridy_ReprProcessor):
  
    """Midi Event Representation Processor.

    Representation Format:
    -----
    Size: L * D:
        - L for the sequence (event) length
        - D = 1 {
            0-127: note-on event,
            128-255: note-off event,
            256-355(default):
                tick-shift event
                256 for one tick, 355 for 100 ticks
                the maximum number of tick-shift can be specified
            356-388 (default):
                velocity event
                the maximum number of quantized velocity can be specified
            }

    Parameters:
    -----
    min_step(optional):
        minimum quantification step
        decide how many ticks to be the basic unit (default = 1)
    tick_dim(optional):
        tick-shift event dimensions
        the maximum number of tick-shift (default = 100)
    velocity_dim(optional):
        velocity event dimensions
        the maximum number of quantized velocity (default = 32, max = 128)

    e.g.

    [C5 - - - E5 - - / G5 - - / /]
    ->
    [380, 60, 259, 188, 64, 258, 192, 256, 67, 258, 195, 257]

    """

    def __init__(self, **kwargs):
        self.name = "midievent"
        min_step = 1
        if "min_step" in kwargs:
            min_step = kwargs["min_step"]
        super(Tegridy_RPR_MidiEventProcessor, self).__init__(min_step)
        self.tick_dim = 100
        self.velocity_dim = 32
        if "tick_dim" in kwargs:
            self.tick_dim = kwargs["tick_dim"]
        if "velocity_dim" in kwargs:
            self.velocity_dim = kwargs["velocity_dim"]
        if self.velocity_dim > 128:
            raise ValueError(
                "velocity_dim cannot be larger than 128", self.velocity_dim
            )
        self.max_vocab = 256 + self.tick_dim + self.velocity_dim
        self.start_index = {
            "note_on": 0,
            "note_off": 128,
            "time_shift": 256,
            "velocity": 256 + self.tick_dim,
        }

    def encode(self, note_seq=None):
        """Return the note token

        Parameters
        ----------
        note_seq : Note List.

        Returns
        ----------
        repr_seq: Representation List

        """
        if note_seq is None:
            return []
        if self.min_step > 1:
            note_seq = self._compress(note_seq)
        notes = note_seq
        events = []
        meta_events = []
        for note in notes:
            token_on = {
                "name": "note_on",
                "time": note.start,
                "pitch": note.pitch,
                "vel": note.velocity,
            }
            token_off = {
                "name": "note_off",
                "time": note.end,
                "pitch": note.pitch,
                "vel": None,
            }
            meta_events.extend([token_on, token_off])
        meta_events.sort(key=lambda x: x["pitch"])    
        meta_events.sort(key=lambda x: x["time"])
        time_shift = 0
        cur_vel = 0
        for me in meta_events:
            duration = int((me["time"] - time_shift) * 100)
            while duration >= self.tick_dim:
                events.append(
                    self.start_index["time_shift"] + self.tick_dim - 1
                )
                duration -= self.tick_dim
            if duration > 0:
                events.append(self.start_index["time_shift"] + duration - 1)
            if me["vel"] is not None:
                if cur_vel != me["vel"]:
                    cur_vel = me["vel"]
                    events.append(
                        self.start_index["velocity"]
                        + int(round(me["vel"] * self.velocity_dim / 128))
                    )
            events.append(self.start_index[me["name"]] + me["pitch"])
            time_shift = me["time"]
        return events

    def decode(self, repr_seq=None):
        """Return the note seq

        Parameters
        ----------
        repr_seq: Representation Sequence List

        Returns
        ----------
        note_seq : Note List.

        """
        # print(repr_seq)
        if repr_seq is None:
            return []
        time_shift = 0.0
        cur_vel = 0
        meta_events = []
        note_on_dict = {}
        notes = []
        for e in repr_seq:
            if self.start_index["note_on"] <= e < self.start_index["note_off"]:
                token_on = {
                    "name": "note_on",
                    "time": time_shift,
                    "pitch": e,
                    "vel": cur_vel,
                }
                meta_events.append(token_on)
            if (
                self.start_index["note_off"]
                <= e
                < self.start_index["time_shift"]
            ):
                token_off = {
                    "name": "note_off",
                    "time": time_shift,
                    "pitch": e - self.start_index["note_off"],
                    "vel": cur_vel,
                }
                meta_events.append(token_off)
            if (
                self.start_index["time_shift"]
                <= e
                < self.start_index["velocity"]
            ):
                time_shift += (e - self.start_index["time_shift"] + 1) * 0.01
            if self.start_index["velocity"] <= e < self.max_vocab:
                cur_vel = int(round(
                    (e - self.start_index["velocity"])
                    * 128
                    / self.velocity_dim)
                )
        skip_notes = []
        for me in meta_events:
            if me["name"] == "note_on":
                note_on_dict[me["pitch"]] = me
            elif me["name"] == "note_off":
                try:
                    token_on = note_on_dict[me["pitch"]]
                    token_off = me
                    if token_on["time"] == token_off["time"]:
                        continue
                    notes.append(
                        Note(
                            velocity=token_on["vel"],
                            pitch=int(token_on["pitch"]),
                            start=token_on["time"],
                            end=token_off["time"],
                        )
                    )
                except:
                    skip_notes.append(me)
        notes.sort(key=lambda x: x.start)
        if self.min_step > 1:
            notes = self._expand(notes)
        # print(notes)
        return notes

###################################################################################

# This is the end of the TMIDI Python module

###################################################################################
