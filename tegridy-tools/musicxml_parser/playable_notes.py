def get_playable_notes(xml_part, melody_only=False):
    notes = []
    measure_number = 1
    for measure in xml_part.measures:
        for note in measure.notes:
            note.measure_number = measure_number
            notes.append(note)
        measure_number += 1

    notes, rests = classify_notes(notes, melody_only=melody_only)
    mark_preceded_by_grace_note_to_chord_notes(notes)
    if melody_only:
        notes = delete_chord_notes_for_melody(notes)
    notes = apply_tied_notes(notes)
    notes.sort(key=lambda x: (x.note_duration.xml_position,
                x.note_duration.grace_order, -x.pitch[1]))
    notes = check_overlapped_notes(notes)
    notes = apply_rest_to_note(notes, rests)
    notes = omit_trill_notes(notes)
    notes = extract_and_apply_slurs(notes)
    # notes = self.rearrange_chord_index(notes)
    return notes, rests


def classify_notes(notes, melody_only=False):
    # classify notes into notes, and rests.
    # calculate grace note order, mark note with preceeding grace notes
    grace_tmp = []
    rests = []
    notes_tmp = []
    for note in notes:
        if melody_only:
            if note.voice != 1:
                continue
        if note.note_duration.is_grace_note:
            grace_tmp.append(note)
            notes_tmp.append(note)
        elif not note.is_rest:
            if len(grace_tmp) > 0:
                rest_grc = []
                added_grc = []
                grace_order = -1
                for grc in reversed(grace_tmp):
                    if grc.voice == note.voice:
                        note.note_duration.preceded_by_grace_note = True
                        grc.note_duration.grace_order = grace_order
                        grc.following_note = note
                        if grc.chord_index == 0:
                            grace_order -= 1
                        added_grc.append(grc)
                    else:
                        rest_grc.append(grc)
                num_added = abs(grace_order) - 1
                for grc in added_grc:
                    # grc.note_duration.grace_order /= num_added
                    grc.note_duration.num_grace = num_added
                if abs(grc.note_duration.grace_order) == num_added:
                    grc.note_duration.is_first_grace_note = True
                grace_tmp = rest_grc
            notes_tmp.append(note)
        else:
            assert note.is_rest
            if note.is_print_object:
                rests.append(note)

    return notes_tmp, rests


def mark_preceded_by_grace_note_to_chord_notes(notes):
    for note in notes:
        if note.note_duration.preceded_by_grace_note:
            onset = note.note_duration.xml_position
            voice = note.voice
            chords = [note for note in notes if
                      note.note_duration.xml_position == onset and
                      note.voice == voice and
                      not note.note_duration.is_grace_note]
            for chd in chords:
                chd.note_duration.preceded_by_grace_note = True


def delete_chord_notes_for_melody(melody_notes):
    note_onset_positions = list(
        set(note.note_duration.xml_position for note in melody_notes))
    note_onset_positions.sort()
    unique_melody = []
    for onset in note_onset_positions:
        notes = [
            note for note in melody_notes if note.note_duration.xml_position == onset]
        if len(notes) == 1:
            unique_melody.append(notes[0])
        else:
            notes.sort(key=lambda x: x.pitch[1])
            unique_melody.append(notes[-1])

    return unique_melody


def apply_tied_notes(notes):
    tie_clean_list = []
    for i in range(len(notes)):
        if notes[i].note_notations.tied_stop == False:
            tie_clean_list.append(notes[i])
        else:
            for j in reversed(range(len(tie_clean_list))):
                if tie_clean_list[j].note_notations.tied_start and tie_clean_list[j].pitch[1] == notes[i].pitch[1]:
                    tie_clean_list[j].note_duration.seconds += notes[i].note_duration.seconds
                    tie_clean_list[j].note_duration.duration += notes[i].note_duration.duration
                    tie_clean_list[j].note_duration.midi_ticks += notes[i].note_duration.midi_ticks
                    if notes[i].note_notations.slurs:
                        for slur in notes[i].note_notations.slurs:
                            tie_clean_list[j].note_notations.slurs.append(slur)
                    break
    return tie_clean_list


def check_overlapped_notes(notes):
    previous_onset = -1
    notes_on_onset = []
    pitches = []
    for note in notes:
        if note.note_duration.is_grace_note:
            continue  # does not count grace note, because it can have same pitch on same xml_position
        if note.note_duration.xml_position > previous_onset:
            previous_onset = note.note_duration.xml_position
            pitches = []
            pitches.append(note.pitch[1])
            notes_on_onset = []
            notes_on_onset.append(note)
        else:  # note has same xml_position
            if note.pitch[1] in pitches:  # same pitch with same
                index_of_same_pitch_note = pitches.index(note.pitch[1])
                previous_note = notes_on_onset[index_of_same_pitch_note]
                if previous_note.note_duration.duration > note.note_duration.duration:
                    note.is_overlapped = True
                else:
                    previous_note.is_overlapped = True
            else:
                pitches.append(note.pitch[1])
                notes_on_onset.append(note)

    return notes


def apply_rest_to_note(notes, rests):
    xml_positions = [note.note_duration.xml_position for note in notes]
    # concat continuous rests
    new_rests = []
    num_rests = len(rests)
    for i in range(num_rests):
        rest = rests[i]
        j = 1
        current_end = rest.note_duration.xml_position + rest.note_duration.duration
        current_voice = rest.voice
        while i + j < num_rests - 1:
            next_rest = rests[i + j]
            if next_rest.note_duration.duration == 0:
              break
            if next_rest.note_duration.xml_position == current_end and next_rest.voice == current_voice:
                rest.note_duration.duration += next_rest.note_duration.duration
                next_rest.note_duration.duration = 0
                current_end = rest.note_duration.xml_position + rest.note_duration.duration
                if next_rest.note_notations.is_fermata:
                    rest.note_notations.is_fermata = True
            elif next_rest.note_duration.xml_position > current_end:
                break
            j += 1

        if not rest.note_duration.duration == 0:
            new_rests.append(rest)

    rests = new_rests

    for rest in rests:
        rest_position = rest.note_duration.xml_position
        closest_note_index = binary_index(xml_positions, rest_position)
        rest_is_fermata = rest.note_notations.is_fermata

        search_index = 0
        while closest_note_index - search_index >= 0:
            prev_note = notes[closest_note_index - search_index]
            if prev_note.voice == rest.voice:
                prev_note_end = prev_note.note_duration.xml_position + prev_note.note_duration.duration
                prev_note_with_rest = prev_note_end + prev_note.following_rest_duration
                if prev_note_end == rest_position:
                    prev_note.following_rest_duration = rest.note_duration.duration
                    if rest_is_fermata:
                        prev_note.followed_by_fermata_rest = True
                elif prev_note_end < rest_position:
                    break
            # elif prev_note_with_rest == rest_position and prev_note.voice == rest.voice:
            #     prev_note.following_rest_duration += rest.note_duration.duration
            search_index += 1

    return notes



def omit_trill_notes(notes):
    def _combine_wavy_lines(wavy_lines):
        num_wavy = len(wavy_lines)
        for i in reversed(range(num_wavy)):
            wavy = wavy_lines[i]
            if wavy.type == 'stop':
                deleted = False
                for j in range(1, i + 1):
                    prev_wavy = wavy_lines[i - j]
                    if prev_wavy.type == 'start' and prev_wavy.number == wavy.number:
                        prev_wavy.end_xml_position = wavy.xml_position
                        wavy_lines.remove(wavy)
                        deleted = True
                        break
                if not deleted:
                    wavy_lines.remove(wavy)
        num_wavy = len(wavy_lines)
        for i in reversed(range(num_wavy)):
            wavy = wavy_lines[i]
            if wavy.end_xml_position == 0:
                wavy_lines.remove(wavy)
        return wavy_lines

    def _apply_wavy_lines(notes, wavy_lines):
        xml_positions = [x.note_duration.xml_position for x in notes]
        num_notes = len(notes)
        omit_indices = []
        for wavy in wavy_lines:
            index = binary_index(xml_positions, wavy.xml_position)
            while abs(notes[index].pitch[1] - wavy.pitch[1]) > 3 and index > 0 \
                    and notes[index - 1].note_duration.xml_position == notes[index].note_duration.xml_position:
                    index -= 1
            note = notes[index]
            wavy_duration = wavy.end_xml_position - wavy.xml_position
            note.note_duration.duration = wavy_duration
            trill_pitch = note.pitch[1]
            next_idx = index + 1
            while next_idx < num_notes and notes[next_idx].note_duration.xml_position < wavy.end_xml_position:
                if notes[next_idx].pitch[1] == trill_pitch:
                    omit_indices.append(next_idx)
                next_idx += 1

        omit_indices.sort()
        if len(omit_indices) > 0:
            for idx in reversed(omit_indices):
                del notes[idx]

        return notes

    num_notes = len(notes)
    omit_index = []
    trill_sign = []
    wavy_lines = []
    for i in range(num_notes):
      note = notes[i]
      if not note.is_print_object:
        omit_index.append(i)
        if note.accidental:
          # TODO: handle accidentals in non-print notes
          if note.accidental == 'natural':
            pass
          elif note.accidental == 'sharp':
            pass
          elif note.accidental == 'flat':
            pass
        if note.note_notations.is_trill:
          trill = {'xml_pos': note.note_duration.xml_position, 'pitch': note.pitch[1]}
          trill_sign.append(trill)
      if note.note_notations.wavy_line:
        wavy_line = note.note_notations.wavy_line
        wavy_line.xml_position = note.note_duration.xml_position
        wavy_line.pitch = note.pitch
        wavy_lines.append(wavy_line)

      # move trill mark to the highest notes of the onset
      if note.note_notations.is_trill:
        notes_in_trill_onset = []
        current_position = note.note_duration.xml_position

        search_index = i
        while search_index + 1 < num_notes and notes[
          search_index + 1].note_duration.xml_position == current_position:
          search_index += 1
          notes_in_trill_onset.append(notes[search_index])
        search_index = i

        while search_index - 1 >= 0 and notes[search_index - 1].note_duration.xml_position == current_position:
          search_index -= 1
          notes_in_trill_onset.append(notes[search_index])

        for other_note in notes_in_trill_onset:
          highest_note = note
          if other_note.voice == note.voice and other_note.pitch[1] > highest_note.pitch[
            1] and not other_note.note_duration.is_grace_note:
            highest_note.note_notations.is_trill = False
            other_note.note_notations.is_trill = True

    wavy_lines = _combine_wavy_lines(wavy_lines)

    for index in reversed(omit_index):
      note = notes[index]
      notes.remove(note)

    if len(trill_sign) > 0:
      for trill in trill_sign:
        for note in notes:
          if note.note_duration.xml_position == trill['xml_pos'] and abs(note.pitch[1] - trill['pitch']) < 4 \
                  and not note.note_duration.is_grace_note:
            note.note_notations.is_trill = True
            break

    notes = _apply_wavy_lines(notes, wavy_lines)

    return notes


def extract_and_apply_slurs(notes):
    resolved_slurs = []
    unresolved_slurs = []
    slur_index = 0
    for note in notes:
        slurs = note.note_notations.slurs
        if slurs:
            for slur in reversed(slurs):
                slur.xml_position = note.note_duration.xml_position
                slur.voice = note.voice
                if slur.type == 'start':
                    slur.index = slur_index
                    unresolved_slurs.append(slur)
                    slur_index += 1
                    note.note_notations.is_slur_start = True
                elif slur.type == 'stop':
                    note.note_notations.is_slur_stop = True
                    for prev_slur in unresolved_slurs:
                        if prev_slur.number == slur.number and prev_slur.voice == slur.voice:
                            prev_slur.end_xml_position = slur.xml_position
                            resolved_slurs.append(prev_slur)
                            unresolved_slurs.remove(prev_slur)
                            note.note_notations.slurs.remove(slur)
                            note.note_notations.slurs.append(prev_slur)
                            break

    for note in notes:
        slurs = note.note_notations.slurs
        note_position = note.note_duration.xml_position
        if not slurs:
            for prev_slur in resolved_slurs:
                if prev_slur.voice == note.voice and prev_slur.xml_position <= note_position <= prev_slur.end_xml_position:
                    note.note_notations.slurs.append(prev_slur)
                    if prev_slur.xml_position == note_position:
                        note.note_notations.is_slur_start = True
                    elif prev_slur.end_xml_position == note_position:
                        note.note_notations.is_slur_stop = True
                    else:
                        note.note_notations.is_slur_continue = True

    return notes


def binary_index(alist, item):
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
