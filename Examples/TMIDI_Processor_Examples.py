###################################################################
# TMIDI use examples
###################################################################

# Requirements...

#pip install tqdm
#pip install pickle
#pip install datetime

###################################################################
# TMIDI module setup...

import os, sys
sys.path.append('tegridy-tools')
sys.path.append('tegridy-tools/tegridy-tools')
import TMIDI

# Importing the rest of the required modules...
import tqdm.auto
import pickle
import MIDI
from datetime import datetime
import multiprocessing

###################################################################
#
# Multi-threaded MIDI Reader example
#
###################################################################
def main(): # Main is for multi-processing on Win...

	print('Multi-threaded MIDI Processor wrapper example')
	print('Starting up...')

	full_path_to_output_dataset_to = "Meddleying-VIRTUOSO-Music-Dataset.data" #@param {type:"string"}
	enable_multi_threading = False #@param {type:"boolean"}
	number_of_parallel_threads = 1 #@param {type:"slider", min:1, max:128, step:1}
	minimum_number_of_notes_per_chord = 2 #@param {type:"slider", min:2, max:10, step:1}

	debug = False #@param {type:"boolean"}

###################################################################

	average_note_pitch = 0
	min_note = 127
	max_note = 0

	files_count = 0

	ev = 0

	chords_list_final = []
	chords_list = ['note', 0, 0, 0, 0, 0]

	melody_chords= []

###################################################################

	def list_average(num):
	  sum_num = 0
	  for t in num:
	      sum_num = sum_num + t           

	  avg = sum_num / len(num)
	  return avg

###################################################################

	print('Loading MIDI files...')
	print('This may take a while on a large dataset in particular.')



	dataset_addr = 'Dataset'

	if not os.path.exists(dataset_addr):
	    os.makedirs(dataset_addr)
	

	files = os.listdir(dataset_addr)
	#print(files)
	
	#print('Now processing the files.')
	#print('Please stand-by...')
	
	if enable_multi_threading: # This code may need to be adjusted to work on different platforms/OSs.

		print('Processing files with multi-threading enabled. Please wait...')
  
		# Main multi-processing loop for MIDIs

		pool = multiprocessing.Pool(processes=number_of_parallel_threads)
		chords = pool.map(TMIDI.Tegridy_MIDI_Processor, files, 0, 1)
		chords_list_final = [chords[0][0]]
		melody = chords[0][1]
		pool.close()
		pool.join()
	else:
		os.chdir(dataset_addr)
		print('Processing files with multi-threading disabled. Please wait...')
		for f in tqdm.auto.tqdm(files):
			files_count += 1
			chords_list, melody = TMIDI.Tegridy_MIDI_Processor(f, 0, 1)

			chords_list_final.extend(chords_list)

			melody_chords.append([files_count, chords_list_final, melody])

	average_note_pitch = int((min_note + max_note) / 2)

	print('Task complete :)')
	print('==================================================')
	print('Number of processed dataset MIDI files:', files_count)
	print('Average note pitch =', average_note_pitch)
	print('Min note pitch =', min_note)
	print('Max note pitch =', max_note)
	#print('Number of MIDI events recorded:', len(events_matrix))
	print('Number of MIDI chords recorded:', len(chords_list_final))
	print('The longest chord:', max(chords_list_final, key=len))

	# Dataset
	MusicDataset = [chords_list]
	os.chdir('../')
	with open(full_path_to_output_dataset_to, 'wb') as filehandle:
	    # store the data as binary data stream
	    pickle.dump(MusicDataset, filehandle)
	print('Dataset was saved at:', full_path_to_output_dataset_to)
	print('Task complete. Enjoy! :)')

###################################################################



	###########################################################
	#
	# MIDI Writer example
	#
	###########################################################

	print('Generating output MIDI file...')
	output_score = melody
	output_ticks = 1000 #score[0]
	output_header = [['track_name', 0, b'TMIDI Processor Composition']] #+ score[1][:3] 

	output_opus = output_header + output_score
	output = [output_ticks,  output_opus]
	midi_data = MIDI.score2midi(output)

	# Stuff for datetime stamp
	now = ''
	now_n = str(datetime.now())
	now_n = now_n.replace(' ', '_')
	now_n = now_n.replace(':', '_')
	now = now_n.replace('.', '_')
    
	fname = './Meddleying-VIRTUOSO-Composition-' + 'generated-on-' + str(now) + '.mid'  

	# Writing MIDI
	print('Writing output MIDI file to disk...')
	with open(fname, 'wb') as midi_file1:
	    midi_file1.write(midi_data)
	    midi_file1.close()
	print('Done!')

	# Stats
	print('Crunching quick stats...')
	print('First Event:', output[1][0], '=== Last Event:', output[1][-1])
	#print('The dataset was scanned', dts, 'times.')
	#print('Generated', i, 'notes for', len(melody_events), 'input melody notes.')
	#print('Generated', c, 'chords for', len(melody_events), 'input melody notes.')
	print('Task complete!')

	# MIDI.py detailed output MIDI stats
	print('Final output MIDI stats:')
	print(MIDI.score2stats(output))

os.chdir('../')

if __name__ == '__main__':
	main()
else:
	main()
