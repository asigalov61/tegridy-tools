#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Tegridy Plots Python Module (TPLOTS)
#	    Version 1.0
#
#	    Project Los Angeles
#
#	    Tegridy Code 2025
#
#       https://github.com/asigalov61/tegridy-tools
#
#
################################################################################
#
#       Copyright 2024 Project Los Angeles / Tegridy Code
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
#
################################################################################
################################################################################
#
#       Critical dependencies
#
#       !pip install numpy==1.24.4
#       !pip install scipy
#       !pip install matplotlib
#       !pip install networkx
#       !pip3 install scikit-learn
#
################################################################################
#
#       Future critical dependencies
#
#       !pip install umap-learn
#       !pip install alphashape
#
################################################################################
'''

################################################################################
# Modules imports
################################################################################

import os
from collections import Counter
from itertools import groupby

import numpy as np

import networkx as nx

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.ndimage import zoom
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import zscore

import matplotlib.pyplot as plt
from PIL import Image

################################################################################
# Constants
################################################################################

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

################################################################################

CHORDS_TYPES = ['WHITE', 'BLACK', 'UNKNOWN', 'MIXED WHITE', 'MIXED BLACK', 'MIXED GRAY']

################################################################################

WHITE_NOTES = [0, 2, 4, 5, 7, 9, 11]

################################################################################

BLACK_NOTES = [1, 3, 6, 8, 10]

################################################################################
# Helper functions
################################################################################

def tones_chord_type(tones_chord, 
                     return_chord_type_index=True,
                     ):

  """
  Returns tones chord type
  """

  WN = WHITE_NOTES
  BN = BLACK_NOTES
  MX = WHITE_NOTES + BLACK_NOTES


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

  """
  Returns tone type
  """

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

def find_closest_points(points, return_points=True):

  """
  Find closest 2D points
  """

  coords = np.array(points)

  num_points = coords.shape[0]
  closest_matches = np.zeros(num_points, dtype=int)
  distances = np.zeros((num_points, num_points))

  for i in range(num_points):
      for j in range(num_points):
          if i != j:
              distances[i, j] = np.linalg.norm(coords[i] - coords[j])
          else:
              distances[i, j] = np.inf

  closest_matches = np.argmin(distances, axis=1)
  
  if return_points:
    points_matches = coords[closest_matches].tolist()
    return points_matches
  
  else:
    return closest_matches.tolist()

################################################################################

def reduce_dimensionality_tsne(list_of_valies,
                                n_comp=2,
                                n_iter=5000,
                                verbose=True
                              ):

  """
  Reduces the dimensionality of the values using t-SNE.
  """

  vals = np.array(list_of_valies)

  tsne = TSNE(n_components=n_comp,
              n_iter=n_iter,
              verbose=verbose)

  reduced_vals = tsne.fit_transform(vals)

  return reduced_vals.tolist()

################################################################################

def compute_mst_edges(similarity_scores_list):

  """
  Computes the Minimum Spanning Tree (MST) edges based on the similarity scores.
  """
  
  num_tokens = len(similarity_scores_list[0])

  graph = nx.Graph()

  for i in range(num_tokens):
      for j in range(i + 1, num_tokens):
          weight = 1 - similarity_scores_list[i][j]
          graph.add_edge(i, j, weight=weight)

  mst = nx.minimum_spanning_tree(graph)

  mst_edges = list(mst.edges(data=False))

  return mst_edges

################################################################################

def square_binary_matrix(binary_matrix, 
                         matrix_size=128,
                         interpolation_order=5,
                         return_square_matrix_points=False
                         ):

  """
  Reduces an arbitrary binary matrix to a square binary matrix
  """

  zoom_factors = (matrix_size / len(binary_matrix), 1)

  resized_matrix = zoom(binary_matrix, zoom_factors, order=interpolation_order)

  resized_matrix = (resized_matrix > 0.5).astype(int)

  final_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
  final_matrix[:, :resized_matrix.shape[1]] = resized_matrix

  points = np.column_stack(np.where(final_matrix == 1)).tolist()

  if return_square_matrix_points:
    return points

  else:
    return resized_matrix

################################################################################

def square_matrix_points_colors(square_matrix_points):

  """
  Returns colors for square matrix points
  """

  cmap = generate_colors(12)

  chords = []
  chords_dict = set()
  counts = []

  for k, v in groupby(square_matrix_points, key=lambda x: x[0]):
    pgroup = [vv[1] for vv in v]
    chord = sorted(set(pgroup))
    tchord = sorted(set([p % 12 for p in chord]))
    chords_dict.add(tuple(tchord))
    chords.append(tuple(tchord))
    counts.append(len(pgroup))

  chords_dict = sorted(chords_dict)

  colors = []

  for i, c in enumerate(chords):
    colors.extend([cmap[round(sum(c) / len(c))]] * counts[i])

  return colors

################################################################################

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

################################################################################

def generate_colors(n):
  return [hsv_to_rgb(i/n, 1, 1) for i in range(n)]

################################################################################

def add_arrays(a, b):
  return [sum(pair) for pair in zip(a, b)]

################################################################################

def calculate_similarities(lists_of_values, metric='cosine'):
  return metrics.pairwise_distances(lists_of_values, metric=metric).tolist()

################################################################################

def get_tokens_embeddings(x_transformer_model):
  return x_transformer_model.net.token_emb.emb.weight.detach().cpu().tolist()

################################################################################

def minkowski_distance_matrix(X, p=3):

  X = np.array(X)

  n = X.shape[0]
  dist_matrix = np.zeros((n, n))

  for i in range(n):
      for j in range(n):
          dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j])**p)**(1/p)

  return dist_matrix.tolist()

################################################################################

def robust_normalize(values):

  values = np.array(values)
  q1 = np.percentile(values, 25)
  q3 = np.percentile(values, 75)
  iqr = q3 - q1

  filtered_values = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]

  min_val = np.min(filtered_values)
  max_val = np.max(filtered_values)
  normalized_values = (values - min_val) / (max_val - min_val)

  normalized_values = np.clip(normalized_values, 0, 1)

  return normalized_values.tolist()

################################################################################

def min_max_normalize(values):

  scaler = MinMaxScaler()

  return scaler.fit_transform(values).tolist()

################################################################################

def remove_points_outliers(points, z_score_threshold=3):

  points = np.array(points)

  z_scores = np.abs(zscore(points, axis=0))

  return points[(z_scores < z_score_threshold).all(axis=1)].tolist()

################################################################################

def generate_labels(lists_of_values, 
                    return_indices_labels=False
                    ):

  ordered_indices = list(range(len(lists_of_values)))
  ordered_indices_labels = [str(i) for i in ordered_indices]
  ordered_values_labels = [str(lists_of_values[i]) for i in ordered_indices]

  if return_indices_labels:
    return ordered_indices_labels
  
  else:
    return ordered_values_labels

################################################################################

def reduce_dimensionality_pca(list_of_values, n_components=2):

  """
  Reduces the dimensionality of the values using PCA.
  """

  pca = PCA(n_components=n_components)
  pca_data = pca.fit_transform(list_of_values)
  
  return pca_data.tolist()

def reduce_dimensionality_simple(list_of_values, 
                                 return_means=True,
                                 return_std_devs=True,
                                 return_medians=False,
                                 return_vars=False
                                 ):
  
  '''
  Reduces dimensionality of the values in a simple way
  '''

  array = np.array(list_of_values)
  results = []

  if return_means:
      means = np.mean(array, axis=1)
      results.append(means)

  if return_std_devs:
      std_devs = np.std(array, axis=1)
      results.append(std_devs)

  if return_medians:
      medians = np.median(array, axis=1)
      results.append(medians)

  if return_vars:
      vars = np.var(array, axis=1)
      results.append(vars)

  merged_results = np.column_stack(results)
  
  return merged_results.tolist()

################################################################################

def reduce_dimensionality_2d_distance(list_of_values, p=5):

  '''
  Reduces the dimensionality of the values using 2d distance
  '''

  values = np.array(list_of_values)

  dist_matrix = distance_matrix(values, values, p=p)

  mst = minimum_spanning_tree(dist_matrix).toarray()

  points = []

  for i in range(len(values)):
      for j in range(len(values)):
          if mst[i, j] > 0:
              points.append([i, j])

  return points

################################################################################

def normalize_to_range(values, n):
    
  min_val = min(values)
  max_val = max(values)
  
  range_val = max_val - min_val
  
  normalized_values = [((value - min_val) / range_val * 2 * n) - n for value in values]
  
  return normalized_values

################################################################################

def reduce_dimensionality_simple_pca(list_of_values, n_components=2):

  '''
  Reduces the dimensionality of the values using simple PCA
  '''

  reduced_values = []

  for l in list_of_values:

    norm_values = [round(v * len(l)) for v in normalize_to_range(l, (n_components+1) // 2)]

    pca_values = Counter(norm_values).most_common()
    pca_values = [vv[0] / len(l) for vv in pca_values]
    pca_values = pca_values[:n_components]
    pca_values = pca_values + [0] * (n_components - len(pca_values))

    reduced_values.append(pca_values)

  return reduced_values

################################################################################

def filter_and_replace_values(list_of_values, 
                              threshold, 
                              replace_value, 
                              replace_above_threshold=False
                              ):

  array = np.array(list_of_values)

  modified_array = np.copy(array)
  
  if replace_above_threshold:
    modified_array[modified_array > threshold] = replace_value
  
  else:
    modified_array[modified_array < threshold] = replace_value
  
  return modified_array.tolist()

################################################################################

def find_shortest_constellation_path(points, 
                                     start_point_idx, 
                                     end_point_idx,
                                     p=5,
                                     return_path_length=False,
                                     return_path_points=False,
                                     ):

    """
    Finds the shortest path between two points of the points constellation
    """

    points = np.array(points)

    dist_matrix = distance_matrix(points, points, p=p)

    mst = minimum_spanning_tree(dist_matrix).toarray()

    G = nx.Graph()

    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                G.add_edge(i, j, weight=mst[i, j])

    path = nx.shortest_path(G, 
                            source=start_point_idx, 
                            target=end_point_idx, 
                            weight='weight'
                            )
    
    path_length = nx.shortest_path_length(G, 
                                          source=start_point_idx, 
                                          target=end_point_idx, 
                                          weight='weight')
        
    path_points = points[np.array(path)].tolist()


    if return_path_points:
      return path_points

    if return_path_length:
      return path_length

    return path

################################################################################
# Core functions
################################################################################

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

  '''ms SONG plot'''

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

    for start, duration, pitch, patch in zip(start_times, durations, pitches, patches):
        rect = plt.Rectangle((start, pitch), duration, note_height, facecolor=colors[patch])
        ax.add_patch(rect)

    ax.set_xlim([min(start_times), max(add_arrays(start_times, durations))])
    ax.set_ylim([min(pitches)-1, max(pitches)+1])

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
        plt.savefig(save_plt, 
                    transparent=save_transparent, 
                    bbox_inches='tight', 
                    pad_inches=0, 
                    facecolor='black'
                    )
        plt.close()
      
      else:
        plt.savefig(save_plt)
        plt.close()

    if return_plt:
      return fig

    plt.show()
    plt.close()

################################################################################

def plot_square_matrix_points(list_of_points,
                              list_of_points_colors,
                              plot_size=(7, 7),
                              point_size = 10,
                              show_grid_lines=False,
                              plot_title = 'Square Matrix Points Plot',
                              return_plt=False,
                              save_plt='',
                              save_only_plt_image=True,
                              save_transparent=False
                              ):

  '''Square matrix points plot'''

  fig, ax = plt.subplots(figsize=plot_size)

  ax.set_facecolor('black')

  if show_grid_lines:
    ax.grid(color='white')

  plt.xlabel('Time Step', c='black')
  plt.ylabel('MIDI Pitch', c='black')

  plt.title(plot_title)

  plt.scatter([p[0] for p in list_of_points], 
              [p[1] for p in list_of_points], 
              c=list_of_points_colors, 
              s=point_size
              )

  if save_plt != '':
    if save_only_plt_image:
      plt.axis('off')
      plt.title('')
      plt.savefig(save_plt, 
                  transparent=save_transparent, 
                  bbox_inches='tight', 
                  pad_inches=0, 
                  facecolor='black'
                  )
      plt.close()
    
    else:
      plt.savefig(save_plt)
      plt.close()

  if return_plt:
    return fig

  plt.show()
  plt.close()

################################################################################

def plot_cosine_similarities(lists_of_values,
                             plot_size=(7, 7),
                             save_plot=''
                            ):

  """
  Cosine similarities plot
  """

  cos_sim = metrics.pairwise_distances(lists_of_values, metric='cosine')

  plt.figure(figsize=plot_size)

  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")

  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]

  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)

  plt.xlabel("Index")
  plt.ylabel("Index")

  plt.tight_layout()

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()
  plt.close()

################################################################################

def plot_points_with_mst_lines(points, 
                               points_labels, 
                               points_mst_edges,
                               plot_size=(20, 20),
                               labels_size=24,
                               save_plot=''
                               ):

  """
  Plots 2D points with labels and MST lines.
  """

  plt.figure(figsize=plot_size)

  for i, label in enumerate(points_labels):
      plt.scatter(points[i][0], points[i][1])
      plt.annotate(label, (points[i][0], points[i][1]), fontsize=labels_size)

  for edge in points_mst_edges:
      i, j = edge
      plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'k-', alpha=0.5)

  plt.title('Points Map with MST Lines', fontsize=labels_size)
  plt.xlabel('X-axis', fontsize=labels_size)
  plt.ylabel('Y-axis', fontsize=labels_size)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()

################################################################################

def plot_points_constellation(points, 
                              points_labels,
                              p=5,                              
                              plot_size=(15, 15),
                              labels_size=12,
                              show_grid=False,
                              save_plot=''
                              ):

  """
  Plots 2D points constellation
  """

  points = np.array(points)

  dist_matrix = distance_matrix(points, points, p=p)

  mst = minimum_spanning_tree(dist_matrix).toarray()

  plt.figure(figsize=plot_size)

  plt.scatter(points[:, 0], points[:, 1], color='blue')

  for i, label in enumerate(points_labels):
      plt.annotate(label, (points[i, 0], points[i, 1]), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=labels_size
                   )

  for i in range(len(points)):
      for j in range(len(points)):
          if mst[i, j] > 0:
              plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k--')

  plt.xlabel('X-axis', fontsize=labels_size)
  plt.ylabel('Y-axis', fontsize=labels_size)
  plt.title('2D Coordinates with Minimum Spanning Tree', fontsize=labels_size)

  plt.grid(show_grid)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()

################################################################################

def binary_matrix_to_images(matrix, 
                            step,
                            overlap,
                            output_folder='./Dataset/', 
                            output_img_prefix='image', 
                            output_img_ext='.png',
                            save_to_array=False,
                            verbose=True
                            ):

    if not save_to_array:

      if verbose:
        print('=' * 70)
        print('Checking output folder dir...')

      os.makedirs(os.path.dirname(output_folder), exist_ok=True)

      if verbose:
        print('Done!')

    if verbose:
      print('=' * 70)
      print('Writing images...')

    matrix = np.array(matrix, dtype=np.uint8)
    
    image_array = []
    
    for i in range(0, max(1, matrix.shape[0]), overlap):
       
        submatrix = matrix[i:i+step, :]

        if submatrix.shape[0] < 128:
          zeros_array = np.zeros((128-submatrix.shape[0], 128))
          submatrix = np.vstack((submatrix, zeros_array))

        img = Image.fromarray(submatrix * 255).convert('1')
        
        if save_to_array:
          image_array.append(np.array(img))

        else:
          img.save(output_folder + output_img_prefix + '_' + str(matrix.shape[1]) + '_' + str(i).zfill(7) + output_img_ext)
  
    if verbose:
      print('Done!')
      print('=' * 70)
      print('Saved', (matrix.shape[0] // min(step, overlap))+1, 'imges!')
      print('=' * 70)

    if save_to_array:
        return np.array(image_array).tolist()

################################################################################

def images_to_binary_matrix(list_of_images):

    image_array = np.array(list_of_images)
   
    original_matrix = []
    
    for img in image_array:

        submatrix = np.array(img)
        original_matrix.extend(submatrix.tolist())
    
    return original_matrix

################################################################################

def square_image_matrix(image_matrix,
                        matrix_size=128,
                        num_pca_components=5,
                        filter_out_zero_rows=False,
                        return_square_matrix_points=False
                        ):

  """
  Reduces an arbitrary image matrix to a square image matrix
  """

  matrix = np.array(image_matrix)

  if filter_out_zero_rows:
    matrix = matrix[~np.all(matrix == 0, axis=1)]

  target_rows = matrix_size

  rows_per_group = matrix.shape[0] // target_rows

  compressed_matrix = np.zeros((target_rows, matrix.shape[1]), dtype=np.int32)

  for i in range(target_rows):
      start_row = i * rows_per_group
      end_row = (i + 1) * rows_per_group
      group = matrix[start_row:end_row, :]
      
      pca = PCA(n_components=num_pca_components)
      pca.fit(group)
      
      principal_component = np.mean(pca.components_, axis=0)
      contributions = np.dot(group, principal_component)
      selected_row_index = np.argmax(contributions)
      
      compressed_matrix[i, :] = group[selected_row_index, :]

  if return_square_matrix_points:
    filtered_matrix = compressed_matrix[~np.all(compressed_matrix == 0, axis=1)]

    row_indexes, col_indexes = np.where(filtered_matrix != 0)
    points = np.column_stack((row_indexes, filtered_matrix[row_indexes, col_indexes])).tolist()

    return points

  else:
    return compressed_matrix.tolist()

################################################################################

def image_matrix_to_images(image_matrix,
                           step,
                           overlap,
                           num_img_channels=3,
                           output_folder='./Dataset/',
                           output_img_prefix='image',
                           output_img_ext='.png',
                           save_to_array=False,
                           verbose=True
                           ):

    if num_img_channels > 1:
      n_mat_channels = 3

    else:
      n_mat_channels = 1

    if not save_to_array:

      if verbose:
        print('=' * 70)
        print('Checking output folder dir...')

      os.makedirs(os.path.dirname(output_folder), exist_ok=True)

      if verbose:
        print('Done!')

    if verbose:
      print('=' * 70)
      print('Writing images...')

    matrix = np.array(image_matrix)

    image_array = []

    for i in range(0, max(1, matrix.shape[0]), overlap):

        submatrix = matrix[i:i+step, :]

        if submatrix.shape[0] < 128:
          zeros_array = np.zeros((128-submatrix.shape[0], 128))
          submatrix = np.vstack((submatrix, zeros_array))

        if n_mat_channels == 3:

          r = (submatrix // (256*256)) % 256
          g = (submatrix // 256) % 256
          b = submatrix % 256

          rgb_image = np.stack((r, g, b), axis=-1).astype(np.uint8)
          img = Image.fromarray(rgb_image, 'RGB')

        else:
          grayscale_image = submatrix.astype(np.uint8)
          img = Image.fromarray(grayscale_image, 'L')

        if save_to_array:
          image_array.append(np.array(img))

        else:
          img.save(output_folder + output_img_prefix + '_' + str(matrix.shape[1]) + '_' + str(i).zfill(7) + output_img_ext)

    if verbose:
      print('Done!')
      print('=' * 70)
      print('Saved', (matrix.shape[0] // min(step, overlap))+1, 'imges!')
      print('=' * 70)

    if save_to_array:
        return np.array(image_array).tolist()

################################################################################

def images_to_image_matrix(list_of_images,
                           num_img_channels=3
                           ):

    if num_img_channels > 1:
      n_mat_channels = 3

    else:
      n_mat_channels = 1

    image_array = np.array(list_of_images)

    original_matrix = []

    for img in image_array:

      if num_img_channels == 3:

        rgb_array = np.array(img)

        matrix = (rgb_array[..., 0].astype(np.int64) * 256*256 +
                  rgb_array[..., 1].astype(np.int64) * 256 +
                  rgb_array[..., 2].astype(np.int64))

      else:
        matrix = np.array(img)

      original_matrix.extend(matrix)

    return original_matrix

################################################################################

def square_matrix_to_RGB_matrix(square_matrix):

  smatrix = np.array(square_matrix)
  sq_matrix = smatrix[:smatrix.shape[1]]

  r = (sq_matrix // (256 ** 2)) % 256
  g = (sq_matrix // 256) % 256
  b = sq_matrix % 256

  rgb_array = np.stack((r, g, b), axis=-1)

  return rgb_array.tolist()

################################################################################

def upsample_square_matrix(square_matrix, upsampling_factor=4):

  smatrix = np.array(square_matrix)
  sq_matrix = smatrix[:smatrix.shape[1]]

  scaling_array = np.ones((upsampling_factor, upsampling_factor))
  scaled_array = np.kron(sq_matrix, scaling_array)
  scaled_array = scaled_array.astype('int')

  return scaled_array.tolist()

################################################################################

def downsample_square_matrix(square_matrix, downsampling_factor=4):

  smatrix = np.array(square_matrix)
  sq_matrix = smatrix[:smatrix.shape[1]]

  dmatrix = sq_matrix[::downsampling_factor, ::downsampling_factor]
  dmatrix = dmatrix.astype('int')

  return dmatrix.tolist()

################################################################################

def plot_parsons_code(parsons_code, 
                      start_pitch=60, 
                      return_plot_dict=False, 
                      return_plot_string=False,
                      plot_size=(10, 10),
                      labels_size=16,
                      save_plot=''
                      ):
  
  '''
  Plot parsons code string
  '''

  if parsons_code[0] != "*":
      return None

  contour_dict = {}
  pitch = 0
  index = 0

  maxp = 0
  minp = 0

  contour_dict[(pitch, index)] = "*"

  for point in parsons_code:
      if point == "R":
          index += 1
          contour_dict[(pitch, index)] = "-"

          index += 1
          contour_dict[(pitch, index)] = "*"
          
      elif point == "U":
          index += 1
          pitch -= 1
          contour_dict[(pitch, index)] = "/"

          index += 1
          pitch -= 1
          contour_dict[(pitch, index)] = "*"

          if pitch < maxp:
              maxp = pitch

      elif point == "D":
          index += 1
          pitch += 1
          contour_dict[(pitch, index)] = "\\"

          index += 1
          pitch += 1
          contour_dict[(pitch, index)] = "*"

          if pitch > minp:
              minp = pitch

  if return_plot_dict:
    return contour_dict
  
  if return_plot_string:

    plot_string = ''

    for pitch in range(maxp, minp+1):
        line = [" " for _ in range(index + 1)]
        for pos in range(index + 1):
            if (pitch, pos) in contour_dict:
                line[pos] = contour_dict[(pitch, pos)]

        plot_string = "".join(line)

    return plot_string

  labels = []
  pitches = []
  positions = []
  cur_pitch = start_pitch
  pitch_idx = 0

  for k, v in contour_dict.items():

    if v != '*':

      pitches.append(cur_pitch)
      positions.append(pitch_idx)

      if v == '/':
        cur_pitch += 1
        labels.append('U')
      
      elif v == '\\':
        cur_pitch -= 1
        labels.append('D')

      elif v == '-':
        labels.append('R')

      pitch_idx += 1

  plt.figure(figsize=plot_size)

  
  plt.plot(pitches)

  for i, point in enumerate(zip(positions, pitches)):
    plt.annotate(labels[i], point, fontsize=labels_size)
  

  plt.title('Parsons Code with Labels', fontsize=labels_size)
  plt.xlabel('Position', fontsize=labels_size)
  plt.ylabel('Pitch', fontsize=labels_size)

  if save_plot != '':
    plt.savefig(save_plot, bbox_inches="tight")
    plt.close()

  plt.show()

  plt.close()
  
################################################################################

def plot_tokens_embeddings_constellation(tokens_embeddings,
                                         start_token,
                                         end_token,
                                         plot_size=(10, 10),
                                         labels_size=12,
                                         show_grid=False,
                                         save_plot=''):
    
    """
    Plots token embeddings constellation using MST and graph layout
    without dimensionality reduction.
    """

    distance_matrix = metrics.pairwise_distances(tokens_embeddings[start_token:end_token], metric='cosine')
    
    token_labels = [str(i) for i in range(start_token, end_token)]

    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    n = distance_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                weight = 1 / (distance_matrix[i, j] + 1e-8)
                G.add_edge(i, j, weight=weight)
    
    pos = nx.kamada_kawai_layout(G, weight='weight')
    
    points = np.array([pos[i] for i in range(n)])
    
    plt.figure(figsize=plot_size)
    plt.scatter(points[:, 0], points[:, 1], color='blue')
    
    for i, label in enumerate(token_labels):
        plt.annotate(label, (points[i, 0], points[i, 1]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=labels_size)
    
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                plt.plot([points[i, 0], points[j, 0]],
                         [points[i, 1], points[j, 1]],
                         'k--', alpha=0.5)
    
    plt.title('Token Embeddings Constellation with MST', fontsize=labels_size)
    plt.grid(show_grid)
    
    if save_plot:
        plt.savefig(save_plot, bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()
        
    plt.close()
    
################################################################################

def find_token_path(tokens_embeddings, 
                    start_token, 
                    end_token, 
                    verbose=False
                   ):
    
    """
    Finds the path of tokens between start_token and end_token using
    the Minimum Spanning Tree (MST) derived from the distance matrix.
    """

    distance_matrix = metrics.pairwise_distances(tokens_embeddings, metric='cosine')
    
    token_labels = [str(i) for i in range(len(distance_matrix))]
    
    if verbose:
        print('Total number of tokens:', len(distance_matrix))
    
    mst = minimum_spanning_tree(distance_matrix).toarray()
    
    n = distance_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                weight = 1 / (distance_matrix[i, j] + 1e-8)
                G.add_edge(i, j, weight=weight)
    
    try:
        start_idx = token_labels.index(str(start_token))
        end_idx = token_labels.index(str(end_token))
        
    except ValueError:
        raise ValueError("Start or end token not found in the provided token labels.")
    
    path_indices = nx.shortest_path(G, source=start_idx, target=end_idx)
    
    token_path = [int(token_labels[idx]) for idx in path_indices]
    
    return token_path

################################################################################
# [WIP] Future dev functions
################################################################################

'''
import umap

def reduce_dimensionality_umap(list_of_values,
                               n_comp=2,
                               n_neighbors=15,
                               ):

  """
  Reduces the dimensionality of the values using UMAP.
  """

  vals = np.array(list_of_values)

  umap_reducer = umap.UMAP(n_components=n_comp,
                           n_neighbors=n_neighbors,
                           n_epochs=5000,
                           verbose=True
                           )

  reduced_vals = umap_reducer.fit_transform(vals)

  return reduced_vals.tolist()
'''

################################################################################

'''
import alphashape
from shapely.geometry import Point
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.stats import zscore

#===============================================================================

coordinates = points

dist_matrix = minkowski_distance_matrix(coordinates, p=3)  # You can change the value of p as needed

# Centering matrix
n = dist_matrix.shape[0]
H = np.eye(n) - np.ones((n, n)) / n

# Apply double centering
B = -0.5 * H @ dist_matrix**2 @ H

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(B)

# Sort eigenvalues and eigenvectors
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Select the top 2 eigenvectors
X_transformed = eigvecs[:, :2] * np.sqrt(eigvals[:2])

#===============================================================================

src_points = X_transformed
src_values = np.array([[p[1]] for p in points]) #np.random.rand(X_transformed.shape[0])

#===============================================================================

# Normalize the points to the range [0, 1]
scaler = MinMaxScaler()
points_normalized = scaler.fit_transform(src_points)

values_normalized = custom_normalize(src_values)

# Remove outliers based on z-score
z_scores = np.abs(zscore(points_normalized, axis=0))
filtered_points = points_normalized[(z_scores < 3).all(axis=1)]
filtered_values = values_normalized[(z_scores < 3).all(axis=1)]

# Compute the concave hull (alpha shape)
alpha = 8  # Adjust alpha as needed
hull = alphashape.alphashape(filtered_points, alpha)

# Create a triangulation
tri = Triangulation(filtered_points[:, 0], filtered_points[:, 1])

# Interpolate the values on the triangulation
interpolator = LinearTriInterpolator(tri, filtered_values[:, 0])
xi, yi = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
zi = interpolator(xi, yi)

# Mask out points outside the concave hull
mask = np.array([hull.contains(Point(x, y)) for x, y in zip(xi.flatten(), yi.flatten())])
zi = np.ma.array(zi, mask=~mask.reshape(zi.shape))

# Plot the filled contour based on the interpolated values
plt.contourf(xi, yi, zi, levels=50, cmap='viridis')

# Plot the original points
#plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c=filtered_values, edgecolors='k')

plt.title('Filled Contour Plot with Original Values')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Value')
plt.show()
'''

################################################################################

def plot_tree_horizontal(data):
    
    """
    Given data as a list of levels (each level is a tuple or list of 
    displacements for each branch), this function computes the cumulative 
    value per branch (starting from 0) and plots each branch
    with the tree level mapped to the x-axis and the cumulative value mapped 
    to the y-axis. This gives a left-to-right tree with branches spanning up 
    (positive) and down (negative).
    
    Parameters:
        data (list of tuple/list): Each element represents a tree level.
                                   It is assumed every level has the same length.
    """
    
    # Convert data to a NumPy array with shape (n_levels, n_branches)
    data = np.array(data)
    n_levels, n_branches = data.shape

    # Compute cumulative sums along each branch.
    # Each branch starts at 0 at level 0.
    cum = np.zeros((n_levels + 1, n_branches))
    for i in range(n_levels):
        cum[i + 1, :] = cum[i, :] + data[i, :]
    
    plt.figure(figsize=(12, 8))
    
    # Plot each branch as a line. For branch j:
    #   - x coordinates are the tree levels (0 to n_levels)
    #   - y coordinates are the corresponding cumulative values.
    for j in range(n_branches):
        x = np.arange(n_levels + 1)
        y = cum[:, j]
        plt.plot(x, y, marker='o', label=f'Branch {j}')
    
    plt.title("Horizontal Tree Visualization: Branches Spanning Up and Down", fontsize=14)
    plt.xlabel("Tree Level (Left = Root)")
    plt.ylabel("Cumulative Value (Up = Positive, Down = Negative)")
    
    # Add a horizontal line at y=0 to emphasize the center.
    plt.axhline(0, color="gray", linestyle="--")
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
################################################################################
# This is the end of TPLOTS Python modules
################################################################################