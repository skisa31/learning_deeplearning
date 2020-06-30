import sys
sys.path.append('.')
from common.util import most_similarity
import pickle

pkl_file = 'cbow_param.pkl'

with open(pkl_file, 'rb') as f:
  params = pickle.load(f)
  word_vecs = params['word_vecs']
  word_to_id = params['word_to_id']
  id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
  most_similarity(query, word_to_id, id_to_word, word_vecs, top=5)