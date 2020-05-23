import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .") # .の前に半角を挿入
    words = text.split(" ") # 半角スペースにより単語を分割

    word_to_id = {}
    id_to_word = {}
    # 単語に対応するidを付与
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

"""
text = 'This is a sample text for this fanction.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
"""

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
	nx = x / np.sqrt(np.sum(x**2) + eps)
	ny = y / np.sqrt(np.sum(y**2) + eps)
	return np.dot(nx, ny)

def most_similarity(query, word_to_id, id_to_word, word_matrix, top=5):
    # 1-クエリを取り出す
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 2-コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 3-コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return