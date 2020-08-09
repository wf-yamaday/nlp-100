import gensim
import scipy
import numpy as np
import pandas as pd

# word2vecモデルの読み込み
file_name = 'Vec_BCCWJ_w2v_all_win10_dim300_skipgram_ns5.txt.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(
    file_name, binary=False, unicode_errors='ignore')


# コサイン類似度
def calc_cos_similer(row):
    word_1 = model[row['word1']]
    word_2 = model[row['word2']]
    return np.dot(word_1, word_2) / (np.linalg.norm(word_1) * np.linalg.norm(word_2))


# 評価用のcsvファイルの読み込み
csv_file = 'jwsan-1400.csv'
df = pd.read_csv(csv_file)

# コサイン類似度の計算
df['cos_simiray'] = df.apply(calc_cos_similer, axis=1)

# スピアマン相関係数の計算
correlation_similarity, _ = scipy.stats.spearmanr(
    df['cos_simiray'], df['similarity'])
correlation_association, _ = scipy.stats.spearmanr(
    df['cos_simiray'], df['association'])

print('類似度との相関係数 = {}'.format(correlation_similarity))
print('関連度との相関係数 = {}'.format(correlation_association))
