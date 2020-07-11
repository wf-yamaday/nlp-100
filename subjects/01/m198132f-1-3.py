import MeCab
import math

# 単語bigramを求める関数(第1章から)
def to_n_gram(text, n=2):
  return [ text[i:i + n] for i in range(len(text) - n + 1)]

# 読み込むファイルのパス
path = './data/'
file_name = 'neko.txt'

# MeCabを使って分かち書きをするためにインスタンスを生成
mecab = MeCab.Tagger('-Owakati')

# ファイルの読み込み
data = []
word_dic = {}
with open(path + file_name) as f:
  for line in f:
      line = line.replace('\u3000', '')
      line = line.replace('\n', '')
      line = line.replace('…', '')
      line = line.replace('一', '')
      if len(line) != 0:
        line = mecab.parse(line)
        line = line.replace('\n', '')
        line = '<BOS> {}<EOS>'.format(line)
        data.append(line)
        for word in line.split(' '):
          if word not in word_dic.keys():
            word_dic[word] = 1
          else:
            word_dic[word] += 1

# bigramを作成
bigrams = {}
results = {}
for line in data:
  text = to_n_gram(line.split(' '))
  for t in text:
    key = str(t).replace('[', '').replace(']','').replace("'",'')
    if key not in bigrams.keys():
      bigrams[key] = 1
    else:
      bigrams[key] += 1
    results[key] = bigrams[key]/word_dic[t[0]]

text = '吾輩は猫ではない。名前もまだない。'
text = text.replace('。', '。\n')
texts = text.split('\n')
texts.remove('')

word_probability = []
for t in texts:
  t = mecab.parse(t)
  t = t.replace('\n', '')
  t = '<BOS> {}<EOS>'.format(t)
  t_bigrams = to_n_gram(t.split(' '))
  for bigram in t_bigrams:
    key = str(bigram).replace('[', '').replace(']','').replace("'",'')
    word_probability.append(results[key])

result = 1
for prob in word_probability:
  result *= prob

print(result**(1/len(word_probability)))
