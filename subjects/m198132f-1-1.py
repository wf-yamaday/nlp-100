import pprint

# 読み込むファイルのパス
path = './data/'
file_name = 'neko.txt.mecab'

# ファイルの読み込み
data = ''
with open(path + file_name) as f:
  data = f.read()

# 読み込み
text_list = []
for line in data.split('\n'):
  # タブをカンマに変換
  line = line.replace('\t', ',')
  # カンマ区切りのリスト
  line_list = line.split(',')

  if line_list[0] != 'EOS':
    morpheme_dic = {}
    morpheme_dic['surface'] = line_list[0]
    morpheme_dic['base'] = line_list[7]
    morpheme_dic['pos'] = line_list[1]
    morpheme_dic['pos1'] = line_list[2]
    text_list.append(morpheme_dic)

data = []
for text in text_list:
  data.append(text['surface'])

results = []
for word in set(data):
  result = {}
  result['word'] = word
  result['count'] =  data.count(word)
  results.append(result)

# 出現回数が多い順にソート
results.sort(key=lambda x: x['count'], reverse=True)
pprint.pprint(results)
