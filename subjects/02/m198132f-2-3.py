# (a) sentiment.txtの全てのレビュー文に対して、講義資料第6回の9ページにあるBag of Words（出現回数）を用いて、ベクトル表現に変換する。（言語処理100本ノック2015版の問題72に相当）
# (b) 全レビュー文を学習データとして用いて、ロジスティック回帰モデルを学習する。（言語処理100本ノック2015版の問題73に相当）
# (c) 学習データに対して、学習されたロジスティック回帰モデルを適用して、予測の正解率、正例に関する適合率、再現率、F1スコアを求めて、画面上に出力する。（言語処理100本ノック2015版の問題76、77に相当）

# ベクトル化のためのCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Naive_bayesモデル
from sklearn.naive_bayes import MultinomialNB

# 適合率，再現率，F1スコアを表示する
from sklearn.metrics import classification_report


# 結果の出力を整形する関数
def print_report_by_label(report, lable):
    pos_label = report[lable]['precision']
    print('{}ラベルのスコア: precision: {}, recall: {}, f1-socre: {}'.format(
        lable, report[lable]['precision'], report[lable]['recall'], report[lable]['f1-score']
    ))


def print_overroll_score(report):
    print('全体のスコア: accuracy: {}, f1-score(macro avg): {}, f1-score(weighted avg): {}'.format(
        report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']
    ))


# 読み込むファイルのパス
path = './data/'
file_name = 'sentiment.txt'

# 目的変数
y = []
# 説明変数
X = []
with open(path + file_name) as f:
    for line in f:
        text = line.split(' ')
        y.append(text[0])
        X.append(' '.join(text[1:]))

stop_words = [',', '.', '\n']

# sklearnのCountVectorizerを使ってBoW化する
vectorizer = CountVectorizer(stop_words=stop_words)
vectorizer.fit(X)
# BoWによるベクトル化
X_bow = vectorizer.transform(X)

# Naive_bayesモデルの構築
mnb = MultinomialNB(alpha=1)
mnb.fit(X_bow, y)

# 推論
y_pred = mnb.predict(X_bow)

# 結果の計算
report = classification_report(y, y_pred, output_dict=True)

# 出力
print_report_by_label(report, '+1')
print_report_by_label(report, '-1')
print_overroll_score(report)
