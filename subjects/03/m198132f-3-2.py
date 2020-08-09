import torch
from torch import nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support


# データセットの読み込み
# 訓練データ
x_train = np.loadtxt('NewsDataSet/x_train.txt')
y_train = np.loadtxt('NewsDataSet/y_train.txt')
x_train = torch.from_numpy(x_train.astype(np.float32)).clone()
y_train = torch.from_numpy(y_train.astype(np.float32)).clone()
y_train = y_train.type(torch.LongTensor)

# 検証データ
x_val = np.loadtxt('NewsDataSet/x_valid.txt')
y_val = np.loadtxt('NewsDataSet/y_valid.txt')
x_val = torch.from_numpy(x_val.astype(np.float32)).clone()
y_val = torch.from_numpy(y_val.astype(np.float32)).clone()
y_val = y_val.type(torch.LongTensor)

# 評価データ
x_test = np.loadtxt('NewsDataSet/x_test.txt')
y_test = np.loadtxt('NewsDataSet/y_test.txt')
x_test = torch.from_numpy(x_test.astype(np.float32)).clone()
y_test = torch.from_numpy(y_test.astype(np.float32)).clone()
y_test = y_test.type(torch.LongTensor)

train = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train), batch_size=1, shuffle=True
)

val = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_val, y_val), batch_size=1, shuffle=False
)

test = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False
)


# ニューラルネットワークの定義
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

dl_dict = {"train": train, "val": val}

# 学習フェーズ
epoch_num = 10
for t in range(epoch_num):
    # epoch毎に学習と検証を繰り返す
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        epoch_corrects = 0

        for i, (x, y) in enumerate(dl_dict[phase]):
            optimizer.zero_grad()
            # ラベルを予測
            y_pred = model(x)
            # 損失を計算
            loss = loss_func(y_pred, y)
            _, y_pred = torch.max(y_pred, 1)

            # 訓練時は重みを更新
            if phase == 'train':
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_corrects += torch.sum(y_pred == y)

        # epochごとのlossと正解率
        epoch_loss = epoch_loss / len(dl_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dl_dict[phase].dataset)
        print('Epoch: {}, phase: {}, loss = {}, acc = {}'.format(
            t+1, phase, epoch_loss, epoch_acc))

# 評価フェーズ
# 結果保存用のデータフレーム
results = pd.DataFrame(columns=['label', 'pred'])
corrects = 0
for x, y in test:
    y_pred = model(x)
    _, y_pred = torch.max(y_pred, 1)
    corrects += torch.sum(y_pred == y)
    # 結果をデータフレーム化
    result = pd.DataFrame(columns=['label', 'pred'])
    result['label'] = y.data.tolist()
    result['pred'] = y_pred.tolist()
    # 全体のデータフレームに追加
    results = results.append(result, ignore_index=True)

# 正解率を計算
test_acc = corrects.double() / len(test.dataset)
print('評価データでの正解率: {}'.format(test_acc))

# 結果の集計
resutls = results.astype('int64')
y_test_true = list(results['label'].values)
y_test_pred = list(results['pred'].values)

# 各カテゴリ毎のprecision, recall, f-scoreを求める
scores = precision_recall_fscore_support(
    y_test_true, y_test_pred, average=None)

dic = {
    0: 'precision',
    1: 'recall',
    2: 'fscore',
    3: 'sample'
}

# カテゴリごとにDataFrameにまとめる
details = pd.DataFrame(columns=['precision', 'recall', 'fscore', 'sample'])
for i, score in enumerate(scores):
    details[dic[i]] = score

# indexにカテゴリ名をつける
details.index = ['ビジネス', '科学技術', 'エンターテイメント', '健康']
print(details)
