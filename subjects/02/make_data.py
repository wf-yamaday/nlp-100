import random

prefix = './data/'
pos_file = 'rt-polarity.pos'
neg_file = 'rt-polarity.neg'

f = open(prefix + pos_file, 'rb')
data = []
line = f.readline()
while line:
    line = line.decode(errors='replace')
    line = '+1 ' + line
    data.append(line)
    line = f.readline()
f.close()

f = open(prefix + neg_file, 'rb')
line = f.readline()
while line:
    line = line.decode(errors='replace')
    line = '-1 ' + line
    data.append(line)
    line = f.readline()
f.close()

data = random.sample(data, len(data))

data_str = ''.join(data)
with open('sentiment.txt', 'wt') as f:
    f.write(data_str)
