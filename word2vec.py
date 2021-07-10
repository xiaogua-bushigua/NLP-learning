import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as Data

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本预处理
sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
sentences_list = " ".join(sentences).split()
# ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'jack', 'like', 'animal',....]
vocab = list(set(sentences_list))  # 构建无重复单词表
vocab_size = len(vocab) # 13
# ['cat', 'banana', 'book', 'like', 'animal', 'hate', 'music', 'movie', 'jack', ...]
word2idx = {w: i for i, w in enumerate(vocab)} # 构建单词和索引位置的字典
# {'banana': 0, 'fish': 1, 'book': 2, 'cat': 3, 'animal': 4, 'music': 5, 'hate': 6,
# 'dog': 7, 'movie': 8, 'milk': 9, 'jack': 10, 'like': 11, 'apple': 12}

# 模型参数
C = 2 # 滑动窗口的大小（一般长度）
batch_size = 8
m = 3 # 嵌入词矩阵维度

# 数据预处理
# 1.
skip_grams = []
for idx in range(C, len(sentences_list) - C):
  center = word2idx[sentences_list[idx]]  # 在原始文本中找到每一个单词作为中心词在单词表中的索引
  context_idx = list(range(idx-C, idx)) + list(range(idx+1, idx+1+C)) # 背景词在原始文本中的索引
  context = [word2idx[sentences_list[i]]  for i in context_idx]

  for w in context:
    skip_grams.append([center, w])

# 2.
def make_data(skip_grams):
  input_data = []
  output_data = []
  for i in range(len(skip_grams)):
    input_data.append(np.eye(vocab_size)[skip_grams[i][0]])
    output_data.append(skip_grams[i][1])
  return input_data, output_data

# 3.
input_data, output_data = make_data(skip_grams)
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)

# 训练模型
# 1.
class word2vec(nn.Module):
  def __init__(self):
    super(word2vec, self).__init__()
    self.W = nn.Parameter(torch.randn(vocab_size, m).type(dtype))
    self.V = nn.Parameter(torch.randn(m, vocab_size).type(dtype))

  def forward(self, X):
    # X [batch_size, vocab_size]
    hidden = torch.mm(X, self.W)  # [batch_size, m]
    output = torch.mm(hidden, self.V) # [batch_size, vocab_size] 分类问题
    return output

# 2.
model = word2vec().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = optim.Adam(model.parameters(), lr=1e-3)

# 3.
for epoch in range(2000):
  for i, (batch_x, batch_y) in enumerate(loader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)
    if (epoch + 1) % 100 == 0:
      print(epoch + 1, i, loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

for i, label in enumerate(vocab):
  W, WT = model.parameters()
  x,y = float(W[i][0]), float(W[i][1])
  plt.scatter(x, y)
  plt.annotate(label, xy=(x, y), xytext=(5, 2),
               textcoords='offset points', ha='right', va='bottom')
plt.show()
