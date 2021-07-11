import pandas as pd
from torchtext.legacy import data
import torch
from torch import optim
import torch.nn as nn

pd.set_option('display.max_columns', None)

###################################################
# 原始数据文本处理

dataset = pd.read_csv("training.1600000.processed.noemoticon.csv",
                      engine="python", header=None)
# print(dataset.info)
# 将标签转换成可分类变量
dataset['sentiment_category'] = dataset[0].astype('category')
# print(dataset.info)
# 把0、4标签转换成0、1标签
dataset['category'] = dataset['sentiment_category'].cat.codes
# print(dataset.info)
dataset.to_csv('training_processed.csv', header=None, index=None)

###################################################
# 划分数据集

# 两个Field对象定义字段的处理方法（文本字段、标签字段）
label = data.LabelField() # 标签
tweet = data.Field(lower=True) # 内容，都转变成小写
# 表头
fields = [('score', None), ('id', None), ('date', None), ('query', None),
          ('name', None), ('tweet', tweet), ('category', None), ('label', label)]
# 读取数据
twitterDataset = data.TabularDataset(
    path='training_processed.csv',
    format='CSV',
    fields=fields,
    skip_header=False # 不跳过表头
)
# 分离 train, test, val
train,test,val = twitterDataset.split(split_ratio=[0.8,0.1,0.1],strata_field='label')
# print(vars(train[5643])) # 查看其中一个样本

###################################################
# 构建词汇表和文本批处理

vocab_size = 20000 # 常见单词
tweet.build_vocab(train, max_size=vocab_size)
label.build_vocab(train)
# print(len(tweet.vocab)) # 20002, 多出来的两个单词是unk和pad，表示未知和填充单词

# 文本批处理, 分割的同时做包装，一批一批的读取数据
device = "cuda" if torch.cuda.is_available() else "cpu"
train_iter,val_iter,test_iter = data.BucketIterator.splits((train,val,test),
                                                             batch_size=32,
                                                             device=device,
                                                             sort_within_batch=True,
                                                             sort_key=
                                                             lambda x:len(x.tweet))
# sort_within_batch=True 一个batch内的数据就会按照sort_key的规则降序排列
# sort_key=lambda x:len(x.tweet)    使用tweet的长度，即包含的单词的数量

###################################################
# 模型构建
class simple_LSTM(nn.Module): # 继承nn.Module
    def __init__(self, hidden_size, embedding_dim, vocal_size):
        super(simple_LSTM, self).__init__() # 调用父类的构造方法
        self.embedding = nn.Embedding(vocal_size, embedding_dim) # 输入
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                               num_layers=1)  # 中间层
        self.predictor = nn.Linear(hidden_size, 2) # 输出层，二分类

    def forward(self, seq):
        output, (hidden, cell) = self.encoder(self.embedding(seq))
        # output    torch.size([24, 32, 100])
        # hidden    torch.size([1, 32, 100])
        # cell    torch.size([1, 32, 100])
        # 24:一条评论多少单词，32:batch_size，100:hidden_size
        preds = self.predictor(hidden.squeeze(0))
        # 不需要hidden中的“1”维度
        return preds
# 创建对象
lstm_model = simple_LSTM(100, 300, vocab_size+2)
lstm_model.to(device)
# 优化器
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
# 损失函数
criterion = nn.CrossEntropyLoss()

###################################################
# 输出准确率
def train_val_test(model,optimizer,criterion,train_iter,val_iter,test_iter,epochs):
    for epoch in range(1, epochs+1):
        train_loss = 0
        val_loss = 0
        model.train() # 声明开始训练
        for indices, batch in enumerate(train_iter):
            optimizer.zero_grad() # 梯度置0
            outputs = model(batch.tweet) # [batch_size, 2]
            loss = criterion(outputs, batch.label)
            loss.backward() # 反向传播
            optimizer.step() # 参数更新
            train_loss += loss.data.item()*batch.tweet.size(0) # 累计每一批的损失值
        train_loss /= len(train_iter) # 计算平均损失
        print("Epoch: {}, Train loss: {:.2f}".format(epoch, train_loss))

        # 声明验证
        model.eval()
        for indices, batch in enumerate(val_iter):
            context = batch.tweet.to(device)
            target = batch.label.to(device)
            pred = model(context)
            loss = criterion(pred, target)
            val_loss += loss.item()*context.size(0) # 累计每一批的损失值
        val_loss /= len(val_iter) # 计算平均损失
        print("Epoch: {}, Val loss: {:.2f}".format(epoch, val_loss))

        # 声明测试
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad(): # 测试时，无需进行梯度计算
            for idx, batch in enumerate(test_iter):
                context = batch.tweet.to(device)
                target = batch.label.to(device)
                outputs = model(context)
                loss = criterion(outputs, target)
                test_loss += loss.item()*context.size(0) # 累计每一批的损失值
                # 获取最大预测值的索引
                preds = outputs.argmax(1)
                correct += preds.eq(target.view_as(preds)).sum().item()
            test_loss /= len(test_iter) # 计算平均损失
        print("Epoch: {}, Test loss: {:.2f}".format(epoch, test_loss))
        print("Accuracy: {}".format(100*correct/(len(test_iter)*batch.tweet.size(1))))

###################################################
# 开始训练
train_val_test(lstm_model, optimizer, criterion,
               train_iter, val_iter, test_iter, epochs=2)