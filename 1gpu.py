'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''#改成gpu版本https://blog.csdn.net/qq_28444159/article/details/78781201#这个文件可以直接跑.用的是gpu
import warnings
import torch

import torch
print(torch.__version__)
print(torch.cuda.is_available())
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps  占位符
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
#P 就是padding 表示空白, S表示start E 表示End
# Transformer Parameters
# Padding Should be Zero index
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'S' : 5, 'E' : 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 5


d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]#
    '''
    关于input_batch ,output_bathc ,traagtet _batch 怎么理解?
    怎么弄了3个???
    '''


    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return Variable(torch.LongTensor(input_batch)), Variable(torch.LongTensor(output_batch)), Variable(torch.LongTensor(target_batch))

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):#hid_idx表示hidden层当前跑到的dex
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token,表示等于0的都进行mask(mask的操作就是矩阵放一个1),非0的才进行学习
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking         ,
    # .byte()把bool值变成1,0矩阵
    pad_attn_mask = pad_attn_mask.byte()#expand 进行复制的函数.根据shape自动识别复制的方向.
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()#就是返回一个上三角矩阵
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is
        # one.#把评分进行mask修正,

        # 修改为负无穷,softmax之后得到的attention是0,表示不对这个位置进行学习.不关心这个位置.attn:1,8,5,5
        attn = nn.Softmax(dim=-1)(scores)#高维矩阵乘法,会自动看做最后的2维组成的矩阵来做乘法.其他的作为数量看.
        context = torch.matmul(attn, V)#attn 和原始矩阵V乘法就表示注意力使用了.
        return context, attn#理解上面attn,因为是左乘一个下三角矩阵,用三种矩阵基本变换来理解就行了,直接就等价于若干个把当前行*一个系数加到下面的行上.所以从V这个矩阵看,时间轴上的变化是从以前影响到将来的是符合要求的.V的行数越大代表时间越靠后.这个为什么表示时间呢,因为跟V的输入有关.

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x
        # d_v] #     batch *头数量*步长* 深度  从这里面就看出来v的第三个分量表示的是时间轴!!!!!解释了attn为什么需要一个下三角剧zhen

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model).cuda()(context)#contiguous()是旧版pytorch写法,为了后面使用.view  https://blog.csdn.net/appleml/article/details/80143212
        return nn.LayerNorm(d_model).cuda()(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()#conv1d和dnn一样.
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model).cuda()(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),
                                                    freeze=True)#nn.Embedding.from_pretrained会设置参数freeze=True,
        # 这样模型里面的参数就不会学习了.当成固定参数.这个位置确实不是学习到的,是人为设计的.
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)#get_attn_pad_mask 这个函数很重要.
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)
        dec_outputs=dec_outputs.cuda()
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).cuda()#只有这个地方跟encoder不一样.
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask).cuda(), 0).cuda()#表示可以遮挡的地方都遮挡上.
#get_attn_pad_mask 这个函数让编码为0的不起作用.get_attn_subsequent_mask 让后续的不起作用.因为inference时候只能从左到有推断.
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns#输出的结果:dec_logits.view(-1, dec_logits.size(-1))表示的是一排数字,表示预测后得到的翻译结果的index码.

def greedy_decoder(model, enc_input, start_symbol):
    """



    #在训练时候是有会提高准确率.Beam search
    https://blog.csdn.net/qq_16234613/article/details/83012046
    在推断的时候会增加准确率,原理是加的召回率.

    greedy就是每一步都找概率最大的作为输出.




    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    next_symbol = start_symbol#从S开始赋值
    for i in range(0, 5):#一个句子有几个单词就写几
        dec_input[0][i] = next_symbol #比如先从5,0,0,0,0开始预测.那么就把5,0,0,0,0当做已知输入进去. 不停的放进去.
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)#得到第一个单词的输出 dec_outputs
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1] #prob表示预测后文字的index
        next_word = prob.data[i]  #取出来.
        next_symbol = next_word.item()# .item函数把tensor变成int类型
    return dec_input

def showgraph(attn):
    pass


model = Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    ###########----------建立自己数据--------------##########################
    '''
    下面用自己的数据:
    '''
    import torch

    import torch.utils.data as Data
    with open ('chineseEnglishDataCoded.bpe') as file:
        tmp=file.readlines()

    BATCH_SIZE = 5  # 批训练的数据个数，每组五个

    x = torch.zeros(10, 15)  # x data (torch tensor)

    y = torch.ones(10, 15)  # y data (torch tensor)

    # 先转换成 torch 能识别的 Dataset

    torch_dataset = Data.TensorDataset(x, y)

    # 把 dataset 放入 DataLoader

    loader = Data.DataLoader(

        dataset=torch_dataset,  # torch TensorDataset format

        batch_size=BATCH_SIZE,  # 每组的大小

        shuffle=False,  # 要不要打乱数据 (打乱比较好)

    )

    for epoch in range(3):  # 对整套数据训练三次，每次训练的顺序可以不同

        for step, (x, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习

            # 假设这里就是你训练的地方...

            # 打出来一些数据

            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',

                  x.numpy(), '| batch y: ', y.numpy())









    ###########----------数据准备完毕--------------##########################
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    model=model.cuda()
    enc_inputs=enc_inputs.cuda()
    dec_inputs=dec_inputs.cuda()
    target_batch=target_batch.cuda()



    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs, target_batch.contiguous().view(-1))#output.shape (5,7),一行代表一个单词的概率分布.所以总的说就是算cross entropy即可.也就是说cross entropy里面的groud true直接填写标签tensor就可以了.更方便.
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Test
greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])#再来看test时候如何使用模型来做预测翻译.  greedy_decoder生成预测开始的编码
predict, _, _, _ = model(enc_inputs, greedy_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])
