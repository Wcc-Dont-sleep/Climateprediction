import torch
import torch.nn as nn
"""
input_dim：输入特征维数
hidden_dim：隐藏层状态的维数（隐藏层节点的个数）
kernel_size：卷积核尺寸
num_layers：层，理解为网络深度
batch_first：控制t（时间步长）放在首维还是第二维，官方不推荐我们把batch放在第一维
bias：偏置
return_all_layers：是否返回全部层
"""
class ConvLSTMCell(nn.Module):
    # 这里面全都是数，衡量后面输入数据的维度/通道尺寸
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 卷积核为一个数组
        self.kernel_size = kernel_size
        # 填充为高和宽分别填充的尺寸
        self.padding_size = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.input_dim + self.hidden_dim,
                              4 * self.hidden_dim,  # 4* 是因为后面输出时要切4片
                              self.kernel_size,
                              padding=self.padding_size,
                              bias=self.bias)


    #cur_state:状态c
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        combined_conv = self.conv(combined)
        #将输入的combined_conv在第dim维拆分成self.hidden_dim块
        #根据隐藏层的层数来进行拆分
        cc_f, cc_i, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # torch.sigmoid(),激活函数--
        # nn.functional中的函数仅仅定义了一些具体的基本操作，
        # 不能构成PyTorch中的一个layer
        # torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)

        #卷积后的结果输入第一个sigmoid激活层，f代表了forget gate输出的结果
        f = torch.sigmoid(cc_f)
        #输入的第二个sigmoid激活层，计算的是input gate的结果，要与g进行相乘
        i = torch.sigmoid(cc_i)
        #第三个sigmoid激活层是output gate结果
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 这里的乘是矩阵对应元素相乘，哈达玛乘积
        #下一个cell status是c_next
        c_next = f * c_cur + i * g
        #下一个隐藏层的结果是h_next
        h_next = i * torch.tanh(c_next)



        return h_next,c_next
    #初始化隐藏层
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # 返回两个是因为cell的尺寸与h一样
        #五维数据返回
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=False,
                 return_all_layers=False):
        super(ConvLstm, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # 为了储存每一层的参数尺寸
        cell_list = []
        for i in range(0, num_layers):
            #注意这里利用lstm单元得出到了输出h，h再作为下一层的输入，依次得到每一层的数据维度并储存
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))
        # 将上面循环得到的每一层的参数尺寸/维度，储存在self.cell_list中，后面会用到
        # 注意这里用了ModuLelist函数，模块化列表
        self.cell_list = nn.ModuleList(cell_list)
        self.activate_func = nn.Tanh()
    # 这里forward有两个输入参数，input_tensor 是一个五维数据
    # （t时间步,b输入batch_size,c输出数据通道数--维度,h,w图像高乘宽）
    # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, input_tensor, hidden_state=None):
        # 先调整一下输出数据的排列
        if not self.batch_first:
            #第一维和第二维互换,将时间步t换到第二维
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # 取出图片的数据，供下面初始化使用
        #b:batch_size，hw：图片高和宽
        b, _, _, h, w = input_tensor.size()
        # 初始化hidden_state,利用后面和lstm单元中的初始化函数
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        # 储存输出数据的列表
        layer_output_list = []
        last_state_list = []
        #输入数据的总长
        seq_len = input_tensor.size(1)

        # 初始化输入数据
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # 每一个时间步都更新 h,c
                # 注意这里self.cell_list是一个模块(容器)
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h)

            # 这一层的输出作为下一次层的输入,
            #将output_inner列表中的张量一个个根据dim维度进行连接起来，成一个张量。
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            # 储存每一层的状态h，c
            last_state_list.append([h, c])  # h,c size is [1,128,121,360]

        # 选择要输出所有数据，还是输出最后一层的数据
        #选择输出最后一层的数据
        if not self.return_all_layers:
            #只取最后一次输出的值
            layer_output_list = layer_output_list[-1:]

            #last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class MyModel(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,length):
        super(MyModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.length = length
        self.num_layers = len(hidden_dim)
        self.encoder = ConvLstm(input_dim=input_dim,
                                hidden_dim=self.hidden_dim,
                                kernel_size=(3,3),
                                num_layers=self.num_layers,
                                batch_first=False,
                                bias=False,
                                return_all_layers=False)
        self.out_channel = hidden_dim[self.num_layers-1]
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(self.out_channel, 16, kernel_size=3, stride=1, padding=1, bias=False),

            nn.ConvTranspose2d(16, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.process_hidden = nn.Conv2d(self.out_channel,self.hidden_dim[0],kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self, input, hidden=None):
        if hidden is None:
            y, output_c = self.encoder(input)
        else:
            y, output_c = self.encoder(input, hidden)

        y = torch.cat(y, dim=2)
        output = []
        for i in range(self.length):
            output.append(self.decoder(y[:, i, ...]))
        output = torch.stack(output, dim=1)
        output = output.permute(1,0,2,3,4)

        return output,output_c
