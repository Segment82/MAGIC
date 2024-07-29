import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K, temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.planes = int(in_planes * 0.5)
        self.kernel_num = K
        
        # Initialize of Non-local operation
        self.context_modeling = nn.Sequential(
            nn.Conv2d(in_planes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.GELU(),  # yapf: disable
            nn.Conv2d(self.planes, in_planes, kernel_size=1))
        self.conv_mask = nn.Conv2d(in_planes, 1, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=2)
    
        # if kernel_num != 1
        if K != 1:
            attention_channel = int(K/2)
            self.fc = nn.Conv2d(K, attention_channel, 1, bias=False)
            self.BN_kennel = nn.BatchNorm2d(attention_channel)
            self.relu = nn.ReLU(inplace=False)
            self.kernel_fc = nn.Conv2d(attention_channel, K, 1, bias=False)

        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

        self.sigmoid = nn.Sigmoid()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def global_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
        context_mask = self.conv_mask(x)
            # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        

        return context

    def forward(self,x):
        
        b, c, _, _ = x.size()
    
        # Dimensional-Reciprocal Fusion

        # Channel dimensional attention
        beta_c = self.avgpool(x) #b,c,1,1           
        
        # Non-local attention
        context = self.global_pool(x)
        beta_g = self.context_modeling(context)
        
        # Spatial dimensional attention        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        beta_s = spatial.repeat(1,int(c/2),1,1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1)
        
        out  = beta_c + beta_g + beta_s
                
        out = self.net(out)

        # Kernel Recalculation, if kernel_num == 1, the Kernel Recalculation is none.
        
        if self.kernel_num != 1 :
            kennel_attention = self.fc(out)    
            kennel_attention = self.BN_kennel(kennel_attention)
            kennel_attention = self.relu(kennel_attention)             
            kernel = self.kernel_fc(kennel_attention)
            kernel = self.sigmoid(kernel)
            out = torch.mul(out, kernel)
        
        att=out.view(x.shape[0],-1) #bs,K
        
        return F.softmax(att/self.temprature,-1)

class MAGIC(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size = 3,stride = 1,padding=1,dilation=1,grounps=1,bias=False,K=1,temprature=30,ratio=8,init_weight=True):
        super().__init__()
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.init_weight=init_weight
        self.attention=Attention(in_planes=in_planes,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)
        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(x) #bs,K
        
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k
        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)

        output = output.view(bs, self.out_planes, output.size(-2), output.size(-1))
        return output       
