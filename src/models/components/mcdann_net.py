import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, gap, fc1, dropout, fc2, relu, sigmoid):
        super(SEBlock, self).__init__()
        self.gap = gap
        # Cải tiến: Thêm dropout và better initialization
        #reduced_dim = max(in_channels // reduction, minimum_reduced_dim)
        self.fc1 = fc1
        self.dropout = dropout
        self.fc2 = fc2
        self.relu = relu
        self.sigmoid = sigmoid

    def forward(self, x):
        b, c, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        # mở rộng thêm 1 chiều để scale với dữ liệu
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


class DenseBlock(nn.Module):
    def __init__(self, lrelu, bn1, conv1, dropout1, bn2, conv2, dropout2 ): # out_channels = 8
        super().__init__()
        self.lrelu = lrelu

        self.bn1 = bn1
        # Cải tiến: Thêm bias=False và dropout
        self.conv1 = conv1
        self.dropout1 = dropout1
        
        self.bn2 = bn2
        self.conv2 = conv2
        self.dropout2 = dropout2

    def forward(self, x):
        # First Composite Function
        out = self.bn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)
        if self.training:
            out = self.dropout1(out)
        x = torch.cat([x, out], dim=1)

        # Second Composite Function
        out = self.bn2(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.training:
            out = self.dropout2(out)
        x = torch.cat([x, out], dim=1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, conv, bn, lrelu, pool):
        super().__init__()
        # Cải tiến: Thêm batch norm và activation
        self.conv = conv
        self.bn = bn
        self.lrelu = lrelu
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.pool(x)
        return x


class DACB(nn.Module):
    def __init__(self, conv1_3, conv1_5, conv1_7, conv1_9, initial_bn, initial_lrelu, dense1, transition1, se1, dense2,
                transition2, se2, dense3, final_conv, final_bn, skip, lrelu, gap):
        super(DACB, self).__init__()
        # Cải tiến: Multi-scale initial convolution
        self.conv1_3 = conv1_3
        self.conv1_5 = conv1_5
        self.conv1_7 = conv1_7
        self.conv1_9 = conv1_9
        self.initial_bn = initial_bn
        self.initial_lrelu = initial_lrelu

        self.dense1 = dense1
        self.transition1 = transition1
        self.se1 = se1

        self.dense2 = dense2
        self.transition2 = transition2
        self.se2 = se2

        # Cải tiến: Additional dense block for deeper features
        self.dense3 = dense3
        self.final_conv = final_conv
        self.final_bn = final_bn

        # Cải tiến: Better skip connection
        self.skip = skip
        self.lrelu = lrelu
        self.gap = gap

    def forward(self, x):                                           # input shape = [batch size, 1, 600]
        # Cải tiến: Multi-scale feature extraction
        f1 = self.conv1_3(x)                                        # input shape = [bs, 4, 300]
        f2 = self.conv1_5(x)                                        # input shape = [bs, 4, 300]
        f3 = self.conv1_7(x)                                        # input shape = [bs, 4, 300]
        f4 = self.conv1_9(x)                                        # input shape = [bs, 4, 300]
        x = torch.cat([f1, f2, f3, f4], dim=1)                      # input shape = 4 * [bs, 4, 300]
        x = self.initial_bn(x)                                      # input shape = [bs, 16, 300]
        x = self.initial_lrelu(x)                                   # input shape = [bs, 16, 300]

        # Store for skip connection
        skip_input = x

        d = self.dense1(x)                                          # input shape = [bs, 16, 300]
        d = self.transition1(d)                                     # input shape = [bs, 32, 300]
        d = self.se1(d)                                             # input shape = [bs, 64, 150]

        d = self.dense2(d)                                          # input shape = [bs, 64, 150]
        d = self.transition2(d)                                     # input shape = [bs, 80, 150]
        d = self.se2(d)                                             # input shape = [bs, 64, 75]

        # Cải tiến: Additional processing
        d = self.dense3(d)                                          # input shape = [bs, 64, 75]
        d = self.final_conv(d)                                      # input shape = [bs, 80, 75]
        d = self.final_bn(d)                                        # input shape = [bs, 64, 75]
        d = self.lrelu(d)                                           # input shape = [bs, 64, 75]

        skip = self.skip(skip_input)                                # input shape = [bs, 16, 300]

        # Cải tiến: Residual connection instead of concatenation
        if d.size(-1) != skip.size(-1):                             # input shape = [bs, 64, 300]
            skip = F.interpolate(skip, size=d.size(-1), mode='linear', align_corners=False)

        # Residual connection,
        out = d + skip                                              # input shape = [bs, 64, 75] + [bs, 64, 75]
        out = self.lrelu(out)                                       # input shape = [bs, 64, 75]
        out = self.gap(out)                                         # input shape = [bs, 64, 75]

        return out                                                  # out shape = [bs, 64, 1]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_leads=12):
        super().__init__()
        pe = torch.zeros(num_leads, d_model)
        position = torch.arange(0, num_leads, dtype=torch.float).unsqueeze(1) # tạo mảng có shape=[12,1], có giá trị 0->11
        # Sinh vị trí cho các lead
        # hàm toán học sinh vị trí
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # cột chẵn dùng hàm sin để sinh vị trí
        pe[:, 0::2] = torch.sin(position * div_term)
        # cột lẻ dùng hàm cos để sinh vị trí
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, num_leads, d_model]
        return x + self.pe[:, :x.size(1)]


class MCDANNNet(nn.Module):
    def __init__(self, num_classes, channels, positional_encoding, lead_attention, classifier):
        super(MCDANNNet, self).__init__()
        self.channels = channels  # nn.ModuleList([DACB() for _ in range(12)])  # 12 module per 12 leads

        # Cải tiến: Positional encoding for leads
        self.positional_encoding = positional_encoding
        
        # Cải tiến: Cross-lead attention mechanism
        self.lead_attention = lead_attention
        
        # Cải tiến: Enhanced classifier with feature fusion
        self.classifier = classifier
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # Khởi tạo params cho Linear layers
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

        # Khởi tạo params cho Linear layers
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

        #Khởi tạo params cho BN layers
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, val=1)
            nn.init.constant_(m.bias, val=0)

    def forward(self, x):  # x shape = (batch size, num_leads, 600)
        batch_size = x.size(0)
        
        # Cải tiến: Adaptive downsampling
        x = x[:, :, ::2]  # Downsample 0.5

        # Cải tiến: Enhanced normalization (per-lead)
        mean = x.mean(dim=2, keepdim=True)  
        std = x.std(dim=2, keepdim=True)    
        x = (x - mean) / (std + 1e-8)                                      # Add 1e-8 to avoid divide by 0
        
        features = []
        for i, channel in enumerate(self.channels):
            lead = x[:, i, :].unsqueeze(1)                                 # lead shape = [batch size, 1, 600]
            feat = channel(lead).squeeze(-1)                               # feat shape = [batch size, 64]
            features.append(feat)
        
        # Cải tiến: Stack features for attention
        stacked_features = torch.stack(features, dim=1)                    # output shape [batch size, num_leads, 64]
        
        # Cải tiến: Add positional encoding to help attention understand lead positions
        stacked_features = self.positional_encoding(stacked_features)      # output shape [batch size, num_leads, 64]
        
        # Cải tiến: Apply cross-lead attention
        attended_features, _ = self.lead_attention(
            stacked_features, stacked_features, stacked_features           # output shape [batch size, num_leads, 64]
        )
        
        # Cải tiến: Combine original and attended features
        enhanced_features = stacked_features + attended_features           # output shape [batch size, num_leads, 64]
        
        # Flatten for classification
        combined = enhanced_features.view(batch_size, -1)                  # output shape [batch size, 768]
        return self.classifier(combined)
