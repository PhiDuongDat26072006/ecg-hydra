import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, minimum_reduced_dim=4, chanel_size_out_gap=1, p_dropout=0.1):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(chanel_size_out_gap)
        # Cải tiến: Thêm dropout và better initialization
        reduced_dim = max(in_channels // reduction, minimum_reduced_dim)
        self.fc1 = nn.Linear(in_channels, reduced_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(reduced_dim, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
    def __init__(self, in_channels, growth_rate=8, kernel_sizes=[5,3], leaky=0.01, p_dropout=0.1):
        super().__init__()
        self.lrelu = nn.LeakyReLU(leaky)
        self.bn1 = nn.BatchNorm1d(in_channels)
        # Cải tiến: Thêm bias=False và dropout
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_sizes[0],
                               padding="same", bias=False)
        self.dropout1 = nn.Dropout1d(p_dropout)
        
        self.bn2 = nn.BatchNorm1d(in_channels + growth_rate)
        self.conv2 = nn.Conv1d(in_channels + growth_rate, growth_rate, kernel_size=kernel_sizes[1],
                               padding="same", bias=False)
        self.dropout2 = nn.Dropout1d(p_dropout)

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
    def __init__(self, in_channels, out_channels=64, leak_scale=0.01):
        super().__init__()
        # Cải tiến: Thêm batch norm và activation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(leak_scale)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

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

    def forward(self, x):   # x shape = [Batch size, 1, 300)
        # Cải tiến: Multi-scale feature extraction
        f1 = self.conv1_3(x)
        f2 = self.conv1_5(x)
        f3 = self.conv1_7(x)
        f4 = self.conv1_9(x)
        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.initial_bn(x)
        x = self.initial_lrelu(x)
        
        # Store for skip connection
        skip_input = x

        d = self.dense1(x)
        d = self.transition1(d)
        d = self.se1(d)

        d = self.dense2(d)
        d = self.transition2(d)
        d = self.se2(d)
        
        # Cải tiến: Additional processing
        d = self.dense3(d)
        d = self.final_conv(d)
        d = self.final_bn(d)
        d = self.lrelu(d)
        
        skip = self.skip(skip_input)
        
        # Cải tiến: Residual connection instead of concatenation
        if d.size(-1) != skip.size(-1): # nếu chiều cuối của 2 dữ liệu khác nhau thì dùng linear interpolate
            skip = F.interpolate(skip, size=d.size(-1), mode='linear', align_corners=False)
        
        out = d + skip  # Residual connection
        out = self.lrelu(out)
        out = self.gap(out)
        
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, num_leads=12):
        super().__init__()
        pe = torch.zeros(num_leads, d_model)
        position = torch.arange(0, num_leads, dtype=torch.float).unsqueeze(1) # shape_arr=(12,1), có giá trị 0->11

        # Sinh vị trí cho các lead
        # hàm sinh vị trí
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # các cột chẵn dùng hàm sin để sinh vị trí
        pe[:, 0::2] = torch.sin(position * div_term)
        # các cột lẻ dùng hàm cos để sinh vị trí
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, num_leads, d_model]
        return x + self.pe[:, :x.size(1), :]


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
        x = (x - mean) / (std + 1e-8)  # Cộng 1e-8 để tránh chia cho 0
        
        features = []
        for i, channel in enumerate(self.channels):
            # lead shape = (batch size, 1, 600)
            lead = x[:, i, :].unsqueeze(1)
            # feat shape = (batch size, 64)
            feat = channel(lead).squeeze(-1)
            features.append(feat)
        
        # Cải tiến: Stack features for attention [batch size, num_leads, 64]
        stacked_features = torch.stack(features, dim=1)
        
        # Cải tiến: Add positional encoding to help attention understand lead positions
        stacked_features = self.positional_encoding(stacked_features)
        
        # Cải tiến: Apply cross-lead attention
        attended_features, _ = self.lead_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Cải tiến: Combine original and attended features
        enhanced_features = stacked_features + attended_features
        
        # Flatten for classification [batch size, 768]
        combined = enhanced_features.view(batch_size, -1)
        return self.classifier(combined)
