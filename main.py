import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import seaborn as sns
import argparse
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utlils import *
from model import MLP

def train_mlp(X_train, y_train, X_test, y_test, input_size, hidden_size, num_classes, epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用

    # 初始化模型并移至GPU
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 转换数据为张量并移至GPU
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 开始训练
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清空梯度
        outputs = model(X_train)  # 前向传播
        loss = criterion(outputs, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 评估模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测值
        accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        kappa = cohen_kappa_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        
        # 使用分类报告和绘制混淆矩阵
        print(classification_report(y_test.cpu().numpy(), predicted.cpu().numpy()))
        
        # 创建混淆矩阵
        conf_matrix = confusion_matrix(y_test.cpu().numpy(), predicted.cpu().numpy())
        target_names = [str(i) for i in range(num_classes)]
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Kappa: {kappa:.4f}')
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Land Cover Classification')
    parser.add_argument('--dataset', type=str, default='paviaU', help='dataset name (default: paviaU)')
    parser.add_argument('--layer_num', type=int, default=4, help='number of Random Patches layers (default: 4)')
    args = parser.parse_args()

    #load data
    if args.dataset == 'paviaU':
        data = sio.loadmat('/home/user_3/game6_land_cover_classification/datasets/paviaU/paviaU.mat')['paviaU']  # Replace with your file path
        ground_truth= sio.loadmat('/home/user_3/game6_land_cover_classification/datasets/paviaU/paviaU_gt.mat')['paviaU_gt'] # Replace with your file path
        print('Data loaded for PaviaU dataset')
        print('Data shape:', data.shape)
        print('Ground truth shape:', ground_truth.shape)
    else:
        print('Invalid dataset name')

    #pca whitening
    num_components = 3
    data_pca = apply_PCA_Whitening(data, num_components)
    print('PCA Whitening done')
    print('PCA Whitened data shape:', data_pca.shape)

    RPNet_feature_map =  torch.from_numpy(data).float()
    for i in range(args.layer_num):
        output = Random_patches_layer(data_pca, patch_size=21, num_patches=20)
        if RPNet_feature_map is None:
            RPNet_feature_map = output
        else:
            RPNet_feature_map = torch.cat((RPNet_feature_map, output), dim=2)
        data_pca = apply_PCA_Whitening(output, num_components)
        
    print('RPNet feature maps generated')
    print('RPNet feature map shape:', RPNet_feature_map.shape)
    print(type(RPNet_feature_map))

    # Flatten feature map for SVM input
    rpnet_features = RPNet_feature_map.numpy().reshape(-1, RPNet_feature_map.size(2))
    labels = ground_truth.flatten()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(rpnet_features, labels, test_size=0.8, random_state=42, stratify=labels)

    # MLP参数
    input_size = X_train.shape[1]   # 输入特征的大小
    hidden_size = 4096      # 隐藏层神经元个数
    num_classes = len(np.unique(labels))  # 分类的类别数
    
    # 训练MLP模型
    model = train_mlp(X_train, y_train, X_test, y_test, input_size, hidden_size, num_classes, epochs=400, learning_rate=1e-3)
    # Save model
    torch.save(model.state_dict(), f'RPNet_MLP_{args.dataset}.pth')
    print('Model saved')

    

    