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
from matplotlib.colors import ListedColormap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用

def train_mlp(X_train, y_train, X_test, y_test, input_size, hidden_size, num_classes, epochs=20, learning_rate=0.001):
    

    # 初始化模型并移至GPU
    model = MLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 转换数据为张量并移至GPU
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 记录训练集和测试集的损失
    train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []

    # 开始训练
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清空梯度
        outputs = model(X_train)  # 前向传播
        loss = criterion(outputs, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss_list.append(loss.item())

        # 测试集评估
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_loss_list.append(test_loss.item())
            
            _, predicted = torch.max(test_outputs.data, 1)  # 获取预测值
            test_accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
            test_accuracy_list.append(test_accuracy)

        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_loss_list, label='Train Loss')
    plt.plot(range(epochs), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    plt.savefig('loss_curve.png')
    plt.show()

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
        
        print(f'Final Accuracy: {accuracy * 100:.2f}%')
        print(f'Final Kappa: {kappa:.4f}')
    
    return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Land Cover Classification')
    parser.add_argument('--dataset', type=str, default='paviaU', help='dataset name (default: paviaU)')
    parser.add_argument('--layer_num', type=int, default=5, help='number of Random Patches layers')
    parser.add_argument('--num_components', type=int, default=3, help='number of PCA components')
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
    num_components = 8
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
    
    # 选择非0类的所有样本
    non_zero_indices = np.where(labels != 0)[0]

    # 使用这些索引获取新的平衡后的labels
    labels = labels[non_zero_indices] -1
    rpnet_features = rpnet_features[non_zero_indices]
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(rpnet_features, labels, test_size=0.9, random_state=42, stratify=labels)

    # MLP参数
    input_size = X_train.shape[1]   # 输入特征的大小
    hidden_size = 4096      # 隐藏层神经元个数
    num_classes = len(np.unique(labels)) # 分类的类别数
    
    # 训练MLP模型
    model = train_mlp(X_train, y_train, X_test, y_test, input_size, hidden_size, num_classes, epochs=100, learning_rate=1e-3)
    # Save model
    torch.save(model.state_dict(), f'RPNet_MLP_{args.dataset}.pth')
    print('Model saved')
    # 评估模型
    model.eval()  # 设置模型为评估模式
    X = torch.from_numpy(RPNet_feature_map.numpy().reshape(-1, RPNet_feature_map.size(2))).float().to(device)
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测值
        
        
        # 绘制预测结果, 将ground_truth和predicted_labels绘制在一个图片中
        predicted_labels = predicted.cpu().numpy()
        # reshape predicted_labels
        predicted_labels = predicted_labels.reshape(ground_truth.shape)
        # 定义颜色调色板
        palette = np.array([
            [216, 191, 216],
            [0, 255, 0],
            [0, 255, 255],
            [45, 138, 86],
            [255, 0, 255],
            [255, 165, 0],
            [159, 31, 239],
            [255, 0, 0],
            [255, 255, 0]
        ])

        # 将颜色转换为范围 [0, 1] 之间的浮点数
        palette = palette / 255.0

        # 创建自定义颜色映射
        cmap_custom = ListedColormap(palette)
        plt.figure(figsize=(10, 5))

        # 绘制真实标签
        plt.subplot(1, 2, 1)
        plt.title('Ground Truth')
        plt.imshow(ground_truth, cmap='jet')
        plt.colorbar()

        # 绘制预测标签
        plt.subplot(1, 2, 2)
        plt.title('Predicted Labels')
        plt.imshow(predicted_labels, cmap=cmap_custom)
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('comparison.png')
    

    