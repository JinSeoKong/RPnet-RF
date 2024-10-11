import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
from model import MLP
from utlils import apply_PCA_Whitening, plot_pca_components, Random_patches_layer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Split data into train and test sets
    X_test = torch.from_numpy(rpnet_features).float().to(device)
    y_test = torch.from_numpy(labels).long().to(device)
    # MLP参数
    input_size = X_test.shape[1]   # 输入特征的大小
    hidden_size = 4096      # 隐藏层神经元个数
    num_classes = len(np.unique(labels))  # 分类的类别数

    
    model = MLP(input_size, hidden_size, num_classes).to(device)
    print(model)
    model.load_state_dict(torch.load('/home/user_3/game6_land_cover_classification/RPnet-RF/RPNet_MLP_paviaU.pth'))

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
        plt.savefig('confusion_matrix_test.png')


        # 绘制预测结果, 将ground_truth和predicted_labels绘制在一个图片中
        predicted_labels = predicted.cpu().numpy()
        # reshape predicted_labels
        predicted_labels = predicted_labels.reshape(ground_truth.shape)

        plt.figure(figsize=(10, 5))

        # 绘制真实标签
        plt.subplot(1, 2, 1)
        plt.title('Ground Truth')
        plt.imshow(ground_truth, cmap='jet')
        plt.colorbar()

        # 绘制预测标签
        plt.subplot(1, 2, 2)
        plt.title('Predicted Labels')
        plt.imshow(predicted_labels, cmap='jet')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('comparison.png')
        