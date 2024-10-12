import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import torch
def apply_PCA_Whitening(data, num_components):
    # Reshape the data into a 2D array (samples x features)
    num_samples, num_features, num_bands = data.shape
    flattened_data = data.reshape(num_samples * num_features, num_bands)

    # Apply PCA to the flattened data
    pca = PCA(n_components=num_components,whiten=True)
    pca.fit(flattened_data)

    # Transform the data back to the original shape
    transformed_data = pca.transform(flattened_data)
    whole_pca = transformed_data.reshape(num_samples, num_features, num_components)
    return whole_pca

def plot_pca_components(pca_data, dataset_name, num_components=2):
    fig, axes = plt.subplots(1, num_components, figsize=(15, 5))
    for i in range(num_components):
        axes[i].imshow(pca_data[:, :, i], cmap='jet')
        axes[i].set_title(f'Principal Component {i+1}')
        axes[i].axis('off')
    fig.suptitle(f'PCA Components for {dataset_name}', fontsize=16)
    plt.savefig(f'PCA_components_{dataset_name}.png')

def Random_patches_layer(data, patch_size=3, num_patches=50,padding=10):
    # 将 NumPy 数组转换为 PyTorch 张量
    
    data = torch.from_numpy(data).float()
    
    # 获取数据的形状 (h, w, num_bands)
    h, w, num_bands = data.shape
    data = data.permute(2,0,1).unsqueeze(0)
    

    # 计算填充的高度和宽度
    pad_h = patch_size
    pad_w = patch_size

    # 镜像填充 (pad=(左, 右, 上, 下))
    # 仅在右和下进行填充
    padded_data = F.pad(data, pad=(pad_w, pad_w, pad_h, pad_h), mode='reflect')  # (左, 右, 上, 下)
    
    # 提取补丁 
    patches = []
    for _ in range(num_patches):
        # 随机选择中心像素的坐标
        center_y = np.random.randint(pad_h, h + pad_h)
        center_x = np.random.randint(pad_w, w + pad_w)
        # 获取补丁的边界
        y_start = center_y - patch_size // 2
        y_end = center_y + patch_size // 2 + 1
        x_start = center_x - patch_size // 2
        x_end = center_x + patch_size // 2 + 1

        # 提取补丁并保存
        patch = padded_data[:,:,y_start:y_end, x_start:x_end].squeeze(0)
        patches.append(patch)
    all_patches = torch.stack(patches)
    
    print('All patches shape:', all_patches.shape)
    selected_patches = all_patches


    conv_output = F.conv2d(data, selected_patches, padding=padding)
    output = F.relu(conv_output)
    output = output.squeeze(0).permute(1, 2, 0)
    print('RPNet output shape:', output.shape)
    return output
