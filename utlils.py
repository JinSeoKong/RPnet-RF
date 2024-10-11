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

    # 将数据 reshape 为 (1, num_bands, h, w) -> 4D 张量
    data = data.permute(2, 0, 1).unsqueeze(0)

    # 计算填充的高度和宽度
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # 镜像填充 (pad=(左, 右, 上, 下))
    # 仅在右和下进行填充
    padded_data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')  # (左, 右, 上, 下)
    print('Padded data shape:', padded_data.shape)
    # 提取补丁 (unfold 将图像分成补丁)
    all_patches = padded_data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # 调整形状为 (num_patches, patch_size, patch_size, num_bands)
    all_patches = all_patches.contiguous().view(-1, num_bands, patch_size, patch_size)
    
    # 从所有补丁中随机选择 num_patches 个
    random_indices = torch.randint(0, all_patches.size(0), (num_patches,))
    selected_patches = all_patches[random_indices]


    conv_output = F.conv2d(data, selected_patches, padding=padding)
    output = F.relu(conv_output)
    output = output.squeeze(0).permute(1, 2, 0)
    print('RPNet output shape:', output.shape)
    return output
