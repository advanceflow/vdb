import scipy.io
mat = scipy.io.loadmat('/home/zr/Downloads/Cleared-Set/FAll/mirflickr25k-fall.mat')
print(mat.keys())
print(mat)
print(mat['FAll'][:10])
print(mat['FAll'].shape)
# (20015, 1)


# label集合
mat = scipy.io.loadmat('/home/zr/Downloads/Cleared-Set/LAll/mirflickr25k-lall.mat')
print(mat.keys())
print(mat)
print(mat['LAll'][:10])
print(mat['LAll'].shape)
# (20015, 24)


# tag集合，文本
mat = scipy.io.loadmat('/home/zr/Downloads/Cleared-Set/YAll/mirflickr25k-yall.mat')
print(mat.keys())
print(mat)
print(mat['YAll'][:10])
print(mat['YAll'].shape)
# (20015, 1386)
# 文本嵌入维度是1386

# 图像
import h5py
file_path = '/home/zr/Downloads/Cleared-Set/IAll/mirflickr25k-iall.mat'
with h5py.File(file_path, 'r') as f:
    # 查看文件中的所有变量
    print("Keys in the file:", list(f.keys()))
    IAll = f['IAll']
    print("LAll的维度:", IAll.shape)
    # (20015, 3, 224, 224)
