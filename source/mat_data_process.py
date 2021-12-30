#%%
import scipy.io as sio
import os

#%%[markdown]
# 简化数据文件名和变量名

#%% 输入部分
'''需要修改的部分，只有如下三行'''
tar_dir = "F:\\STUDY\\数据\\109小轴承试验台轴承复合故障数据集\\1500\\outer"
old_represent = "outer" # 旧的命名
new_represent = 'o' # 新的命名
'''需要修改部分的截止线'''

filename_list = os.listdir(tar_dir)
print("原始文件名有："+str(filename_list))

#%%
data_variable_name = [i.replace('.mat',"") for i in filename_list]
print("原始变量名有："+str(data_variable_name))

#%%
# 使用简化表示代替原始表示
new_filename_list = [i.replace(old_represent,new_represent) for i in filename_list]
new_data_variable_name = [i.replace(".mat","") for i in new_filename_list]

for i in range(len(filename_list)):
    data = sio.loadmat(os.path.join(tar_dir,filename_list[i]))[data_variable_name[i]]
    sio.savemat(os.path.join(tar_dir,new_filename_list[i]),{new_data_variable_name[i]:data})

#%%
# 查看新的mat文件包含的键值
datadict = sio.loadmat("F:\STUDY\数据\\109小轴承试验台轴承复合故障数据集\900\inner\\i_H1.mat")
print(datadict.keys()) # out:  dict_keys(['__header__', '__version__', '__globals__', 'i_H1'])


'''
======================================================================================================
'''
#%%[markdown]
# 对数据采集软件原始的mat数据文件整理，把各个通道的数据单独分离出来   
# 原始数据文件的键值字典  ['__header__', '__version__', '__globals__', 'File_Header', 'Channel_1_Header', 'Channel_2_Header', 'Channel_3_Header', 'Channel_1_Data', 'Channel_2_Data', 'Channel_3_Data']

# 包含数据的键值为： ['Channel_1_Data', 'Channel_2_Data', 'Channel_3_Data'] 
 
# 'File_Header'—— 包含了采样率、通道数等文件信息的结构体。
# 'Channel_1_Header'和'Channel_2_Header'和'Channel_3_Header' ——包括了通道的单位、信号名字等。

# 109小轴承试验台复合故障数据采集，各通道和对应传感器关系如下：
# 1通道——垂直V振动加速度传感器；
# 2通道——水平H加速度传感器；
# 3通道——转速计信号

# 目标文件名和变量名的格式： 轴承状态_H/V文件序号
# 原始文件名的格式： 有点混乱，但是最后是“_序号”这样的格式，可以使用正则化表达式得到文件序号；

#%%
rpm = ['900','1200','1500']
base_dir = "F:\STUDY\数据\\109小轴承试验台轴承复合故障数据集"

# 需要修改的部分
old_represent = "roller_cage"
new_represent = 'rc'
#%%
# 遍历不同转速，同一健康状态的数据文件
for i in range(len(rpm)):
    tar_dir = os.path.join(base_dir,rpm[i],old_represent) # out 示例："F:\STUDY\数据\\109小轴承试验台轴承复合故障数据集\900\inner_cage"

    # 得到文件序号（倒数第五个字符）或者（使用正则化方法，先找到_1.mat这种格式的字符串，再提取字符串中的数字）
    # 本复合故障数据，每个类型数据文件的数量没有超过10个，直接使用查找倒数第五个字符即可。

    filename_list = os.listdir(tar_dir)
    file_nums = [i[-5] for i in filename_list] # 倒数第五位是文件号字符串

    # 统一mat文件命名格式：“轴承状态_H/V文件序号.mat”
    # 包含数据的键值为： ['Channel_1_Data', 'Channel_2_Data', 'Channel_3_Data']
    # 1通道——垂直V振动加速度传感器；
    # 2通道——水平H加速度传感器；
    # 3通道——转速计信号
    
    for j in range(len(file_nums)):
        data_h = sio.loadmat(os.path.join(tar_dir,filename_list[j]))['Channel_2_Data']
        data_v = sio.loadmat(os.path.join(tar_dir,filename_list[j]))['Channel_1_Data']
        data_h_name = new_represent + '_' + "H" + file_nums[j]
        data_v_name = new_represent + '_' + "V" + file_nums[j]
        exec(data_h_name + "=" + 'data_h')
        exec(data_v_name + "=" + 'data_v')
        sio.savemat(os.path.join(tar_dir,data_h_name + '.mat'),{data_h_name:data_h})
        sio.savemat(os.path.join(tar_dir,data_v_name + '.mat'),{data_h_name:data_h})

        # print(sio.loadmat(os.path.join(tar_dir,data_v_name + '.mat')).keys())

#%%
# 查看新的mat文件包含的键值
datadict = sio.loadmat("F:\STUDY\数据\\109小轴承试验台轴承复合故障数据集\900\inner_cage\\ic_H1.mat")
print(datadict.keys()) # out:  dict_keys(['__header__', '__version__', '__globals__', 'ic_H1'])
print(datadict['ic_H1'].shape) # out: (512362, 1)
