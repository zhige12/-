# coding=utf-8
import numpy as np  # 修复：原numpy导入方式，统一用np别名更规范
from matplotlib import pyplot as plt


# 1. 计算两个向量的欧式距离（欧几里得距离）
def distEclud(vecA, vecB):
    # 计算逻辑：向量差 → 各元素平方 → 求和 → 开平方
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# 2. 初始化K个聚类中心（取数据集前K个样本作为初始中心）
def initCenter(dataSet, k):  # 修复：原"ini tCenter"有空格，改为initCenter
    print('2.initialize cluster center...')
    shape = dataSet.shape
    n = shape[1]  # n：数据集的特征数（列数，如二维数据n=2）
    # 初始化k行n列的全0数组，用于存储K个聚类中心
    classCenter = np.array(np.zeros((k, n)))
    # 遍历每个特征列，将数据集前k个样本的特征值赋值给聚类中心
    for j in range(n):
        firstK = dataSet[:k, j]  # 修复：原"firstk"大小写不一致，改为firstK
        classCenter[:, j] = firstK
    return classCenter


# 3. 实现K-Means核心算法
def myKMeans(dataSet, k):  # 修复：原"my KMeans"有空格，改为myKMeans
    m = len(dataSet)  # m：数据集的样本数（行数）
    # 初始化m行2列的数组：第1列=样本所属簇索引，第2列=样本到簇中心的平方距离
    clusterPoints = np.array(np.zeros((m, 2)))
    classCenter = initCenter(dataSet, k)  # 调用函数获取初始聚类中心
    clusterChanged = True  # 标记簇分配是否变化（用于循环终止条件）
    print('3. recompute and reallocated...')

    # 循环：直到簇分配不再变化（聚类稳定）
    while clusterChanged:
        clusterChanged = False  # 先重置为False，若后续有更新再设为True
        # 遍历每个样本，分配到最近的簇（修复：原range(5,m)跳过前5个样本，改为range(m)）
        for i in range(m):
            minDist = float('inf')  # 初始化最小距离为无穷大
            minIndex = -1  # 初始化最小距离对应的簇索引
            # 遍历每个聚类中心，计算样本到中心的距离
            for j in range(k):
                distJI = distEclud(classCenter[j, :], dataSet[i, :])
                # 若当前距离更小，更新最小距离和簇索引
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 若样本的簇索引发生变化，标记为簇分配变化，并更新聚类结果
            if clusterPoints[i, 0] != minIndex:
                clusterChanged = True
                clusterPoints[i, :] = minIndex, minDist ** 2  # 存储簇索引和平方距离

        # 重新计算每个簇的中心（用簇内所有样本的特征均值作为新中心）
        for cent in range(k):
            # 筛选出属于第cent个簇的所有样本
            ptsInClust = dataSet[np.nonzero(clusterPoints[:, 0] == cent)[0]]
            # 按列求均值（axis=0），更新该簇的中心
            classCenter[cent, :] = np.mean(ptsInClust, axis=0)
    return classCenter, clusterPoints


# 4. 可视化聚类结果（修复：处理无city.png的情况，可注释地图部分）
def show(dataSet, k, classCenter, clusterPoints):
    print('4. load the map...')
    fig = plt.figure(figsize=(8, 6))  # 增加图的大小，更清晰

    # --------------- 可选：若没有city.png，注释以下3行（避免报错）---------------
    rect = [0.1, 0.1, 1.0, 1.0]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    try:
        imgP = plt.imread('/Users/weijiazhi/7 实验素材/ch6/city.png')  # 读取背景地图（需与代码同目录）
        ax0.imshow(imgP)
    except FileNotFoundError:
        print("提示：未找到city.png，仅显示聚类散点")

    # 绘制聚类散点和簇中心
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    print('5. show the clusters...')
    numSamples = len(dataSet)
    # 聚类散点的标记样式（最多支持5个簇，可根据k扩展）
    mark = ['ok', '^b', 'om', 'og', 'sc']

    # 遍历每个样本，绘制散点（按簇分配不同样式）
    for i in range(numSamples):
        markIndex = int(clusterPoints[i, 0]) % k  # 确保标记索引不越界
        ax1.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex], alpha=0.7)  # alpha增加透明度

    # 遍历每个簇中心，绘制红色大圆点（突出显示）
    for i in range(k):
        ax1.plot(classCenter[i, 0], classCenter[i, 1], 'or', markersize=12, markerfacecolor='red')

    plt.title(f'K-Means Clustering Result (k={k})')
    plt.show()


# ---------------------- 主程序：加载数据 + 运行K-Means + 可视化 ----------------------
print('1. load dataset...')
# 方式1：加载testSet.txt（需手动创建，格式：每行2个数值，用空格分隔）
# 示例testSet.txt内容（可复制到文本文件中）：
# 1.658985 4.285136
# -3.453687 3.424321
# 4.838138 -1.151539
# -5.379713 -3.362104
# 0.972564 2.924086
try:
    dataSet = np.loadtxt('/Users/weijiazhi/7 实验素材/ch6/testSet.txt')
except FileNotFoundError:
    # 方式2：若没有testSet.txt，自动生成100个二维随机测试数据（避免报错）
    print("提示：未找到testSet.txt，自动生成测试数据")
    np.random.seed(42)  # 固定随机种子，结果可复现
    dataSet = np.random.randn(100, 2) * 5  # 100个样本，每个样本2个特征，正态分布

K = 5  # 聚类数量（可根据需求调整）
# 调用K-Means函数，获取聚类中心和样本聚类结果
classCenter, classPoints = myKMeans(dataSet, K)
# 可视化结果
show(dataSet, K, classCenter, classPoints)
