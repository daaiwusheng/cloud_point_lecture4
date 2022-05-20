# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import KDTree
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def get_cosine_with_z_axis(point):
    x = point[0]
    y = point[1]
    z = point[2]

    length = np.sqrt(x * x + y * y + z * z)
    cosine = abs(z) / length
    return cosine


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    segmengted_cloud = np.zeros((0, data.shape[1]))
    inlier_cloud_final = np.zeros((0, data.shape[1]))
    # 1. 构建kdtree
    print("开始构建KDTree")
    tree = KDTree(data, leaf_size=8)
    print("构建KDTree 完成")
    # 2. 计算每个点的法向量
    print("开始计算法向量")
    search_radious = 1.0
    neighbour_indexes = tree.query_radius(data, r=search_radious)
    normals_array = np.zeros((data.shape[0], data.shape[1]))
    bad_normal_indexes = np.zeros((data.shape[0], 1), dtype=int)
    groudn_cloud_num = 0
    threshold = 3
    threshold_cosine_with_z_axis = 0.87  # 余弦30°
    for i in range(len(data)):
        if len(neighbour_indexes[i]) >= threshold:
            # 满足这个条件,就可以计算法向量了
            neighbour_points_mean = np.mean(data[neighbour_indexes[i], :], axis=0)
            neighbour_points_reduce_mean = data[neighbour_indexes[i], :] - neighbour_points_mean
            H = np.dot(neighbour_points_reduce_mean.T, neighbour_points_reduce_mean)
            eigenvalues, eignevectors = np.linalg.eig(H)
            normal_index = np.argmin(eigenvalues)
            current_normal = eignevectors[:, normal_index].T
            normals_array[i, :] = current_normal
            current_cosine_with_z = get_cosine_with_z_axis(current_normal)
            if current_cosine_with_z < threshold_cosine_with_z_axis:
                bad_normal_indexes[neighbour_indexes[i]] = 1
            else:
                groudn_cloud_num += 1
        else:
            bad_normal_indexes[neighbour_indexes[i]] = 1
    print("法向量计算完毕")
    # 3. 拟合地面,根据法向量是合格的点
    # inlier_threshold = 0.7 * (1 - bad_normal_indexes.sum(axis=0) / len(data))
    inlier_threshold = 0.6
    print("inlier_threshold = {}".format(inlier_threshold))

    tau = 0.5  # 属于内点的阈值
    num_iteration = 50
    segmented_ground_indexes_set = set()
    for i in range(num_iteration):
        # 随机选取三个点
        print("第 {} 次迭代".format(i))
        print("随机选取三个点")
        ps = np.zeros((0, data.shape[1]))
        while len(ps) < 3:
            random_choice = np.random.choice(len(data), 1)[0]
            current_point = data[random_choice, :]
            if bad_normal_indexes[random_choice] != 1:
                # 这里只看不等于1就行了,因为上面计算法向量的过程,把不合格的点都给标记出来了
                ps = np.append(ps, [current_point], axis=0)
        print("选点完毕")
        # 开始几何平面
        print("-" * 25)
        print("开始拟合平面")
        a = (ps[1][1] - ps[0][1]) * (ps[2][2] - ps[0][2]) - (ps[1][2] - ps[0][2]) * (ps[2][1] - ps[0][1])
        b = (ps[1][2] - ps[0][2]) * (ps[2][0] - ps[0][0]) - (ps[1][0] - ps[0][0]) * (ps[2][2] - ps[0][2])
        c = (ps[1][0] - ps[0][0]) * (ps[2][1] - ps[0][1]) - (ps[1][1] - ps[0][1]) * (ps[2][0] - ps[0][0])
        d = 0 - (a * ps[0][0] + b * ps[0][1] + c * ps[0][2])
        print("平面拟合完毕")
        inlier_cloud_current = np.zeros((0, data.shape[1]))  # 存实际的点的坐标
        inlier_index = np.zeros((0, 1), dtype=int)  # 存内点的索引
        print("计算其余点是否是内点")
        for j in range(len(data)):
            current_point_j = data[j, :]
            distance_to_plane = abs(a * current_point_j[0] + b * current_point_j[1] + c * current_point_j[2] + d) / \
                                np.sqrt(a * a + b * b + c * c)
            if distance_to_plane < tau and bad_normal_indexes[j] != 1:
                inlier_cloud_current = np.append(inlier_cloud_current, [current_point_j], axis=0)
                inlier_index = np.append(inlier_index, j)
        # 所谓的 法线向上的点, 不全都是地面点,只能说大部分是地面点
        # 基于上述原因, 所以要进行平面拟合, 不然就没办法知道哪些属于地面点.
        # 还有一个重点需要考虑, 拟合出来的平面的内点, 应该设置一个下限, 比如占所谓的地面点或者总点数比例多少,
        # 才能视她们为地面点, 不然或许拟合的平面是别的食物的平面,

        inlier_percent = len(inlier_cloud_current) / groudn_cloud_num
        if inlier_percent > 0.1:
            # 所有认为是地面点的内点存起来, 后面再从原始点云中删掉
            segmented_ground_indexes_set.update(inlier_index)

        print("内点比例为: {}".format(inlier_percent))
        if inlier_percent > inlier_threshold:  # 可以提前终止
            break

    # 屏蔽结束
    segmented_ground = data[list(segmented_ground_indexes_set)]
    segmengted_cloud = np.delete(data, list(segmented_ground_indexes_set), axis=0)

    print('origin data points num:', data.shape[0])
    print("地面点num : ", len(segmented_ground))
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud, segmented_ground


def scan_traditional_method(core_points_list, dict_cluster_index, neighbor_indexes):
    cluster_tag = 0
    while len(core_points_list):
        neighbour_points_indexes_later = set()
        current_core_point_index = core_points_list[0]
        neighbour_points_indexes_later.add(current_core_point_index)
        while len(neighbour_points_indexes_later):
            dict_cluster_index[current_core_point_index] = cluster_tag

            for point_i in neighbor_indexes[current_core_point_index]:
                if point_i in core_points_list and point_i != current_core_point_index:
                    neighbour_points_indexes_later.add(point_i)
                else:
                    dict_cluster_index[point_i] = cluster_tag

            if current_core_point_index in neighbour_points_indexes_later:
                neighbour_points_indexes_later.remove(current_core_point_index)
            core_points_list.remove(current_core_point_index)
            # 获取下一个邻域内的点
            if len(neighbour_points_indexes_later):
                current_core_point_index = neighbour_points_indexes_later.pop()
                if len(neighbour_points_indexes_later) == 0:
                    dict_cluster_index[current_core_point_index] = cluster_tag
                    core_points_list.remove(current_core_point_index)
            print("聚类中 --- {cluster}----点={point}".format(cluster=cluster_tag, point=current_core_point_index))

        cluster_tag += 1
        print("{} 聚类完毕".format(cluster_tag))


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    print("开始聚类")
    print("开始构建kdtree")
    tree = KDTree(data, leaf_size=8)
    print("构建kdtree完毕")
    neighbor_indexes = tree.query_radius(data, r=1.0)
    print("开始搜索核心点")
    core_points_set = set()
    clusters_index = np.zeros(len(data), dtype=int)
    unvisited_core_points_set = set()
    for i in range(len(data)):
        clusters_index[i] = -1
        unvisited_core_points_set.add(i)
        if len(neighbor_indexes[i]) >= 3:
            # 满足此条件就认为是核心点, 因为包含基准点本身如果邻域内有三个点那么就能计算出一个法向量了
            core_points_set.add(i)

    print("核心点搜索完毕")
    print("开始dbscan聚类")
    cluster_tag = 0

    while len(core_points_set):
        # 随机取出一个core point
        current_core_point = core_points_set.pop()
        visited_points_set = set()
        # 根据这个核心点找到它辐射到的点,
        current_neighbor_points_set = set(neighbor_indexes[current_core_point])
        later_visit_core_points_set = set()
        later_visit_core_points_set.update(current_neighbor_points_set & core_points_set)
        visited_points_set.update(current_neighbor_points_set)
        while len(later_visit_core_points_set):
            current_core_point = later_visit_core_points_set.pop()
            if current_core_point in core_points_set:
                core_points_set.remove(current_core_point)
            current_neighbor_points_set = set(neighbor_indexes[current_core_point])
            later_visit_core_points_set.update(current_neighbor_points_set & core_points_set)
            visited_points_set.update(current_neighbor_points_set)
        clusters_index[list(visited_points_set)] = cluster_tag
        cluster_tag += 1

    print("结束dbscan聚类")
    # 屏蔽结束

    return clusters_index


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection='3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters_v_1(segmented_ground, segmented_cloud, cluster_index):
    def colormap(c, num_clusters):
        # outlier:
        if c == -1:
            color = [0] * 3
        # surrouding object:
        else:
            color = [c / num_clusters * 128 + 127] * 3
            color[c % 3] = 0

        return color

    # 地面的颜色
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0, 0, 255] for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    print(num_clusters)

    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, num_clusters) for c in cluster_index
        ]
    )
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])


def main():
    root_dir = '/Users/wangyu/Desktop/点云算法/第四章/data_object_velodyne/testing/velodyne'  # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]  # 略去第一个item 没有意义这里
    iteration_num = len(cat)

    # for i in range(iteration_num):
    #     # for test
    #     if i > 0:
    #         break
    filename = os.path.join(root_dir, cat[250])
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)
    segmented_points, segmented_ground = ground_segmentation(data=origin_points)
    # cluster_index = clustering(segmented_points)
    _, cluster_index = cluster.dbscan(segmented_points, 0.3, min_samples=3)
    #
    # plot_clusters(segmented_points, cluster_index)

    plot_clusters_v_1(segmented_ground, segmented_points, cluster_index)

if __name__ == '__main__':
    main()
