import numpy as np
import open3d as o3d

# 读取数据文件
points = []
with open("airplane.txt") as f:
    cnt = 0
    for line in f:
        if cnt>1000:
            break
        line_data = line.strip().split(',')
        # 从每行中提取点坐标
        x, y, z, _, _, _ = map(float, line_data)
        points.append([x, y, z])
        cnt+=1

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 将点转换成球体
spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.02).translate(point) for point in pcd.points]



# 创建一个Visualizer对象
vis = o3d.visualization.Visualizer()

# 创建一个绘制点云的Geometry对象
geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

vis.create_window()

# 将点渲染成球体
meshes = []
for i, point in enumerate(pcd.points):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)

    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.5, 0.5, 0.5])  # 设置球体的颜色为白色
    mesh_sphere.translate(point)

    # 生成旋转矩阵并转换为 4x4 变换矩阵
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    R = np.pad(R, ((0, 1), (0, 1)), mode='constant')
    R[3, 3] = 1

    # 应用变换矩阵
    mesh_sphere.transform(R)
    meshes.append(mesh_sphere)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
meshes.append(mesh_frame)

for mesh in meshes:
    vis.add_geometry(mesh)


# 启动Visualizer
vis.run()

