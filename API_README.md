# VoxelMap API 说明

本文档说明 `voxel_map_util_refactored.hpp` 中各核心 API 的用途、调用流程及注意事项，
对应的调用点在 `voxelMapping_refactored.cpp` 中均以 `// [VoxelMap]` 标注。

---

## 1. 核心数据结构

| 类型 | 说明 |
|---|---|
| `PointWithCovariance` | 单个点 + 其在 world 系下的 3×3 协方差矩阵 |
| `PointToPlaneMatch` | 点-面匹配结果：点坐标、法向量、面中心、面协方差、距离 d、所在层 |
| `Plane` | 拟合平面：normal/center/radius/min_eigen_value/plane_cov 等 |
| `VoxelLocation` | 体素的三维整数坐标 (x, y, z)，作为哈希表 key |
| `OctoTree` | 单个体素的八叉树，维护点云、拟合平面、子树，支持增量更新 |
| `std::unordered_map<VoxelLocation, OctoTree*>` | **主地图**：所有体素的集合 |

> **注意**：`OctoTree` 通过裸指针管理，调用方负责生命周期；地图析构时需手动 delete。

---

## 2. API 一览

### 2.1 CalcBodyCovariance
```cpp
void CalcBodyCovariance(Eigen::Vector3d &pb,
                        const float range_inc,
                        const float degree_inc,
                        Eigen::Matrix3d &cov);
```
**作用**：根据激光测距误差 `range_inc` 和角度误差 `degree_inc`，
计算点 `pb`（body 系）的 3×3 协方差矩阵 `cov`。

**注意**：
- `pb[2] == 0` 时分母为零，调用前需检查并替换为一个小值（如 `0.001`）。
- 输出的 `cov` 是 **body 系**下的协方差，变换到 world 系需额外叠加旋转不确定度：
  ```cpp
  cov_world = R * cov_body * R^T
            + (-p×) * rot_cov * (-p×)^T
            + t_cov;
  ```

---

### 2.2 TransformLidar
```cpp
void TransformLidar(const StatesGroup &state,
                    const shared_ptr<ImuProcess> &p_imu,
                    const PointCloudXYZI::Ptr &input_cloud,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);
```
**作用**：将点云从 body 系（LiDAR 坐标系）变换到 world 系，使用当前 `state.rot_end` 和 `state.pos_end`。

**注意**：
- 每次 EKF 迭代状态更新后需重新调用，以得到与当前状态一致的 world 系点云。
- 地图初始化、EKF 迭代匹配、地图增量更新三个阶段均需独立调用。

---

### 2.3 BuildVoxelMap
```cpp
void BuildVoxelMap(const std::vector<PointWithCovariance> &input_points,
                   const float voxel_size,
                   const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size,
                   const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VoxelLocation, OctoTree*> &feat_map);
```
**作用**：将首帧点云分配到各体素，为每个体素创建 `OctoTree` 并触发初始平面拟合。

**参数说明**：
| 参数 | 说明 |
|---|---|
| `voxel_size` | 体素边长（米），影响地图分辨率 |
| `max_layer` | OctoTree 最大层数，超过则强制设为叶节点 |
| `layer_point_size` | 各层触发平面拟合的最少点数（长度须 = max_layer+1） |
| `max_points_size` | 单体素最大点数，达到后停止插入新点 |
| `max_cov_points_size` | 维护协方差的最大点数，达到后停止更新协方差 |
| `planer_threshold` | 平面判断阈值：最小特征值 < 该值则认为是平面 |

**注意**：
- 仅在**首帧**（`init_map == false`）时调用一次。
- 调用前须确保 `input_points` 的协方差已变换到 **world 系**。

---

### 2.4 BuildResidualListOMP
```cpp
void BuildResidualListOMP(
    const unordered_map<VoxelLocation, OctoTree*> &voxel_map,
    const double voxel_size,
    const double sigma_num,
    const int max_layer,
    const std::vector<PointWithCovariance> &pv_list,
    std::vector<PointToPlaneMatch> &ptpl_list,
    std::vector<Eigen::Vector3d> &non_match);
```
**作用**：OpenMP 并行地对每个待匹配点，在 `voxel_map` 中搜索最近平面，
输出通过 3σ 检验的点-面匹配对。

**匹配判断条件**：
```
dis_to_plane < sigma_num * sqrt(sigma_l)
其中 sigma_l = J_nq * plane_cov * J_nq^T + n^T * point_cov * n
     J_nq = [p_w - center, -normal]  (1×6 Jacobian)
```

**跨体素回退搜索**：若当前体素无匹配，自动检查相邻体素（±1 格）。

**注意**：
- `pv_list` 中的 `point` 须为 **body 系**，`point_world` 须为当前状态下的 **world 系**。
- `sigma_num` 建议取 3.0（3σ 准则），过大会引入误匹配，过小会降低有效点数。
- 若不使用 OpenMP，可替换为 `BuildResidualListNormal`（单线程版本）。

---

### 2.5 UpdateVoxelMap
```cpp
void UpdateVoxelMap(const std::vector<PointWithCovariance> &input_points,
                    const float voxel_size,
                    const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size,
                    const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VoxelLocation, OctoTree*> &feat_map);
```
**作用**：将当前帧新点增量插入地图，内部调用 `OctoTree::UpdateOctoTree`，
点数达到阈值后触发 `InitPlane` 或 `UpdatePlane` 重新拟合平面。

**OctoTree 内部更新策略**：

```
已初始化且是平面 → 每累积 update_size_threshold 个新点 → InitPlane/UpdatePlane
未初始化         → 积累到 max_plane_update_threshold 后 → InitOctoTree
                   若不是平面 → CutOctoTree 向子节点分裂
```

**注意**：
- 调用前建议按协方差范数**从小到大排序**（`var_contrast`），优先插入高质量点，使平面拟合更稳定。
- 使用 EKF **收敛后的最终状态**（`state.rot_end/pos_end`）变换点云，而非迭代中间状态。
- 当 `all_points_num >= max_cov_points_size` 时，该体素不再更新协方差（`update_cov_enable_ = false`）。
- 当 `all_points_num >= max_points_size` 时，该体素停止接受新点（`update_enable_ = false`）。

---

### 2.6 PubVoxelMap / PubPlaneMap
```cpp
void PubVoxelMap(const std::unordered_map<VoxelLocation, OctoTree*> &voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub);

void PubPlaneMap(const std::unordered_map<VoxelLocation, OctoTree*> &feat_map,
                 const ros::Publisher &plane_map_pub);
```
**作用**：以 RViz `Cylinder Marker` 形式发布地图中的平面，颜色由平面协方差的迹映射（Jet 色图）。

**注意**：
- `PubVoxelMap` 仅发布 `is_update == true` 的平面（本帧新增/更新），效率更高，适合实时发布。
- `PubPlaneMap` 发布所有层（含子层）的平面，适合调试。
- `pub_max_voxel_layer` 控制发布的最大树深，0 表示只发布根层平面。

---

## 3. 主循环调用流程

```
每帧点云到来
│
├─ [Phase 1] 地图初始化（仅首帧，init_map == false）
│   ├─ TransformLidar(feats_undistort)          → world_lidar
│   ├─ 逐点 CalcBodyCovariance + 叠加状态协方差  → pv_list (world 系协方差)
│   └─ BuildVoxelMap(pv_list)                   → voxel_map
│
└─ [Phase 2 + 3] 正常帧（init_map == true）
    │
    ├─ [EKF 迭代，共 NUM_MAX_ITERATIONS 次]
    │   ├─ TransformLidar(feats_down_body)           → world_lidar（当前估计）
    │   ├─ 逐点叠加状态协方差                         → pv_list (world 系协方差)
    │   ├─ BuildResidualListOMP(voxel_map, pv_list)  → ptpl_list（点-面匹配）
    │   ├─ 逐匹配点 CalcBodyCovariance + J_nq        → R_inv（观测噪声）
    │   ├─ 构建 Hsub、求解 EKF 更新量
    │   └─ state += solution → 收敛则退出迭代
    │
    └─ [Phase 3] 地图增量更新
        ├─ TransformLidar(feats_down_body)       → world_lidar（最终状态）
        ├─ 逐点叠加状态协方差 → pv_list，按 var_contrast 排序
        ├─ UpdateVoxelMap(pv_list)               → 更新 voxel_map
        └─ PubVoxelMap(voxel_map)                → RViz 可视化（可选）
```

---

## 4. 参数配置建议

| 参数 | 推荐范围 | 说明 |
|---|---|---|
| `voxel_size` | 0.5 ~ 2.0 m | 室内取小值，室外取大值 |
| `max_layer` | 2 ~ 3 | 层数越多细节越丰富，但计算量增加 |
| `min_eigen_value` (planer_threshold) | 0.003 ~ 0.01 | 越小要求平面越严格 |
| `max_points_size` | 50 ~ 200 | 单体素上限，影响内存 |
| `max_cov_points_size` | 50 ~ 100 | 一般设为 max_points_size 的一半 |
| `sigma_num` | 2.5 ~ 3.0 | 匹配阈值，室内可适当减小 |
| `layer_point_size` | `[5, 5, 5, ...]` | 各层最少点数，长度 = max_layer+1 |

---

## 5. 常见问题

**Q: `CalcBodyCovariance` 崩溃或结果异常？**
A: 检查点的 z 分量是否为 0，函数内部假设点不在 xy 平面上。

**Q: 有效匹配点数（effct_feat_num）过少？**
A: 可能原因：① `sigma_num` 太小；② `voxel_size` 过大导致平面太粗；
③ `min_eigen_value` 太小导致太少平面被识别；④ 初始状态偏差大。

**Q: 地图更新后平面不稳定？**
A: 建议增大 `max_cov_points_size`，或在插入前按 `var_contrast` 排序以优先插入低噪声点。

**Q: 内存持续增长？**
A: `OctoTree` 通过裸指针存储，程序退出时需遍历 `voxel_map` 并 delete 所有 `OctoTree*`。
当前实现未提供自动析构，长时间运行需注意。
