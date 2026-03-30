#ifndef VOXELMAP_HPP
#define VOXELMAP_HPP

#include "common_lib.h"
#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define HASH_P 116101
#define MAX_N 10000000000

// 3D point with covariance
typedef struct PointWithCov {
  Eigen::Vector3d point;
  Eigen::Vector3d point_world;
  Eigen::Matrix3d cov;
} PointWithCov;

typedef struct PointPlane {
  Eigen::Vector3d point;
  Eigen::Vector3d normal;
  Eigen::Vector3d center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
} PointPlane;

typedef struct Plane {
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Vector3d x_normal;
  Eigen::Vector3d y_normal;
  Eigen::Matrix3d covariance;
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d = 0;
  int points_size = 0;

  bool is_plane = false;
  bool is_init = false;
  int id;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;
  bool update_enable = true;
} Plane;

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

class OctoTree {
public:
  std::vector<PointWithCov> temp_points_; // all points in an octo tree
  std::vector<PointWithCov> new_points_;  // new points in an octo tree
  Plane *plane_ptr_;

  int max_layer_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  int max_cov_points_size_;
  bool init_octo_;
  bool update_cov_enable_;
  bool update_enable_;

  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size, int max_cov_points_size, float planer_threshold)
      : max_layer_(max_layer), layer_(layer),
        layer_point_size_(layer_point_size), max_points_size_(max_point_size),
        max_cov_points_size_(max_cov_points_size),
        planer_threshold_(planer_threshold) {
    plane_ptr_ = new Plane();

    temp_points_.clear();
    octo_state_ = 0;
    new_points_num_ = 0;
    all_points_num_ = 0;

    // when new points num > 5, do a update
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_cov_enable_ = true;
    update_enable_ = true;
    max_plane_update_threshold_ = layer_point_size_[layer_];

    for (int i = 0; i < 8; i++)
      leaves_[i] = nullptr;
  }

  void InitPlane(const std::vector<PointWithCov> &points, Plane *plane) {
    plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->points_size = points.size();
    plane->radius = 0;
    for (auto pv : points) {
      plane->covariance += pv.point * pv.point.transpose();
      plane->center += pv.point;
    }
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size -
                        plane->center * plane->center.transpose();

    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evals_real;
    evals_real = evals.real();
    Eigen::Matrix3f::Index evals_min, evals_max;
    evals_real.rowwise().sum().minCoeff(&evals_min);
    evals_real.rowwise().sum().maxCoeff(&evals_max);
    int evals_mid = 3 - evals_min - evals_max;
    Eigen::Vector3d evec_min = evecs.real().col(evals_min);
    Eigen::Vector3d evec_mid = evecs.real().col(evals_mid);
    Eigen::Vector3d evec_max = evecs.real().col(evals_max);

    // plane covariance calculation
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
        1.0 / plane->points_size;

    if (evals_real(evals_min) < planer_threshold_) {
      std::vector<int> index(points.size());
      std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
      for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F;
        for (int m = 0; m < 3; m++) {
          if (m != (int)evals_min) {
            Eigen::Matrix<double, 1, 3> F_m =
                (points[i].point - plane->center).transpose() /
                ((plane->points_size) *
                 (evals_real[evals_min] - evals_real[m])) *
                (evecs.real().col(m) * evecs.real().col(evals_min).transpose() +
                 evecs.real().col(evals_min) * evecs.real().col(m).transpose());
            F.row(m) = F_m;
          } else {
            Eigen::Matrix<double, 1, 3> F_m;
            F_m << 0, 0, 0;
            F.row(m) = F_m;
          }
        }
        J.block<3, 3>(0, 0) = evecs.real() * F;
        J.block<3, 3>(3, 0) = J_Q;
        plane->plane_cov += J * points[i].cov * J.transpose();
      }

      plane->normal << evecs.real()(0, evals_min), evecs.real()(1, evals_min),
          evecs.real()(2, evals_min);
      plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid),
          evecs.real()(2, evals_mid);
      plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max),
          evecs.real()(2, evals_max);
      plane->min_eigen_value = evals_real(evals_min);
      plane->mid_eigen_value = evals_real(evals_mid);
      plane->max_eigen_value = evals_real(evals_max);
      plane->radius = sqrt(evals_real(evals_max));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = true;
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }

      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
    } else {
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      plane->is_plane = false;
      plane->normal << evecs.real()(0, evals_min), evecs.real()(1, evals_min),
          evecs.real()(2, evals_min);
      plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid),
          evecs.real()(2, evals_mid);
      plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max),
          evecs.real()(2, evals_max);
      plane->min_eigen_value = evals_real(evals_min);
      plane->mid_eigen_value = evals_real(evals_mid);
      plane->max_eigen_value = evals_real(evals_max);
      plane->radius = sqrt(evals_real(evals_max));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
    }
  }

  void UpdatePlane(const std::vector<PointWithCov> &points, Plane *plane) {
    Eigen::Matrix3d old_covariance = plane->covariance;
    Eigen::Vector3d old_center = plane->center;
    Eigen::Matrix3d sum_ppt =
        (plane->covariance + plane->center * plane->center.transpose()) *
        plane->points_size;
    Eigen::Vector3d sum_p = plane->center * plane->points_size;
    for (size_t i = 0; i < points.size(); i++) {
      Eigen::Vector3d pv = points[i].point;
      sum_ppt += pv * pv.transpose();
      sum_p += pv;
    }
    plane->points_size = plane->points_size + points.size();
    plane->center = sum_p / plane->points_size;
    plane->covariance = sum_ppt / plane->points_size -
                        plane->center * plane->center.transpose();

    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evals_real;
    evals_real = evals.real();
    Eigen::Matrix3d::Index evals_min, evals_max;
    evals_real.rowwise().sum().minCoeff(&evals_min);
    evals_real.rowwise().sum().maxCoeff(&evals_max);
    int evals_mid = 3 - evals_min - evals_max;
    Eigen::Vector3d evec_min = evecs.real().col(evals_min);
    Eigen::Vector3d evec_mid = evecs.real().col(evals_mid);
    Eigen::Vector3d evec_max = evecs.real().col(evals_max);
    if (evals_real(evals_min) < planer_threshold_) {
      plane->normal << evecs.real()(0, evals_min), evecs.real()(1, evals_min),
          evecs.real()(2, evals_min);
      plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid),
          evecs.real()(2, evals_mid);
      plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max),
          evecs.real()(2, evals_max);
      plane->min_eigen_value = evals_real(evals_min);
      plane->mid_eigen_value = evals_real(evals_mid);
      plane->max_eigen_value = evals_real(evals_max);
      plane->radius = sqrt(evals_real(evals_max));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));

      plane->is_plane = true;
      plane->is_update = true;
    } else {
      plane->normal << evecs.real()(0, evals_min), evecs.real()(1, evals_min),
          evecs.real()(2, evals_min);
      plane->y_normal << evecs.real()(0, evals_mid), evecs.real()(1, evals_mid),
          evecs.real()(2, evals_mid);
      plane->x_normal << evecs.real()(0, evals_max), evecs.real()(1, evals_max),
          evecs.real()(2, evals_max);
      plane->min_eigen_value = evals_real(evals_min);
      plane->mid_eigen_value = evals_real(evals_mid);
      plane->max_eigen_value = evals_real(evals_max);
      plane->radius = sqrt(evals_real(evals_max));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = false;
      plane->is_update = true;
    }
  }

  void InitOctoTree() {
    if (temp_points_.size() > max_plane_update_threshold_) {
      InitPlane(temp_points_, plane_ptr_);
      if (plane_ptr_->is_plane == true) {
        octo_state_ = 0;
        if (temp_points_.size() > max_cov_points_size_) {
          update_cov_enable_ = false;
        }
        if (temp_points_.size() > max_points_size_) {
          update_enable_ = false;
        }
      } else {
        octo_state_ = 1;
        CutOctoTree();
      }
      init_octo_ = true;
      new_points_num_ = 0;
      //      temp_points_.clear();
    }
  }

  void CutOctoTree() {
    if (layer_ >= max_layer_) {
      octo_state_ = 0;
      return;
    }
    for (size_t i = 0; i < temp_points_.size(); i++) {
      int xyz[3] = {0, 0, 0};
      if (temp_points_[i].point[0] > voxel_center_[0]) {
        xyz[0] = 1;
      }
      if (temp_points_[i].point[1] > voxel_center_[1]) {
        xyz[1] = 1;
      }
      if (temp_points_[i].point[2] > voxel_center_[2]) {
        xyz[2] = 1;
      }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(
            max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
            max_cov_points_size_, planer_threshold_);
        leaves_[leafnum]->voxel_center_[0] =
            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      }
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_num_++;
    }
    for (uint i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        if (leaves_[i]->temp_points_.size() >
            leaves_[i]->max_plane_update_threshold_) {
          init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
          if (leaves_[i]->plane_ptr_->is_plane) {
            leaves_[i]->octo_state_ = 0;
          } else {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_num_ = 0;
        }
      }
    }
  }

  void UpdateOctoTree(const PointWithCov &pv) {
    if (!init_octo_) {
      new_points_num_++;
      all_points_num_++;
      temp_points_.push_back(pv);
      if (temp_points_.size() > max_plane_update_threshold_) {
        InitOctoTree();
      }
    } else {
      if (plane_ptr_->is_plane) {
        if (update_enable_) {
          new_points_num_++;
          all_points_num_++;
          if (update_cov_enable_) {
            temp_points_.push_back(pv);
          } else {
            new_points_.push_back(pv);
          }
          if (new_points_num_ > update_size_threshold_) {
            if (update_cov_enable_) {
              InitPlane(temp_points_, plane_ptr_);
            }
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_cov_points_size_) {
            update_cov_enable_ = false;
            std::vector<PointWithCov>().swap(temp_points_);
          }
          if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
            plane_ptr_->update_enable = false;
            std::vector<PointWithCov>().swap(new_points_);
          }
        } else {
          return;
        }
      } else {
        if (layer_ < max_layer_) {
          if (temp_points_.size() != 0)
            std::vector<PointWithCov>().swap(temp_points_);

          if (new_points_.size() != 0)
            std::vector<PointWithCov>().swap(new_points_);

          int xyz[3] = {0, 0, 0};
          if (pv.point[0] > voxel_center_[0])
            xyz[0] = 1;

          if (pv.point[1] > voxel_center_[1])
            xyz[1] = 1;

          if (pv.point[2] > voxel_center_[2])
            xyz[2] = 1;

          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          if (leaves_[leafnum] != nullptr) {
            leaves_[leafnum]->UpdateOctoTree(pv);
          } else {
            leaves_[leafnum] = new OctoTree(
                max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                max_cov_points_size_, planer_threshold_);
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] =
                voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] =
                voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] =
                voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            leaves_[leafnum]->UpdateOctoTree(pv);
          }
        } else {
          if (update_enable_) {
            new_points_num_++;
            all_points_num_++;
            if (update_cov_enable_) {
              temp_points_.push_back(pv);
            } else {
              new_points_.push_back(pv);
            }
            if (new_points_num_ > update_size_threshold_) {
              if (update_cov_enable_) {
                init_plane(temp_points_, plane_ptr_);
              } else {
                update_plane(new_points_, plane_ptr_);
                new_points_.clear();
              }
              new_points_num_ = 0;
            }
            if (all_points_num_ >= max_cov_points_size_) {
              update_cov_enable_ = false;
              std::vector<PointWithCov>().swap(temp_points_);
            }
            if (all_points_num_ >= max_points_size_) {
              update_enable_ = false;
              plane_ptr_->update_enable = false;
              std::vector<PointWithCov>().swap(new_points_);
            }
          }
        }
      }
    }
  }
};

void BuildVoxelMap(const std::vector<PointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const PointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }

  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->InitOctoTree();
  }
}

void UpdateVoxelMap(const std::vector<PointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    const PointWithCov p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.point[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->UpdateOctoTree(p_v);
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
}

void BuildSingleResidual(const PointWithCov &pv, const OctoTree *current_octo,
                         const int current_layer, const int max_layer,
                         const double sigma_num, bool &is_sucess, double &prob,
                         PointPlane &single_ptpl) {
  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_world;

  if (current_octo->plane_ptr_->is_plane) {
    Plane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center;

    double proj_x = p_world_to_center.dot(plane.x_normal);
    double proj_y = p_world_to_center.dot(plane.y_normal);
    float dis_to_plane =
        fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
             plane.normal(2) * p_w(2) + plane.d);
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
        }
        return;
      } else {
        // is_sucess = false;
        return;
      }
    } else {
      // is_sucess = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {

          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          BuildSingleResidual(pv, leaf_octo, current_layer + 1, max_layer,
                              sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_sucess = false;
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<PointWithCov> &pv_list,
                          std::vector<PointPlane> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }

#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); ++i) {
    PointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.point_world[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      PointPlane single_ptpl;
      bool is_sucess = false;
      double prob = 0;
      BuildSingleResidual(pv, current_octo, 0, max_layer, sigma_num, is_sucess,
                          prob, single_ptpl);
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        if (loc_xyz[0] >
            (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
          near_position.x = near_position.x + 1;
        } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                 current_octo->quater_length_)) {
          near_position.x = near_position.x - 1;
        }
        if (loc_xyz[1] >
            (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
          near_position.y = near_position.y + 1;
        } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                 current_octo->quater_length_)) {
          near_position.y = near_position.y - 1;
        }
        if (loc_xyz[2] >
            (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
          near_position.z = near_position.z + 1;
        } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                 current_octo->quater_length_)) {
          near_position.z = near_position.z - 1;
        }
        auto iter_near = voxel_map.find(near_position);
        if (iter_near != voxel_map.end()) {
          BuildSingleResidual(pv, iter_near->second, 0, max_layer, sigma_num,
                              is_sucess, prob, single_ptpl);
        }
      }
      if (is_sucess) {

        mylock.lock();
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        useful_ptpl[i] = false;
        mylock.unlock();
      }
    }
  }

  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

#endif // VOXELMAP_HPP