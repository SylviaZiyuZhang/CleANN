// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "common.h"
#include "index.h"
#include "parameters.h"

namespace py = pybind11;

namespace diskannpy
{

template <typename DT>
class DynamicMemoryIndex
{
  public:
    DynamicMemoryIndex(diskann::Metric m, size_t dimensions, size_t max_vectors, uint32_t complexity, uint32_t insert_complexity,
                       uint32_t graph_degree, uint32_t bridge_start_lb, uint32_t bridge_start_hb, uint32_t bridge_end_lb, uint32_t bridge_end_hb, float bridge_prob, uint32_t cleaning_threshold,
                       bool saturate_graph, uint32_t max_occlusion_size, float alpha,
                       uint32_t num_threads, uint32_t filter_complexity, uint32_t num_frozen_points,
                       uint32_t initial_search_complexity, uint32_t initial_search_threads,
                       bool concurrent_consolidation);

    void load(const std::string &index_path);
    void build(const py::array_t<DT, py::array::c_style | py::array::forcecast> &data,
      const size_t num_points_to_load,
      const py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &tags);
    int insert(const py::array_t<DT, py::array::c_style | py::array::forcecast> &vector, DynamicIdType id);
    py::array_t<int> batch_insert(py::array_t<DT, py::array::c_style | py::array::forcecast> &vectors,
                                  py::array_t<DynamicIdType, py::array::c_style | py::array::forcecast> &ids, int32_t num_inserts,
                                  int num_threads = 0);
    int mark_deleted(DynamicIdType id);
    void save(const std::string &save_path, bool compact_before_save = false);
    void save_graph_synchronized(const std::string &file_name);
    void compare_with_alt_graph(const std::string &alt_file_name);
    void save_edge_analytics(const std::string &save_path);
    void set_start_points_at_random(DT radius, uint32_t random_seed);
    NeighborsAndDistances<DynamicIdType> search(py::array_t<DT, py::array::c_style | py::array::forcecast> &query, uint64_t knn,
                                      uint64_t complexity, const bool improvement_allowed);
    NeighborsAndDistances<DynamicIdType> batch_search(py::array_t<DT, py::array::c_style | py::array::forcecast> &queries,
                                            uint64_t num_queries, uint64_t knn, uint64_t complexity,
                                            uint32_t num_threads);
    void consolidate_delete();
    size_t num_points();
    void print_status();


  private:
    const uint32_t _initial_search_complexity;
    const diskann::IndexWriteParameters _write_parameters;

  public:
    diskann::Index<DT, DynamicIdType, filterT> _index;
};

}; // namespace diskannpy