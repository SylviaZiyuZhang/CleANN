// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <string>

#include "common.h"
#include "distance.h"

namespace diskannpy
{
template <typename DT>
void build_disk_index(diskann::Metric metric, const std::string &data_file_path, const std::string &index_prefix_path,
                      uint32_t complexity, uint32_t insert_complexity, uint32_t graph_degree, 
                      uint32_t bridge_start_lb, uint32_t bridge_start_hb, uint32_t bridge_end_lb, uint32_t bridge_end_hb, float bridge_prob, uint32_t cleaning_threshold, double final_index_ram_limit,
                      double indexing_ram_budget, uint32_t num_threads, uint32_t pq_disk_bytes);

template <typename DT, typename TagT = DynamicIdType, typename LabelT = filterT>
void build_memory_index(diskann::Metric metric, const std::string &vector_bin_path,
                           const std::string &index_output_path, uint32_t graph_degree, uint32_t complexity, uint32_t insert_complexity,
                           float alpha, uint32_t bridge_start_lb, uint32_t bridge_start_hb, uint32_t bridge_end_lb, uint32_t bridge_end_hb, float bridge_prob, uint32_t cleaning_threshold,
                           uint32_t num_threads, bool use_pq_build,
                           size_t num_pq_bytes, bool use_opq, bool use_tags = false,
                           const std::string& filter_labels_file = "", const std::string& universal_label = "",
                           uint32_t filter_complexity = 0);

}
