// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "omp.h"
#include "defaults.h"

namespace diskann
{

class IndexWriteParameters

{
  public:
    const uint32_t search_list_size; // L
    const uint32_t insert_list_size; // insert L
    const uint32_t bridge_start_lb;
    const uint32_t bridge_start_hb;
    const uint32_t bridge_end_lb;
    const uint32_t bridge_end_hb;
    const float bridge_prob;
    const uint32_t cleaning_threshold;
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf

  private:
    IndexWriteParameters(const uint32_t search_list_size, const uint32_t insert_list_size, const uint32_t max_degree,
                         const uint32_t bridge_start_lb, const uint32_t bridge_start_hb, const uint32_t bridge_end_lb, const uint32_t bridge_end_hb, const float bridge_prob, const uint32_t cleaning_threshold,
                         const bool saturate_graph, const uint32_t max_occlusion_size, const float alpha,
                         const uint32_t num_threads,
                         const uint32_t filter_list_size)
        : search_list_size(search_list_size), insert_list_size(insert_list_size), max_degree(max_degree),
          bridge_start_lb(bridge_start_lb), bridge_start_hb(bridge_start_hb), bridge_end_lb(bridge_end_lb), bridge_end_hb(bridge_end_hb), bridge_prob(bridge_prob), cleaning_threshold(cleaning_threshold),
          saturate_graph(saturate_graph), max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexSearchParams
{
  public:
    IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
        : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads)
    {
    }
    const uint32_t initial_search_list_size; // search L
    const uint32_t num_search_threads;       // search threads
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t insert_list_size, // insert L
                                const uint32_t max_degree,        // R
                                const uint32_t bridge_start_lb,
                                const uint32_t bridge_start_hb,
                                const uint32_t bridge_end_lb,
                                const uint32_t bridge_end_hb,
                                const float bridge_prob,
                                const uint32_t cleaning_threshold)
        : _search_list_size(search_list_size), _insert_list_size(insert_list_size), _max_degree(max_degree),
          _bridge_start_lb(bridge_start_lb), _bridge_start_hb(bridge_start_hb), _bridge_end_lb(bridge_end_lb), _bridge_end_hb(bridge_end_hb), _bridge_prob(bridge_prob), _cleaning_threshold(cleaning_threshold)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads == 0 ? omp_get_num_procs() : num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size;
        return *this;
    }

    IndexWriteParameters build() const
    {
        return IndexWriteParameters(_search_list_size, _insert_list_size, _max_degree,
                                    _bridge_start_lb, _bridge_start_hb, _bridge_end_lb, _bridge_end_hb, _bridge_prob, _cleaning_threshold,
                                    _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_threads, _filter_list_size);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _insert_list_size(wp.insert_list_size), _max_degree(wp.max_degree),
          _bridge_start_lb(wp.bridge_start_lb), _bridge_start_hb(wp.bridge_start_hb), _bridge_end_lb(wp.bridge_end_lb), _bridge_end_hb(wp.bridge_end_hb), _bridge_prob(wp.bridge_prob), _cleaning_threshold(wp.cleaning_threshold),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _filter_list_size(wp.filter_list_size)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _insert_list_size{};
    uint32_t _bridge_start_lb{};
    uint32_t _bridge_start_hb{};
    uint32_t _bridge_end_lb{};
    uint32_t _bridge_end_hb{};
    float _bridge_prob{};
    uint32_t _cleaning_threshold{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
};

} // namespace diskann
