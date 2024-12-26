// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "abstract_graph_store.h"
#include "locking.h"

namespace diskann
{

class InMemGraphStore : public AbstractGraphStore
{
  public:
    InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree);

    // returns tuple of <nodes_read, start, num_frozen_points>
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                        const size_t num_points) override;
    virtual int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                      const uint32_t start) override;

    virtual const std::vector<location_t> &get_neighbours(const location_t i) const override;
    virtual const tsl::robin_set<location_t> &get_in_neighbours(const location_t i) const override;
    virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
    virtual void clear_neighbours(const location_t i) override;
    virtual void swap_neighbours(const location_t a, location_t b) override;

    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) override;

    virtual size_t resize_graph(const size_t new_size) override;
    virtual void clear_graph() override;

    virtual size_t get_max_range_of_graph() override;
    virtual uint32_t get_max_observed_degree() override;

    virtual size_t get_edge_count() override;

    // dynamic consolidation
    virtual bool is_tombstoned(const location_t i) override;
    virtual int get_num_consolidates(const location_t i) override;
    virtual void record_consolidate(const location_t i) override;
    virtual void mark_live(const location_t i) override;
    virtual void mark_tombstoned(const location_t i) override;
    virtual void swap_tombstone_record(const location_t a, const location_t b) override;

  protected:
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);
#ifdef EXEC_ENV_OLS
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(AlignedFileReader &reader, size_t expected_num_points);
#endif

    // TODO (SylviaZiyuZhang): modify save_graph to support deleted node delegation
    int save_graph(const std::string &index_path_prefix, const size_t active_points, const size_t num_frozen_points,
                   const uint32_t start);

  private:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;

    std::vector<std::vector<uint32_t>> _graph;
    std::vector<tsl::robin_set<uint32_t>> _reverse_graph;
    std::vector<non_recursive_mutex> _reverse_graph_locks; 
    // -1 in _consolidate_hits indicates that the point is live
    // if the point is tombstoned, records the number of times it has been
    // fixed on-the-fly
    std::vector<int> _consolidate_hits;
};

} // namespace diskann
