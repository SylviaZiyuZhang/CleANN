// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#include "parameters.h"

BOOST_AUTO_TEST_SUITE(IndexWriteParametersBuilder_tests)

BOOST_AUTO_TEST_CASE(test_build)
{
    uint32_t search_list_size = rand();
    uint32_t insert_list_size = rand();
    uint32_t max_degree = rand();
    uint32_t bridge_start_lb = rand();
    uint32_t bridge_start_hb = rand();
    uint32_t bridge_end_lb = rand();
    uint32_t bridge_end_hb = rand();
    uint32_t bridge_prob = (float)rand();
    uint32_t cleaning_threshold = (rand() % max_degree) + 1;
    float alpha = (float)rand();
    uint32_t filter_list_size = rand();
    uint32_t max_occlusion_size = rand();
    bool saturate_graph = true;

    diskann::IndexWriteParametersBuilder builder(search_list_size, insert_list_size, max_degree,
        bridge_start_lb, bridge_start_hb, bridge_end_lb, bridge_end_hb, bridge_prob, cleaning_threshold);

    builder.with_alpha(alpha)
        .with_filter_list_size(filter_list_size)
        .with_max_occlusion_size(max_occlusion_size)
        .with_num_threads(0)
        .with_saturate_graph(saturate_graph);

    {
        auto parameters = builder.build();

        BOOST_TEST(search_list_size == parameters.search_list_size);
        BOOST_TEST(insert_list_size == parameters.insert_list_size);
        BOOST_TEST(bridge_start_lb == parameters.bridge_start_lb);
        BOOST_TEST(bridge_start_hb == parameters.bridge_start_hb);
        BOOST_TEST(bridge_end_lb == parameters.bridge_end_lb);
        BOOST_TEST(bridge_end_hb == parameters.bridge_end_hb);
        BOOST_TEST(bridge_prob == parameters.bridge_prob);
        BOOST_TEST(cleaning_threshold == parameters.cleaning_threshold);
        BOOST_TEST(max_degree == parameters.max_degree);
        BOOST_TEST(alpha == parameters.alpha);
        BOOST_TEST(filter_list_size == parameters.filter_list_size);
        BOOST_TEST(max_occlusion_size == parameters.max_occlusion_size);
        BOOST_TEST(saturate_graph == parameters.saturate_graph);

        BOOST_TEST(parameters.num_threads > (uint32_t)0);
    }

    {
        uint32_t num_threads = rand() + 1;
        saturate_graph = false;
        builder.with_num_threads(num_threads).with_saturate_graph(saturate_graph);

        auto parameters = builder.build();

        BOOST_TEST(search_list_size == parameters.search_list_size);
        BOOST_TEST(insert_list_size == parameters.insert_list_size);
        BOOST_TEST(bridge_start_lb == parameters.bridge_start_lb);
        BOOST_TEST(bridge_start_hb == parameters.bridge_start_hb);
        BOOST_TEST(bridge_end_lb == parameters.bridge_end_lb);
        BOOST_TEST(bridge_end_hb == parameters.bridge_end_hb);
        BOOST_TEST(bridge_prob == parameters.bridge_prob);
        BOOST_TEST(cleaning_threshold == parameters.cleaning_threshold);
        BOOST_TEST(max_degree == parameters.max_degree);
        BOOST_TEST(alpha == parameters.alpha);
        BOOST_TEST(filter_list_size == parameters.filter_list_size);
        BOOST_TEST(max_occlusion_size == parameters.max_occlusion_size);
        BOOST_TEST(saturate_graph == parameters.saturate_graph);

        BOOST_TEST(num_threads == parameters.num_threads);
    }
}

BOOST_AUTO_TEST_SUITE_END()
