//
//  UGraph.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include "UGraph.hpp"
#include <iostream>


auto UGraph::add_edge(Edge edge) -> bool {
    if (edge.a > edge.b) {
        std::swap(edge.a, edge.b);
    }
    edges.push_back(edge);
    return true;
}

auto UGraph::clean_duplicate_edges() -> void {
    std::sort(edges.begin(), edges.end(), [] (const auto &e1, const auto &e2) {
        return e1.a < e2.a ? true :
            e1.a == e2.a ? e1.b < e2.b :
            false;
    });
    auto last = std::unique(edges.begin(), edges.end(), [] (const auto &e1, const auto &e2) {
        return e1.a == e2.a && e1.b == e2.b;
    });
    edges.erase(last, edges.end());
}

