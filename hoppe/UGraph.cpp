//
//  UGraph.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include "UGraph.hpp"
#include <iostream>


auto UGraph::add_edge(Edge edge) -> bool {
    auto iterator = edges.begin();
    const auto end = edges.end();
    if (edge.a > edge.b) {
        std::swap(edge.a, edge.b);
    }
    assert(edge.a != edge.b);
    while (iterator != end) {
        const auto &e = *iterator;
        if (edge.a == e.a && edge.b == e.b) {
            return false;
        }
        if (e.cost > edge.cost) {
            break;
        }
        iterator++;
    }
    edges.insert(iterator, edge);
    return true;
}
