//
//  UGraph.cpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#include "UGraph.hpp"
#include <iostream>


auto find_root(std::vector<Subset> &subsets, int i) -> int {
    if (subsets[i].parent != i) {
        // Update the value so we don't need to repeatedly look up
        subsets[i].parent = find_root(subsets, subsets[i].parent);
    }
    return subsets[i].parent;
}

auto set_union(std::vector<Subset> &subsets, int a, int b) -> void {
    auto a_root = find_root(subsets, a),
        b_root = find_root(subsets, b);
    if (subsets[a_root].rank < subsets[b_root].rank) {
        subsets[a_root].parent = b_root;
    } else if (subsets[a_root].rank > subsets[b_root].rank) {
        subsets[b_root].parent = a_root;
    } else {
        subsets[b_root].parent = a_root;
        subsets[a_root].rank++;
    }
}

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

auto UGraph::generate_mst() -> UGraph { 
    UGraph mst(num_nodes);
    
    std::sort(edges.begin(), edges.end(), [] (const auto &e1, const auto &e2) {
        return e1.cost < e2.cost;
    });
    
    std::vector<Subset> subsets;
    for (auto i = 0; i < edges.size(); i++) {
        subsets.push_back(Subset(i));
    }
    
    auto idx = 0;
    while (mst.edges.size() < num_nodes - 1) {
        if (edges.size() == 0 || idx >= edges.size()) {
            // Failed - the graph itself is not connected
            return mst;
        }
        const auto &edge = edges[idx++];

        const auto a_root = find_root(subsets, (int) edge.a),
            b_root = find_root(subsets, (int) edge.b);
        if (a_root != b_root) {
            mst.add_edge(edge);
            set_union(subsets, a_root, b_root);
        }
    }
    return mst;
}

auto UGraph::traverse_dfs(int begin, std::function<void (int)> func) const -> void {
    std::vector<std::size_t> queue, explored;
    queue.push_back(begin);
    
    while (!queue.empty()) {
        const auto p = queue[0];
        queue.erase(queue.begin(), queue.begin() + 1);
        if (std::find(explored.begin(), explored.end(), p) != explored.end()) {
            continue;
        }
        explored.push_back(p);
        func((int) p);
        // Neighbor iterator
        const auto loc_func = [p] (const auto &edge) {
            return edge.a == p || edge.b == p;
        };
        auto nb_it = std::find_if(edges.begin(), edges.end(), loc_func);
        while (nb_it != edges.end()) {
            queue.push_back(nb_it->a == p ? nb_it->b : nb_it->a);
            nb_it = std::find_if(nb_it + 1, edges.end(), loc_func);
        }
    }
}

