//
//  UGraph.hpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#ifndef UGraph_hpp
#define UGraph_hpp

#include <vector>
#include <mutex>
#include <functional>


struct Edge {
    std::size_t a, b;
    float cost;
};

class UGraph {
public:
    UGraph(std::size_t num_nodes) : num_nodes(num_nodes) {}
    
    UGraph(const UGraph &) = delete;
    
    UGraph &operator=(const UGraph &) = delete;
    
    UGraph(UGraph &&) = default;
    
    ~UGraph() = default;
    
    auto add_edge(Edge edge) -> bool;
    
    auto clean_duplicate_edges() -> void;
    
    auto generate_mst() -> UGraph;
    
    auto traverse_dfs(int begin, std::function<void(int)> func) const -> void;
    
    std::vector<Edge> edges;
    
    std::size_t num_nodes;
};

struct Subset {
    Subset(int parent) : parent(parent), rank(0) {}

    int parent, rank;
};

#endif /* UGraph_hpp */
