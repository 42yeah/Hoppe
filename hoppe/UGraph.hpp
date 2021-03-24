//
//  UGraph.hpp
//  hoppe
//
//  Created by apple on 24/03/2021.
//

#ifndef UGraph_hpp
#define UGraph_hpp

#include <vector>


struct Edge {
    std::size_t a, b;
    float cost;
};

class UGraph {
public:
    UGraph(std::size_t num_nodes) : num_nodes(num_nodes) {}
    
    ~UGraph() = default;
    
    auto add_edge(Edge edge) -> bool;
    
    std::vector<Edge> edges;
    
    std::size_t num_nodes;
};

#endif /* UGraph_hpp */
