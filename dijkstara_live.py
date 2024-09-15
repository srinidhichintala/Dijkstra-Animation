import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

# Graph Creation
G = nx.Graph()
G.add_weighted_edges_from([(1, 2, 1), (1, 3, 4), (2, 3, 2), (2, 4, 5), (3, 4, 1)])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', width=2, font_size=15)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Dijkstra's Algorithm
def dijkstra(graph, start):
    queue, visited, min_dist = [(0, start)], set(), {start: 0}
    while queue:
        (cost, node) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node].items():
                new_cost = cost + weight
                if neighbor not in min_dist or new_cost < min_dist[neighbor]:
                    min_dist[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
    return min_dist

graph = {1: {2: 1, 3: 4}, 2: {1: 1, 3: 2, 4: 5}, 3: {1: 4, 2: 2, 4: 1}, 4: {2: 5, 3: 1}}
shortest_paths = dijkstra(graph, 1)
print(shortest_paths)

# Visualization
fig, ax = plt.subplots()
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', width=2, font_size=15)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

path = [1, 2, 3, 4]  # Example path
x_data, y_data = [], []
line, = plt.plot([], [], 'ro-', markersize=10)

def init():
    line.set_data([], [])
    return line,

def update(num):
    x_data.append(pos[path[num]][0])
    y_data.append(pos[path[num]][1])
    line.set_data(x_data, y_data)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, interval=1000)
plt.show()
