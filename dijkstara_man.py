import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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

# Visualization with Image
fig, ax = plt.subplots()
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', width=2, font_size=15)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

path = [1, 2, 3, 4]  # Example path

# Load the image
img = mpimg.imread('running.png')

def update(num):
    ax.clear()
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black', width=2, font_size=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if num < len(path):
        x, y = pos[path[num]]
        imagebox = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    return ax,

ani = animation.FuncAnimation(fig, update, frames=len(path), blit=False, interval=1000)
plt.show()
