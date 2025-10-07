import heapq

class ShortestPathSolver:
    def __init__(self):
        """初始化最短路径求解器"""
        pass
    
    def dijkstra(self, graph, start):
        """
        Dijkstra算法：求解从起点到所有其他节点的最短路径
        适用于：带非负权重的有向图或无向图，单源最短路径
        """
        # 初始化距离字典，起点距离为0，其他为无穷大
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        
        # 优先队列，存储(距离, 节点)，按距离排序
        priority_queue = [(0, start)]
        
        # 记录路径
        path = {node: [] for node in graph}
        path[start] = [start]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # 如果当前距离大于已知最短距离，跳过
            if current_distance > distances[current_node]:
                continue
                
            # 遍历邻居节点
            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                
                # 如果找到更短的路径
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    path[neighbor] = path[current_node] + [neighbor]
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        return distances, path
    
    def floyd(self, graph, node_list):
        """
        Floyd算法：求解所有节点对之间的最短路径
        适用于：带权图（可含负权边，但不能有负权回路），全源最短路径
        """
        n = len(node_list)
        node_index = {node: i for i, node in enumerate(node_list)}
        
        # 初始化距离矩阵
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        # 初始化路径矩阵
        next_node = [[None] * n for _ in range(n)]
        
        # 填充初始距离
        for u in graph:
            for v, weight in graph[u].items():
                i, j = node_index[u], node_index[v]
                dist[i][j] = weight
                next_node[i][j] = v
        
        # Floyd核心算法
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
        
        # 构建路径字典
        path = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    path[(node_list[i], node_list[j])] = [node_list[i]]
                    continue
                    
                current = node_list[i]
                path[(node_list[i], node_list[j])] = [current]
                while current != node_list[j]:
                    current = next_node[i][node_index[current]]
                    if current is None:  # 无路径
                        path[(node_list[i], node_list[j])] = None
                        break
                    path[(node_list[i], node_list[j])].append(current)
        
        # 构建距离字典
        distance_dict = {}
        for i in range(n):
            for j in range(n):
                distance_dict[(node_list[i], node_list[j])] = dist[i][j]
        
        return distance_dict, path
    
    def _heuristic(self, a, b):
        """启发函数：曼哈顿距离，用于A*算法"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def astar(self, grid, start, goal):
        """
        A*算法：求解网格中从起点到终点的最短路径
        适用于：已知起点和终点的网格图，通过启发函数加速搜索
        """
        # 定义四个方向
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
        
        # 开放列表：存储(总代价f, 实际代价g, 当前位置)
        open_list = []
        heapq.heappush(open_list, (0, 0, start))
        
        # 记录路径
        came_from = {}
        
        # g_score：从起点到当前位置的实际代价
        g_score = {start: 0}
        
        # f_score：预估总代价 = g_score + 启发函数
        f_score = {start: self._heuristic(start, goal)}
        
        while open_list:
            _, current_g, current = heapq.heappop(open_list)
            
            # 到达目标
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], current_g  # 返回路径和总代价
            
            # 探索邻居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查是否在网格范围内且不是障碍物(0表示可通行，1表示障碍物)
                if (0 <= neighbor[0] < len(grid) and 
                    0 <= neighbor[1] < len(grid[0]) and 
                    grid[neighbor[0]][neighbor[1]] == 0):
                    
                    # 计算临时g值
                    tentative_g_score = current_g + 1  # 假设每步代价为1
                    
                    # 如果该邻居未被访问过，或找到更优路径
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))
        
        # 如果没有找到路径
        return None, float('inf')


# 测试代码
if __name__ == "__main__":
    solver = ShortestPathSolver()
    
    # 1. 测试Dijkstra算法
    print("=" * 50)
    print("Dijkstra算法测试")
    graph = {
        'A': {'B': 4, 'C': 2},
        'B': {'C': 5, 'D': 10},
        'C': {'D': 3},
        'D': {}
    }
    distances, paths = solver.dijkstra(graph, 'A')
    print("从A到各节点的最短距离:", distances)
    print("从A到各节点的路径:", paths)
    
    # 2. 测试Floyd算法
    print("\n" + "=" * 50)
    print("Floyd算法测试")
    node_list = ['A', 'B', 'C', 'D']
    dist_dict, path_dict = solver.floyd(graph, node_list)
    print("所有节点对之间的最短距离:")
    for (u, v), d in dist_dict.items():
        print(f"{u}到{v}的距离: {d}")
    print("\n所有节点对之间的路径:")
    for (u, v), p in path_dict.items():
        print(f"{u}到{v}的路径: {p}")
    
    # 3. 测试A*算法
    print("\n" + "=" * 50)
    print("A*算法测试")
    # 网格图：0表示可通行，1表示障碍物
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (4, 4)
    path, cost = solver.astar(grid, start, goal)
    print(f"从{start}到{goal}的路径: {path}")
    print(f"路径总代价: {cost}")
