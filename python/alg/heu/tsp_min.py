import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

# -------------------------- 数据结构定义 --------------------------
@dataclass
class Location:
    id: int
    x: float
    y: float
    terrain: str  # 新增地形类型

@dataclass
class Supply:
    id: int
    size: int
    criticality: int
    penalty_per_day: int

@dataclass
class Staff:
    id: int
    home_location: int
    max_distance: float
    max_patients: int
    skills: Dict[str, bool]
    work_days: List[int] = None  # 记录工作天数，用于休息约束

@dataclass
class Request:
    id: int
    location_id: int
    day_start: int
    day_end: int
    supply_id: int
    quantity: int
    specialty_needed: str
    delivered_day: Optional[int] = None
    served: bool = False

@dataclass
class VehicleRoute:
    vehicle_id: int
    locations: List[int]  # 配送位置顺序（含仓库往返）
    requests: List[int]
    total_distance: float = 0.0
    cost: float = 0.0  # 单路线成本

@dataclass
class StaffAssignment:
    staff_id: int
    locations: List[int]  # 服务位置顺序（含家往返）
    requests: List[int]
    total_distance: float = 0.0
    cost: float = 0.0  # 单分配成本

# -------------------------- 全局参数与数据加载 --------------------------
class Config:
    # 车辆参数
    VEHICLE_CAPACITY = 10
    VEHICLE_MAX_DISTANCE = 300
    VEHICLE_DISTANCE_COST = 50
    VEHICLE_DAY_COST = 200
    VEHICLE_FIXED_COST = 10000
    VEHICLE_FIXED_WEIGHT = 1.2  # 固定成本权重提升
    
    # 人员参数
    STAFF_DISTANCE_COST = 20
    STAFF_DAY_COST = 150
    STAFF_FIXED_COST = 5000
    
    # 算法参数（优化后）
    POPULATION_SIZE = 100  # 增大种群多样性
    MAX_GENERATIONS = 200  # 延长进化周期
    CROSSOVER_RATE = 0.7  # 降低交叉率，保护优质解
    MUTATION_RATE = 0.08  # 提高变异率，增加探索
    TOURNAMENT_SIZE = 3  # 减小锦标赛规模，增加选择压力
    ELITE_RATE = 0.1  # 提高精英保留比例
    DAYS = 10
    LOCAL_SEARCH_RATE = 0.3  # 局部搜索概率

class DataLoader:
    @staticmethod
    def load_locations() -> Dict[int, Location]:
        """加载位置数据，新增地形类型"""
        locations = {
            1: Location(1, 0, 0, "urban"),      # 中央仓库
            2: Location(2, 30, 40, "urban"),    # Urban Zone A
            3: Location(3, 50, 20, "rural"),    # Rural Outpost B
            4: Location(4, 10, 70, "urban"),    # Urban Clinic C
            5: Location(5, 60, 60, "remote"),   # Remote Site D
            6: Location(6, 80, 10, "mountain"), # Mountain Clinic（高阻力）
            7: Location(7, 20, 80, "rural"),    # Emergency Camp A
            8: Location(8, 40, 50, "urban")     # Field Hospital E
        }
        return locations

    @staticmethod
    def load_supplies() -> Dict[int, Supply]:
        supplies = {
            1: Supply(1, 2, 3, 300),
            2: Supply(2, 1, 2, 200),
            3: Supply(3, 3, 4, 500),
            4: Supply(4, 2, 1, 100)
        }
        return supplies

    @staticmethod
    def load_staff(locations: Dict[int, Location]) -> Dict[int, Staff]:
        staff = {
            1: Staff(
                id=1, home_location=4, max_distance=120, max_patients=3,
                skills={"Surgery": True, "Pediatrics": False, "Trauma": True, "General": True},
                work_days=[]
            ),
            2: Staff(
                id=2, home_location=5, max_distance=100, max_patients=2,
                skills={"Surgery": False, "Pediatrics": True, "Trauma": True, "General": False},
                work_days=[]
            ),
            3: Staff(
                id=3, home_location=6, max_distance=150, max_patients=2,
                skills={"Surgery": True, "Pediatrics": True, "Trauma": False, "General": True},
                work_days=[]
            ),
            4: Staff(
                id=4, home_location=1, max_distance=200, max_patients=3,
                skills={"Surgery": True, "Pediatrics": True, "Trauma": True, "General": True},
                work_days=[]
            ),
            5: Staff(
                id=5, home_location=7, max_distance=80, max_patients=1,
                skills={"Surgery": False, "Pediatrics": False, "Trauma": True, "General": True},
                work_days=[]
            )
        }
        return staff

    @staticmethod
    def load_requests() -> List[Request]:
        requests = [
            Request(1, 2, 1, 3, 1, 2, "Surgery"),
            Request(2, 3, 1, 2, 2, 1, "Pediatrics"),
            Request(3, 4, 2, 4, 3, 2, "Trauma"),
            Request(4, 5, 2, 3, 1, 1, "Surgery"),
            Request(5, 6, 3, 5, 4, 2, "General"),
            Request(6, 7, 4, 6, 3, 1, "Trauma"),
            Request(7, 8, 1, 3, 2, 3, "Pediatrics"),
            Request(8, 3, 5, 6, 1, 2, "Surgery"),
            Request(9, 5, 6, 7, 4, 1, "General"),
            Request(10, 6, 7, 9, 3, 2, "Trauma"),
            Request(11, 7, 2, 4, 1, 1, "Surgery"),
            Request(12, 8, 3, 5, 2, 2, "Pediatrics")
        ]
        return requests

# -------------------------- 遗传算法核心实现 --------------------------
class Chromosome:
    def __init__(self, days: int):
        self.days = days
        self.vehicle_schedules: Dict[int, List[VehicleRoute]] = {d: [] for d in range(1, days+1)}
        self.staff_schedules: Dict[int, List[StaffAssignment]] = {d: [] for d in range(1, days+1)}
        self.fitness = 0.0
        self.total_cost = 0.0
        self.request_status: Dict[int, bool] = {r.id: False for r in DataLoader.load_requests()}
        self.resource_utilization = 0.0  # 新增：资源利用率（多目标优化）

class GeneticAlgorithm:
    def __init__(self):
        self.locations = DataLoader.load_locations()
        self.supplies = DataLoader.load_supplies()
        self.staff = DataLoader.load_staff(self.locations)
        self.requests = DataLoader.load_requests()
        self.config = Config()
        self.request_supply_map = {r.id: self.supplies[r.supply_id] for r in self.requests}
        self.population: List[Chromosome] = []

    def calculate_distance(self, loc1_id: int, loc2_id: int) -> float:
        """地形感知距离计算（附加功能：地形路由）"""
        loc1 = self.locations[loc1_id]
        loc2 = self.locations[loc2_id]
        base_dist = math.hypot(loc1.x - loc2.x, loc1.y - loc2.y)
        # 地形阻力系数：山地×1.5，偏远×1.2，其他×1.0
        terrain_factor = 1.0
        if loc1.terrain == "mountain" or loc2.terrain == "mountain":
            terrain_factor = 1.5
        elif loc1.terrain == "remote" or loc2.terrain == "remote":
            terrain_factor = 1.2
        return base_dist * terrain_factor

    def initialize_population(self) -> None:
        """贪心初始化：按时间窗+位置聚类合并请求"""
        for _ in range(self.config.POPULATION_SIZE):
            chromosome = Chromosome(self.config.DAYS)
            self.initialize_chromosome(chromosome)
            self.evaluate_fitness(chromosome)
            self.population.append(chromosome)

    def initialize_chromosome(self, chromosome: Chromosome) -> None:
        """改进初始化策略：减少冗余车辆，优化路线合并"""
        requests = [r for r in DataLoader.load_requests()]
        unassigned_requests = [r.id for r in requests]
        
        # 1. 车辆配送初始化：按时间窗分组+位置聚类
        for day in range(1, self.config.DAYS + 1):
            # 筛选当日可配送请求，按criticality排序（优先高紧急度）
            available = [
                r_id for r_id in unassigned_requests
                if requests[r_id-1].day_start <= day <= requests[r_id-1].day_end
            ]
            available_sorted = sorted(
                available, 
                key=lambda x: self.request_supply_map[x].criticality, 
                reverse=True
            )
            if not available_sorted:
                continue
            
            # 位置聚类：将距离近的请求合并（减少车辆使用）
            clusters = self.cluster_requests(available_sorted, requests)
            vehicle_id = 1
            for cluster in clusters:
                # 检查容量约束
                total_size = sum(
                    self.request_supply_map[r_id].size * requests[r_id-1].quantity
                    for r_id in cluster
                )
                if total_size > self.config.VEHICLE_CAPACITY:
                    # 容量超限则拆分聚类
                    sub_clusters = self.split_cluster(cluster, requests, total_size)
                    for sub in sub_clusters:
                        self.add_vehicle_route(chromosome, day, vehicle_id, sub, requests)
                        vehicle_id += 1
                else:
                    self.add_vehicle_route(chromosome, day, vehicle_id, cluster, requests)
                    vehicle_id += 1
                # 标记已分配请求
                for r_id in cluster:
                    unassigned_requests.remove(r_id)
        
        # 2. 人员调度初始化：技能匹配+距离最短
        self.initialize_staff_schedule(chromosome, requests)

    def cluster_requests(self, request_ids: List[int], requests: List[Request]) -> List[List[int]]:
        """将距离近的请求聚类，减少车辆路线"""
        clusters = []
        unclustered = request_ids.copy()
        while unclustered:
            center = unclustered[0]
            center_loc = requests[center-1].location_id
            cluster = [center]
            unclustered.remove(center)
            # 加入距离中心50km内的请求
            for r_id in unclustered.copy():
                r_loc = requests[r_id-1].location_id
                if self.calculate_distance(center_loc, r_loc) < 50:
                    cluster.append(r_id)
                    unclustered.remove(r_id)
            clusters.append(cluster)
        return clusters

    def split_cluster(self, cluster: List[int], requests: List[Request], total_size: int) -> List[List[int]]:
        """容量超限的聚类拆分"""
        sub_clusters = []
        current = []
        current_size = 0
        for r_id in cluster:
            r_size = self.request_supply_map[r_id].size * requests[r_id-1].quantity
            if current_size + r_size <= self.config.VEHICLE_CAPACITY:
                current.append(r_id)
                current_size += r_size
            else:
                sub_clusters.append(current)
                current = [r_id]
                current_size = r_size
        sub_clusters.append(current)
        return sub_clusters

    def add_vehicle_route(self, chromosome: Chromosome, day: int, vehicle_id: int, 
                         request_ids: List[int], requests: List[Request]) -> None:
        """生成车辆路线并优化顺序（最短路径）"""
        locations = [requests[r_id-1].location_id for r_id in request_ids]
        # 优化路线顺序：仓库→最近点→次近点...→仓库（贪心路径）
        route = self.optimize_route(locations)
        # 计算总距离
        total_distance = 0.0
        for i in range(len(route)-1):
            total_distance += self.calculate_distance(route[i], route[i+1])
        # 更新染色体
        chromosome.vehicle_schedules[day].append(
            VehicleRoute(
                vehicle_id=vehicle_id,
                locations=[loc for loc in route if loc != 1],  # 排除仓库
                requests=request_ids,
                total_distance=total_distance
            )
        )
        # 标记配送日
        for r_id in request_ids:
            requests[r_id-1].delivered_day = day

    def optimize_route(self, locations: List[int]) -> List[int]:
        """贪心优化路线顺序：从仓库出发，每次去最近未访问点"""
        if not locations:
            return [1, 1]  # 仓库往返
        route = [1]  # 起点：仓库
        unvisited = locations.copy()
        while unvisited:
            last = route[-1]
            # 找最近未访问点
            nearest = min(unvisited, key=lambda loc: self.calculate_distance(last, loc))
            route.append(nearest)
            unvisited.remove(nearest)
        route.append(1)  # 终点：仓库
        return route

    def initialize_staff_schedule(self, chromosome: Chromosome, requests: List[Request]) -> None:
        """人员调度初始化：技能匹配+距离最短"""
        for day in range(1, self.config.DAYS + 1):
            # 筛选可服务请求（已配送且未服务）
            available = [
                r.id for r in requests
                if r.delivered_day is not None and r.delivered_day <= day - 1 and not chromosome.request_status[r.id]
            ]
            if not available:
                continue
            
            # 按技能分组请求
            skill_groups = {}
            for r_id in available:
                skill = requests[r_id-1].specialty_needed
                if skill not in skill_groups:
                    skill_groups[skill] = []
                skill_groups[skill].append(r_id)
            
            # 为每组请求分配匹配技能的人员
            for skill, r_ids in skill_groups.items():
                # 筛选有对应技能的人员
                valid_staff = [
                    sid for sid, staff in self.staff.items()
                    if staff.skills.get(skill, False)
                ]
                if not valid_staff:
                    continue
                
                # 按距离最短分配
                for r_id in r_ids:
                    r_loc = requests[r_id-1].location_id
                    # 找距离请求位置最近的人员
                    best_staff = min(
                        valid_staff,
                        key=lambda sid: self.calculate_distance(self.staff[sid].home_location, r_loc)
                    )
                    # 分配请求给人员
                    self.assign_staff_request(chromosome, day, best_staff, r_id, requests)

    def assign_staff_request(self, chromosome: Chromosome, day: int, staff_id: int, 
                            r_id: int, requests: List[Request]) -> None:
        """为人员分配请求并更新行程"""
        staff = self.staff[staff_id]
        r_loc = requests[r_id-1].location_id
        # 检查人员当日是否已分配
        day_assignments = chromosome.staff_schedules[day]
        existing = next((a for a in day_assignments if a.staff_id == staff_id), None)
        
        if existing:
            # 添加到现有分配
            existing.requests.append(r_id)
            existing.locations.append(r_loc)
        else:
            # 创建新分配
            existing = StaffAssignment(
                staff_id=staff_id,
                locations=[r_loc],
                requests=[r_id]
            )
            day_assignments.append(existing)
        
        # 优化人员路线并计算距离
        home = staff.home_location
        route = [home] + existing.locations + [home]
        total_distance = sum(
            self.calculate_distance(route[i], route[i+1]) 
            for i in range(len(route)-1)
        )
        existing.total_distance = total_distance
        chromosome.request_status[r_id] = True

    def evaluate_fitness(self, chromosome: Chromosome) -> None:
        """多目标适应度评估：总成本+资源利用率（附加功能：多目标优化）"""
        vehicle_cost = self.calculate_vehicle_cost(chromosome)
        staff_cost = self.calculate_staff_cost(chromosome)
        delay_penalty = self.calculate_delay_penalty(chromosome)
        constraint_penalty = self.calculate_constraint_penalty(chromosome)
        
        # 总成本
        chromosome.total_cost = vehicle_cost + staff_cost + delay_penalty + constraint_penalty
        # 资源利用率：(总请求数/总资源使用量)，越高越好
        total_requests = len(self.requests)
        unique_vehicles = len({r.vehicle_id for s in chromosome.vehicle_schedules.values() for r in s})
        unique_staff = len({a.staff_id for s in chromosome.staff_schedules.values() for a in s})
        chromosome.resource_utilization = total_requests / (unique_vehicles + unique_staff + 1)  # +1避免除零
        
        # 多目标适应度：加权求和（总成本权重更高）
        chromosome.fitness = (1.0 / (1.0 + chromosome.total_cost)) * 0.7 + \
                             (chromosome.resource_utilization / total_requests) * 0.3

    def calculate_vehicle_cost(self, chromosome: Chromosome) -> float:
        """车辆成本计算：增加固定成本权重"""
        unique_vehicles = set()
        total_distance = 0.0
        vehicle_days = 0
        
        for day_schedules in chromosome.vehicle_schedules.values():
            for route in day_schedules:
                unique_vehicles.add(route.vehicle_id)
                total_distance += route.total_distance
                vehicle_days += 1
        
        # 固定成本加权（激励减少车辆数量）
        fixed_cost = len(unique_vehicles) * self.config.VEHICLE_FIXED_COST * self.config.VEHICLE_FIXED_WEIGHT
        day_cost = vehicle_days * self.config.VEHICLE_DAY_COST
        distance_cost = total_distance * self.config.VEHICLE_DISTANCE_COST
        return fixed_cost + day_cost + distance_cost

    def calculate_staff_cost(self, chromosome: Chromosome) -> float:
        unique_staff = set()
        total_distance = 0.0
        staff_days = 0
        
        for day_schedules in chromosome.staff_schedules.values():
            for assignment in day_schedules:
                unique_staff.add(assignment.staff_id)
                total_distance += assignment.total_distance
                staff_days += 1
        
        fixed_cost = len(unique_staff) * self.config.STAFF_FIXED_COST
        day_cost = staff_days * self.config.STAFF_DAY_COST
        distance_cost = total_distance * self.config.STAFF_DISTANCE_COST
        return fixed_cost + day_cost + distance_cost

    def calculate_delay_penalty(self, chromosome: Chromosome) -> float:
        total_penalty = 0.0
        requests = DataLoader.load_requests()
        
        for req in requests:
            delivered_day = None
            for day in chromosome.vehicle_schedules:
                for route in chromosome.vehicle_schedules[day]:
                    if req.id in route.requests:
                        delivered_day = day
                        break
                if delivered_day:
                    break
            
            if delivered_day is None:
                delay_days = self.config.DAYS - req.day_end
                if delay_days > 0:
                    supply = self.request_supply_map[req.id]
                    total_penalty += delay_days * supply.penalty_per_day
            else:
                is_served = req.id in chromosome.request_status and chromosome.request_status[req.id]
                if not is_served:
                    latest_service_day = min(delivered_day + 5, self.config.DAYS)
                    delay_days = self.config.DAYS - latest_service_day
                    if delay_days > 0:
                        supply = self.request_supply_map[req.id]
                        total_penalty += delay_days * supply.penalty_per_day
        return total_penalty

    def calculate_constraint_penalty(self, chromosome: Chromosome) -> float:
        penalty = 0.0
        requests = DataLoader.load_requests()
        
        # 车辆约束
        for day, routes in chromosome.vehicle_schedules.items():
            for route in routes:
                total_size = sum(
                    self.request_supply_map[r_id].size * requests[r_id-1].quantity
                    for r_id in route.requests
                )
                if total_size > self.config.VEHICLE_CAPACITY:
                    penalty += (total_size - self.config.VEHICLE_CAPACITY) * 1000
                if route.total_distance > self.config.VEHICLE_MAX_DISTANCE:
                    penalty += (route.total_distance - self.config.VEHICLE_MAX_DISTANCE) * 200
        
        # 人员约束
        staff_work_days = {sid: [] for sid in self.staff.keys()}
        for day, assignments in chromosome.staff_schedules.items():
            for assignment in assignments:
                staff = self.staff[assignment.staff_id]
                staff_work_days[assignment.staff_id].append(day)
                
                if len(assignment.requests) > staff.max_patients:
                    penalty += (len(assignment.requests) - staff.max_patients) * 500
                if assignment.total_distance > staff.max_distance:
                    penalty += (assignment.total_distance - staff.max_distance) * 200
                for r_id in assignment.requests:
                    req = requests[r_id-1]
                    if not staff.skills.get(req.specialty_needed, False):
                        penalty += 5000
        
        # 连续工作约束
        for sid, days in staff_work_days.items():
            days.sort()
            consecutive = 1
            for i in range(1, len(days)):
                if days[i] == days[i-1] + 1:
                    consecutive += 1
                    if consecutive > 5:
                        penalty += 1000
                else:
                    consecutive = 1
        
        return penalty

    def select_parent(self) -> Chromosome:
        """锦标赛选择：增加选择压力"""
        candidates = random.sample(self.population, self.config.TOURNAMENT_SIZE)
        return max(candidates, key=lambda x: x.fitness)

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """按请求聚类交叉：保护优质路线片段"""
        child1 = Chromosome(self.config.DAYS)
        child2 = Chromosome(self.config.DAYS)
        
        # 复制父代基因
        child1.vehicle_schedules = {d: routes.copy() for d, routes in parent1.vehicle_schedules.items()}
        child1.staff_schedules = {d: assns.copy() for d, assns in parent1.staff_schedules.items()}
        child1.request_status = parent1.request_status.copy()
        
        child2.vehicle_schedules = {d: routes.copy() for d, routes in parent2.vehicle_schedules.items()}
        child2.staff_schedules = {d: assns.copy() for d, assns in parent2.staff_schedules.items()}
        child2.request_status = parent2.request_status.copy()
        
        if random.random() < self.config.CROSSOVER_RATE:
            # 随机选择请求聚类作为交叉单元
            all_requests = set(r.id for r in self.requests)
            cross_requests = random.sample(list(all_requests), k=len(all_requests)//2)
            
            # 交换聚类对应的车辆路线
            for day in child1.vehicle_schedules:
                for route in child1.vehicle_schedules[day]:
                    if any(r in cross_requests for r in route.requests):
                        # 从parent2找对应请求的路线
                        for p2_route in parent2.vehicle_schedules[day]:
                            if any(r in cross_requests for r in p2_route.requests):
                                route.locations = p2_route.locations.copy()
                                route.requests = p2_route.requests.copy()
                                route.total_distance = p2_route.total_distance
                                break
            
            # 交换聚类对应的人员分配
            for day in child1.staff_schedules:
                for assn in child1.staff_schedules[day]:
                    if any(r in cross_requests for r in assn.requests):
                        for p2_assn in parent2.staff_schedules[day]:
                            if any(r in cross_requests for r in p2_assn.requests):
                                assn.locations = p2_assn.locations.copy()
                                assn.requests = p2_assn.requests.copy()
                                assn.total_distance = p2_assn.total_distance
                                break
            
            # 反向操作：child2从parent1获取交叉片段
            for day in child2.vehicle_schedules:
                for route in child2.vehicle_schedules[day]:
                    if any(r in cross_requests for r in route.requests):
                        for p1_route in parent1.vehicle_schedules[day]:
                            if any(r in cross_requests for r in p1_route.requests):
                                route.locations = p1_route.locations.copy()
                                route.requests = p1_route.requests.copy()
                                route.total_distance = p1_route.total_distance
                                break
            
            for day in child2.staff_schedules:
                for assn in child2.staff_schedules[day]:
                    if any(r in cross_requests for r in assn.requests):
                        for p1_assn in parent1.staff_schedules[day]:
                            if any(r in cross_requests for r in p1_assn.requests):
                                assn.locations = p1_assn.locations.copy()
                                assn.requests = p1_assn.requests.copy()
                                assn.total_distance = p1_assn.total_distance
                                break
            
            # 更新请求状态
            self.update_request_status(child1)
            self.update_request_status(child2)
        
        return child1, child2

    def update_request_status(self, chromosome: Chromosome) -> None:
        """更新请求服务状态"""
        for r_id in chromosome.request_status:
            chromosome.request_status[r_id] = any(
                r_id in assn.requests 
                for day_sched in chromosome.staff_schedules.values()
                for assn in day_sched
            )

    def mutate(self, chromosome: Chromosome) -> None:
        """定向变异：优先优化高成本路线"""
        requests = DataLoader.load_requests()
        high_cost_routes = []
        
        # 识别高成本车辆路线（距离超200km）
        for day in chromosome.vehicle_schedules:
            for route in chromosome.vehicle_schedules[day]:
                if route.total_distance > 200:
                    high_cost_routes.append((day, route))
        
        # 对高成本路线增加变异概率
        for day, route in high_cost_routes:
            if random.random() < self.config.MUTATION_RATE * 1.5:  # 高概率变异
                self.mutate_vehicle_route(chromosome, day, route, requests)
        
        # 普通车辆路线变异
        for day in chromosome.vehicle_schedules:
            for route in chromosome.vehicle_schedules[day]:
                if route not in [r for d, r in high_cost_routes] and random.random() < self.config.MUTATION_RATE:
                    self.mutate_vehicle_route(chromosome, day, route, requests)
        
        # 人员调度变异（优先闲置人员）
        self.mutate_staff_schedule(chromosome, requests)

    def mutate_vehicle_route(self, chromosome: Chromosome, day: int, route: VehicleRoute, requests: List[Request]) -> None:
        """车辆路线变异：调整顺序或替换请求"""
        if len(route.locations) > 1:
            # 50%概率调整顺序，50%概率替换请求
            if random.random() < 0.5:
                # 2-opt局部优化：交换两点顺序
                i, j = random.sample(range(len(route.locations)), 2)
                route.locations[i], route.locations[j] = route.locations[j], route.locations[i]
                # 同步更新请求顺序
                route.requests = [
                    r_id for loc in route.locations
                    for r_id in route.requests
                    if requests[r_id-1].location_id == loc
                ]
            else:
                # 替换部分请求（同时间窗内）
                current_requests = route.requests.copy()
                day_start = min(requests[r_id-1].day_start for r_id in current_requests)
                day_end = max(requests[r_id-1].day_end for r_id in current_requests)
                # 找同时间窗的未分配请求
                candidate_requests = [
                    r.id for r in requests
                    if r.id not in current_requests and day_start <= day <= r.day_end
                ]
                if candidate_requests:
                    # 替换1个请求
                    replace_idx = random.randint(0, len(current_requests)-1)
                    new_r_id = random.choice(candidate_requests)
                    current_requests[replace_idx] = new_r_id
                    route.requests = current_requests
                    route.locations = [requests[r_id-1].location_id for r_id in current_requests]
            
            # 重新计算距离
            full_route = [1] + route.locations + [1]
            route.total_distance = sum(
                self.calculate_distance(full_route[i], full_route[i+1])
                for i in range(len(full_route)-1)
            )

    def mutate_staff_schedule(self, chromosome: Chromosome, requests: List[Request]) -> None:
        """人员调度变异：提高闲置人员利用率"""
        # 统计人员使用天数
        staff_usage = {sid: 0 for sid in self.staff.keys()}
        for day_sched in chromosome.staff_schedules.values():
            for assn in day_sched:
                staff_usage[assn.staff_id] += 1
        # 闲置人员（使用天数少的）
        idle_staff = [sid for sid, cnt in staff_usage.items() if cnt < 3]
        
        for day in chromosome.staff_schedules:
            for assn in chromosome.staff_schedules[day]:
                if random.random() < self.config.MUTATION_RATE:
                    # 优先替换为闲置人员（技能匹配）
                    if idle_staff:
                        valid_idle = [
                            sid for sid in idle_staff
                            if all(self.staff[sid].skills.get(requests[r_id-1].specialty_needed, False) 
                                   for r_id in assn.requests)
                        ]
                        if valid_idle:
                            assn.staff_id = random.choice(valid_idle)
                    # 调整服务顺序
                    if len(assn.locations) > 1:
                        random.shuffle(assn.locations)
                        assn.requests = [
                            r_id for loc in assn.locations
                            for r_id in assn.requests
                            if requests[r_id-1].location_id == loc
                        ]
                    # 重新计算距离
                    home = self.staff[assn.staff_id].home_location
                    full_route = [home] + assn.locations + [home]
                    assn.total_distance = sum(
                        self.calculate_distance(full_route[i], full_route[i+1])
                        for i in range(len(full_route)-1)
                    )
        
        self.update_request_status(chromosome)

    def local_search(self, chromosome: Chromosome) -> None:
        """2-opt局部搜索优化路线（混合算法：附加功能1）"""
        # 优化车辆路线
        for day in chromosome.vehicle_schedules:
            for route in chromosome.vehicle_schedules[day]:
                if len(route.locations) < 2:
                    continue
                # 2-opt优化：尝试交换任意两点减少距离
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(route.locations)-1):
                        for j in range(i+1, len(route.locations)):
                            # 原顺序：i-1 → i → ... → j → j+1
                            # 新顺序：i-1 → j → ... → i → j+1
                            old_dist = self.calculate_distance(route.locations[i-1], route.locations[i]) + \
                                       self.calculate_distance(route.locations[j], route.locations[j+1] if j+1 < len(route.locations) else 1)
                            new_dist = self.calculate_distance(route.locations[i-1], route.locations[j]) + \
                                       self.calculate_distance(route.locations[i], route.locations[j+1] if j+1 < len(route.locations) else 1)
                            if new_dist < old_dist:
                                # 反转i到j的顺序
                                route.locations[i:j+1] = route.locations[i:j+1][::-1]
                                # 同步更新请求
                                route.requests = [
                                    r_id for loc in route.locations
                                    for r_id in route.requests
                                    if self.requests[r_id-1].location_id == loc
                                ]
                                # 重新计算总距离
                                full_route = [1] + route.locations + [1]
                                route.total_distance = sum(
                                    self.calculate_distance(full_route[k], full_route[k+1])
                                    for k in range(len(full_route)-1)
                                )
                                improved = True
                                break
                        if improved:
                            break
        
        # 优化人员路线
        for day in chromosome.staff_schedules:
            for assn in chromosome.staff_schedules[day]:
                if len(assn.locations) < 2:
                    continue
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(assn.locations)-1):
                        for j in range(i+1, len(assn.locations)):
                            home = self.staff[assn.staff_id].home_location
                            old_dist = self.calculate_distance(assn.locations[i-1], assn.locations[i]) + \
                                       self.calculate_distance(assn.locations[j], assn.locations[j+1] if j+1 < len(assn.locations) else home)
                            new_dist = self.calculate_distance(assn.locations[i-1], assn.locations[j]) + \
                                       self.calculate_distance(assn.locations[i], assn.locations[j+1] if j+1 < len(assn.locations) else home)
                            if new_dist < old_dist:
                                assn.locations[i:j+1] = assn.locations[i:j+1][::-1]
                                assn.requests = [
                                    r_id for loc in assn.locations
                                    for r_id in assn.requests
                                    if self.requests[r_id-1].location_id == loc
                                ]
                                full_route = [home] + assn.locations + [home]
                                assn.total_distance = sum(
                                    self.calculate_distance(full_route[k], full_route[k+1])
                                    for k in range(len(full_route)-1)
                                )
                                improved = True
                                break
                        if improved:
                            break

    def evolve(self) -> None:
        """进化主循环：增加局部搜索"""
        for gen in range(self.config.MAX_GENERATIONS):
            # 精英保留
            elite_size = int(self.config.ELITE_RATE * self.config.POPULATION_SIZE)
            elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_size]
            
            # 生成新种群
            new_population = elites.copy()
            while len(new_population) < self.config.POPULATION_SIZE:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                # 局部搜索增强（附加功能：混合算法）
                if random.random() < self.config.LOCAL_SEARCH_RATE:
                    self.local_search(child1)
                    self.local_search(child2)
                self.evaluate_fitness(child1)
                self.evaluate_fitness(child2)
                new_population.append(child1)
                if len(new_population) < self.config.POPULATION_SIZE:
                    new_population.append(child2)
            
            self.population = new_population
            
            # 打印进度
            best = max(self.population, key=lambda x: x.fitness)
            if gen % 20 == 0:
                print(f"Generation {gen}: Best Fitness = {best.fitness:.6f}, Total Cost = {best.total_cost:.2f}")

    def get_best_solution(self) -> Chromosome:
        return max(self.population, key=lambda x: x.fitness)

# -------------------------- 结果输出 --------------------------
def output_solution(ga: GeneticAlgorithm, best: Chromosome) -> None:
    print(f"Name = Your Name")
    
    total_vehicle_distance = 0.0
    vehicle_usage_days = 0
    unique_vehicles = set()
    total_staff_distance = 0.0
    staff_usage_days = 0
    unique_staff = set()
    
    for day in range(1, ga.config.DAYS + 1):
        vehicle_routes = best.vehicle_schedules[day]
        print(f"DAY = {day} NUMBER_OF_VEHICLES = {len(vehicle_routes)}")
        for route in vehicle_routes:
            print(f"{route.vehicle_id} {' '.join(map(str, route.locations))}")
            total_vehicle_distance += route.total_distance
            vehicle_usage_days += 1
            unique_vehicles.add(route.vehicle_id)
        
        staff_assignments = best.staff_schedules[day]
        print(f"NUMBER_OF_MEDICAL_STAFF = {len(staff_assignments)}")
        for assn in staff_assignments:
            print(f"{assn.staff_id} {' '.join(map(str, assn.locations))}")
            total_staff_distance += assn.total_distance
            staff_usage_days += 1
            unique_staff.add(assn.staff_id)
    
    print(f"\nSUMMARY: TOTAL_VEHICLE_DISTANCE = {total_vehicle_distance:.0f}")
    print(f"VEHICLE_USAGE_DAYS = {vehicle_usage_days}")
    print(f"UNIQUE_VEHICLES_USED = {len(unique_vehicles)}")
    print(f"TOTAL_STAFF_DISTANCE = {total_staff_distance:.0f}")
    print(f"STAFF_USAGE_DAYS = {staff_usage_days}")
    print(f"UNIQUE_STAFF_USED = {len(unique_staff)}")
    print(f"TOTAL_DELAY_PENALTIES = {ga.calculate_delay_penalty(best):.0f}")
    print(f"TOTAL_COST = {best.total_cost:.0f}")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    ga = GeneticAlgorithm()
    print("Initializing population...")
    ga.initialize_population()
    print("Starting evolution...")
    ga.evolve()
    
    best_solution = ga.get_best_solution()
    print("\nBest Solution Found:")
    output_solution(ga, best_solution)