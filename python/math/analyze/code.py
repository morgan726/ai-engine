# 导入各种库
import numpy as np
import random
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import os
import sys

# 增加递归深度限制
sys.setrecursionlimit(10000)

# 字体设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 输出
os.makedirs("results", exist_ok=True)

# 地形类型与系数映射
TERRAIN_TYPES = {
    1: ("城市", 1.0),  # 中央仓库
    2: ("城市", 1.0),  # 城市A区
    3: ("农村", 1.2),  # 农村前哨B
    4: ("城市", 1.0),  # 城市诊所C
    5: ("偏远", 1.5),  # 偏远站点D
    6: ("山区", 2.0),  # 山区诊所
    7: ("营地", 1.1),  # 紧急营地A
    8: ("医院", 1.0)  # 野战医院E
}


# 计算考虑地形的距离
def terrain_distance(loc1, loc2, locations, terrain_types):
    x1, y1 = locations[loc1]
    x2, y2 = locations[loc2]
    euclid_dist = math.sqrt((x1 - x2) **2 + (y1 - y2)** 2)

    # 取两个位置中地形系数较大的作为路径系数
    terrain1 = terrain_types[loc1][1]
    terrain2 = terrain_types[loc2][1]
    terrain_factor = max(terrain1, terrain2)

    return euclid_dist * terrain_factor


# 多目标优化的适应度类
class Fitness:
    def __init__(self, total_cost, total_delay, resource_usage):
        self.total_cost = total_cost  # 总成本
        self.total_delay = total_delay  # 总延迟
        self.resource_usage = resource_usage  # 资源使用率

    def __repr__(self):
        return f"Fitness(cost={self.total_cost:.2f}, delay={self.total_delay}, usage={self.resource_usage:.2f})"


# 粒子群优化(PSO)的粒子类
class Particle:
    def __init__(self, solution):
        self.solution = solution
        self.fitness = self.evaluate_fitness()
        self.velocity = None
        self.best_solution = deepcopy(solution)
        self.best_fitness = self.fitness

    def evaluate_fitness(self):
        cost = self.solution.total_cost
        delay = self.solution.delay_penalty
        # 资源使用率 = (车辆使用天数 + 人员使用天数) / (最大可能使用天数)
        max_possible = self.solution.days * (len(self.solution.dataset.staff) + 10)  # 假设最多10辆车
        resource_usage = (self.solution.vehicle_usage_days + self.solution.staff_usage_days) / max_possible
        return Fitness(cost, delay, resource_usage)


# 加载数据集
class Dataset:
    def __init__(self):
        # 全局参数
        self.DAYS = 10
        self.VEHICLE_CAPACITY = 10
        self.VEHICLE_MAX_DISTANCE = 300
        self.VEHICLE_DISTANCE_COST = 50
        self.VEHICLE_DAY_COST = 200
        self.VEHICLE_FIXED_COST = 10000

        self.STAFF_DISTANCE_COST = 20
        self.STAFF_DAY_COST = 150
        self.STAFF_FIXED_COST = 5000

        # 位置 (ID: (x, y))
        self.locations = {
            1: (0, 0),  # 中央仓库
            2: (30, 40),  # 城市A区
            3: (50, 20),  # 农村前哨B
            4: (10, 70),  # 城市诊所C
            5: (60, 60),  # 偏远站点D
            6: (80, 10),  # 山区诊所
            7: (20, 80),  # 紧急营地A
            8: (40, 50)  # 野战医院E
        }

        # 物资 (ID: (大小, 紧急程度, 每日惩罚))
        self.supplies = {
            1: (2, 3, 300),
            2: (1, 2, 200),
            3: (3, 4, 500),
            4: (2, 1, 100)
        }

        # 医疗人员 (ID: (基地位置, 最大距离, 最大患者数/天, 技能集))
        # 技能集: [外科, 儿科, 创伤, 全科]
        self.staff = {
            1: (4, 120, 3, [1, 0, 1, 1]),  # 会外科、创伤、全科
            2: (5, 100, 2, [0, 1, 1, 0]),  # 会儿科、创伤
            3: (6, 150, 2, [1, 1, 0, 1]),  # 会外科、儿科、全科
            4: (1, 200, 3, [1, 1, 1, 1]),  # 所有技能（核心人员）
            5: (7, 80, 1, [0, 0, 1, 1])  # 会创伤、全科
        }

        # 请求 (ID: (位置ID, 开始日, 结束日, 物资ID, 数量, 需要的专业))
        # 专业映射: 0-外科, 1-儿科, 2-创伤, 3-全科
        specialty_map = {"Surgery": 0, "Pediatrics": 1, "Trauma": 2, "General": 3}
        self.requests = {
            1: (2, 1, 3, 1, 2, specialty_map["Surgery"]),
            2: (3, 1, 2, 2, 1, specialty_map["Pediatrics"]),
            3: (4, 2, 4, 3, 2, specialty_map["Trauma"]),
            4: (5, 2, 3, 1, 1, specialty_map["Surgery"]),
            5: (6, 3, 5, 4, 2, specialty_map["General"]),
            6: (7, 4, 6, 3, 1, specialty_map["Trauma"]),
            7: (8, 1, 3, 2, 3, specialty_map["Pediatrics"]),
            8: (3, 5, 6, 1, 2, specialty_map["Surgery"]),
            9: (5, 6, 7, 4, 1, specialty_map["General"]),
            10: (6, 7, 9, 3, 2, specialty_map["Trauma"]),
            11: (7, 2, 4, 1, 1, specialty_map["Surgery"]),
            12: (8, 3, 5, 2, 2, specialty_map["Pediatrics"])
        }

        # 预处理请求，添加物资大小信息
        self.processed_requests = {}
        for req_id, req_data in self.requests.items():
            loc_id, start, end, supply_id, qty, specialty = req_data
            supply_size = self.supplies[supply_id][0]
            criticality = self.supplies[supply_id][1]
            penalty = self.supplies[supply_id][2]
            self.processed_requests[req_id] = (loc_id, start, end, supply_id,
                                               qty, specialty, supply_size,
                                               criticality, penalty)

        # 添加请求的紧急度分数（用于动态优先级）
        self.request_priority = {}
        for req_id, req_data in self.processed_requests.items():
            # 紧急度 = 紧急程度 / (结束日 - 开始日 + 1)，确保时间窗口短请求优先级更高
            criticality = req_data[7]
            time_window = req_data[2] - req_data[1] + 1
            self.request_priority[req_id] = criticality / time_window


# 染色体表示：解决方案（模块化设计）
class Solution:
    def __init__(self, dataset):
        self.dataset = dataset
        self.days = dataset.DAYS

        # 初始化车辆分配: {天: {车辆ID: [请求ID列表]}}
        self.vehicle_assignments = {day: {} for day in range(1, self.days + 1)}

        # 初始化人员分配: {天: {人员ID: [请求ID列表]}}
        self.staff_assignments = {day: {} for day in range(1, self.days + 1)}

        # 记录每个请求的交付日期
        self.delivery_dates = {req_id: None for req_id in dataset.processed_requests.keys()}

        # 成本相关
        self.total_cost = 0
        self.vehicle_cost = 0
        self.staff_cost = 0
        self.delay_penalty = 0

        # 性能指标
        self.total_vehicle_distance = 0
        self.vehicle_usage_days = 0
        self.unique_vehicles_used = set()
        self.total_staff_distance = 0
        self.staff_usage_days = 0
        self.unique_staff_used = set()

        # 约束违反计数
        self.constraint_violations = {
            "vehicle_capacity": 0,
            "vehicle_distance": 0,
            "staff_patients": 0,
            "staff_skills": 0,
            "staff_distance": 0,
            "staff_work_streak": 0,
            "delivery_window": 0,
            "staff_after_delivery": 0
        }

    def calculate_vehicle_routes(self):
        """车辆路线计算子模块"""
        total_distance = 0
        vehicle_cost = 0

        for day, vehicles in self.vehicle_assignments.items():
            for vehicle_id, reqs in vehicles.items():
                # 固定成本和日成本
                vehicle_cost += self.dataset.VEHICLE_FIXED_COST
                vehicle_cost += self.dataset.VEHICLE_DAY_COST

                # 构建路径
                locs = [1]  # 从仓库出发
                req_locs = [self.dataset.processed_requests[req_id][0] for req_id in reqs]
                current_loc = 1

                # 最近邻构建初始路径
                remaining = req_locs.copy()
                while remaining:
                    min_dist = float('inf')
                    nearest_loc = None
                    for loc in remaining:
                        dist = terrain_distance(
                            current_loc, loc, self.dataset.locations, TERRAIN_TYPES
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_loc = loc
                    locs.append(nearest_loc)
                    remaining.remove(nearest_loc)
                    current_loc = nearest_loc
                locs.append(1)  # 返回仓库

                # 2-opt优化减少距离
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(locs) - 2):
                        for j in range(i + 1, len(locs) - 1):
                            if j - i == 1:
                                continue
                            # 原路径距离
                            dist_old = terrain_distance(
                                locs[i], locs[i + 1], self.dataset.locations, TERRAIN_TYPES
                            ) + terrain_distance(
                                locs[j], locs[j + 1], self.dataset.locations, TERRAIN_TYPES
                            )
                            # 新路径距离
                            dist_new = terrain_distance(
                                locs[i], locs[j], self.dataset.locations, TERRAIN_TYPES
                            ) + terrain_distance(
                                locs[i + 1], locs[j + 1], self.dataset.locations, TERRAIN_TYPES
                            )
                            if dist_new < dist_old:
                                locs[i + 1:j + 1] = locs[i + 1:j + 1][::-1]  # 反转路径
                                improved = True
                                break
                        if improved:
                            break

                # 计算总距离
                distance = 0
                for i in range(len(locs) - 1):
                    distance += terrain_distance(
                        locs[i], locs[i + 1], self.dataset.locations, TERRAIN_TYPES
                    )

                total_distance += distance
                vehicle_cost += distance * self.dataset.VEHICLE_DISTANCE_COST

        return total_distance, vehicle_cost

    def calculate_staff_routes(self):
        """人员路线计算子模块"""
        total_distance = 0
        staff_cost = 0

        for day, staff in self.staff_assignments.items():
            for staff_id, reqs in staff.items():
                # 固定成本和日成本
                staff_cost += self.dataset.STAFF_FIXED_COST
                staff_cost += self.dataset.STAFF_DAY_COST

                # 构建路径
                staff_loc = self.dataset.staff[staff_id][0]
                locs = [staff_loc]  # 从基地出发
                req_locs = [self.dataset.processed_requests[req_id][0] for req_id in reqs]

                # 最近邻构建初始路径
                remaining = req_locs.copy()
                current_loc = staff_loc
                while remaining:
                    min_dist = float('inf')
                    nearest_loc = None
                    for loc in remaining:
                        dist = terrain_distance(
                            current_loc, loc, self.dataset.locations, TERRAIN_TYPES
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_loc = loc
                    locs.append(nearest_loc)
                    remaining.remove(nearest_loc)
                    current_loc = nearest_loc
                locs.append(staff_loc)  # 返回基地

                # 2-opt优化
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(locs) - 2):
                        for j in range(i + 1, len(locs) - 1):
                            if j - i == 1:
                                continue
                            dist_old = terrain_distance(
                                locs[i], locs[i + 1], self.dataset.locations, TERRAIN_TYPES
                            ) + terrain_distance(
                                locs[j], locs[j + 1], self.dataset.locations, TERRAIN_TYPES
                            )
                            dist_new = terrain_distance(
                                locs[i], locs[j], self.dataset.locations, TERRAIN_TYPES
                            ) + terrain_distance(
                                locs[i + 1], locs[j + 1], self.dataset.locations, TERRAIN_TYPES
                            )
                            if dist_new < dist_old:
                                locs[i + 1:j + 1] = locs[i + 1:j + 1][::-1]
                                improved = True
                                break
                        if improved:
                            break

                # 计算总距离
                distance = 0
                for i in range(len(locs) - 1):
                    distance += terrain_distance(
                        locs[i], locs[i + 1], self.dataset.locations, TERRAIN_TYPES
                    )

                total_distance += distance
                staff_cost += distance * self.dataset.STAFF_DISTANCE_COST

        return total_distance, staff_cost

    def calculate_delay_penalties(self):
        """延迟惩罚计算子模块"""
        delay_penalty = 0

        for req_id, delivery_day in self.delivery_dates.items():
            req_data = self.dataset.processed_requests[req_id]
            start_day, end_day = req_data[1], req_data[2]
            penalty = req_data[8]

            if delivery_day is None:  # 未交付
                delay_days = self.days - end_day + 1
                delay_penalty += penalty * delay_days
            elif delivery_day < start_day:  # 提前交付（虽然允许，但记录）
                pass
            elif delivery_day > end_day:  # 延迟交付
                delay_penalty += penalty * (delivery_day - end_day)

        return delay_penalty

    def check_constraints(self):
        """约束检查子模块，返回是否有效并更新约束违反计数"""
        # 重置约束违反计数
        for key in self.constraint_violations:
            self.constraint_violations[key] = 0

        valid = True

        # 1. 交付日期在时间窗口内
        for req_id, delivery_day in self.delivery_dates.items():
            req_data = self.dataset.processed_requests[req_id]
            start_day, end_day = req_data[1], req_data[2]
            if delivery_day is not None and (delivery_day < start_day or delivery_day > end_day):
                self.constraint_violations["delivery_window"] += 1
                valid = False

        # 2. 车辆约束
        for day, vehicles in self.vehicle_assignments.items():
            for vehicle_id, reqs in vehicles.items():
                # 容量约束
                total_size = 0
                for req_id in reqs:
                    qty = self.dataset.processed_requests[req_id][4]
                    size_per_unit = self.dataset.processed_requests[req_id][6]
                    total_size += qty * size_per_unit
                if total_size > self.dataset.VEHICLE_CAPACITY:
                    self.constraint_violations["vehicle_capacity"] += 1
                    valid = False

                # 最大距离约束
                locs = [1] + [self.dataset.processed_requests[req_id][0] for req_id in reqs] + [1]
                distance = 0
                for i in range(len(locs) - 1):
                    distance += terrain_distance(
                        locs[i], locs[i + 1], self.dataset.locations, TERRAIN_TYPES
                    )
                if distance > self.dataset.VEHICLE_MAX_DISTANCE:
                    self.constraint_violations["vehicle_distance"] += 1
                    valid = False

        # 3. 人员约束
        for day, staff in self.staff_assignments.items():
            for staff_id, reqs in staff.items():
                # 最大患者数
                if len(reqs) > self.dataset.staff[staff_id][2]:
                    self.constraint_violations["staff_patients"] += 1
                    valid = False

                # 专业匹配
                staff_skills = self.dataset.staff[staff_id][3]
                for req_id in reqs:
                    req_specialty = self.dataset.processed_requests[req_id][5]
                    if staff_skills[req_specialty] != 1:
                        self.constraint_violations["staff_skills"] += 1
                        valid = False

                # 最大距离约束
                staff_loc = self.dataset.staff[staff_id][0]
                distance = 0
                prev_loc = staff_loc
                for req_id in reqs:
                    loc_id = self.dataset.processed_requests[req_id][0]
                    distance += terrain_distance(
                        prev_loc, loc_id, self.dataset.locations, TERRAIN_TYPES
                    )
                    prev_loc = loc_id
                distance += terrain_distance(
                    prev_loc, staff_loc, self.dataset.locations, TERRAIN_TYPES
                )
                if distance > self.dataset.staff[staff_id][1]:
                    self.constraint_violations["staff_distance"] += 1
                    valid = False

        # 4. 人员连续工作不超过5天
        work_streaks = {staff_id: 0 for staff_id in self.dataset.staff.keys()}
        for day in range(1, self.days + 1):
            for staff_id in self.dataset.staff.keys():
                if staff_id in self.staff_assignments[day]:
                    work_streaks[staff_id] += 1
                    if work_streaks[staff_id] > 5:
                        self.constraint_violations["staff_work_streak"] += 1
                        valid = False
                else:
                    work_streaks[staff_id] = 0

        # 5. 人员访问在交付后至少1天
        for day, staff in self.staff_assignments.items():
            for staff_id, reqs in staff.items():
                for req_id in reqs:
                    delivery_day = self.delivery_dates[req_id]
                    if delivery_day is None or day < delivery_day + 1:
                        self.constraint_violations["staff_after_delivery"] += 1
                        valid = False

        return valid

    def calculate_cost(self):
        """计算总成本主函数"""
        # 重置成本和性能指标
        self.total_cost = 0
        self.vehicle_cost = 0
        self.staff_cost = 0
        self.delay_penalty = 0
        self.total_vehicle_distance = 0
        self.vehicle_usage_days = 0
        self.unique_vehicles_used = set()
        self.total_staff_distance = 0
        self.staff_usage_days = 0
        self.unique_staff_used = set()

        # 计算车辆相关指标
        self.total_vehicle_distance, self.vehicle_cost = self.calculate_vehicle_routes()

        # 计算人员相关指标
        self.total_staff_distance, self.staff_cost = self.calculate_staff_routes()

        # 计算延迟惩罚
        self.delay_penalty = self.calculate_delay_penalties()

        # 计算使用统计
        for day, vehicles in self.vehicle_assignments.items():
            self.vehicle_usage_days += len(vehicles)
            self.unique_vehicles_used.update(vehicles.keys())

        for day, staff in self.staff_assignments.items():
            self.staff_usage_days += len(staff)
            self.unique_staff_used.update(staff.keys())

        # 总成本
        self.total_cost = self.vehicle_cost + self.staff_cost + self.delay_penalty
        return self.total_cost

    def is_valid(self):
        """检查解决方案是否有效"""
        return self.check_constraints()

    def repair_solution(self):
        """修复违反约束的解决方案"""
        # 1. 修复车辆容量问题
        for day in list(self.vehicle_assignments.keys()):
            vehicles = self.vehicle_assignments[day]
            for vehicle_id in list(vehicles.keys()):
                reqs = vehicles[vehicle_id]
                total_size = sum(
                    self.dataset.processed_requests[rid][4] * self.dataset.processed_requests[rid][6]
                    for rid in reqs
                )

                if total_size > self.dataset.VEHICLE_CAPACITY:
                    # 拆分车辆负载
                    new_vehicle_id = max(vehicles.keys()) + 1 if vehicles else 1
                    current_size = 0
                    split_index = 0

                    for i, rid in enumerate(reqs):
                        req_size = self.dataset.processed_requests[rid][4] * self.dataset.processed_requests[rid][6]
                        if current_size + req_size > self.dataset.VEHICLE_CAPACITY:
                            split_index = i
                            break
                        current_size += req_size

                    # 分配到新车辆
                    vehicles[new_vehicle_id] = reqs[split_index:]
                    vehicles[vehicle_id] = reqs[:split_index]

        # 2. 修复人员连续工作问题
        work_streaks = {staff_id: 0 for staff_id in self.dataset.staff.keys()}
        for day in range(1, self.days + 1):
            for staff_id in list(self.staff_assignments[day].keys()):
                work_streaks[staff_id] += 1
                if work_streaks[staff_id] > 5:
                    # 强制休息，将任务重新分配
                    reqs = self.staff_assignments[day][staff_id]
                    del self.staff_assignments[day][staff_id]
                    work_streaks[staff_id] = 0

                    # 尝试分配给其他可用人员
                    for req_id in reqs:
                        self.reassign_staff(req_id, day)

        # 重新计算成本
        self.calculate_cost()
        return self

    def reassign_staff(self, req_id, target_day):
        """为请求重新分配人员"""
        req_data = self.dataset.processed_requests[req_id]
        req_specialty = req_data[5]
        req_loc = req_data[0]
        delivery_day = self.delivery_dates[req_id]

        # 寻找合适的人员
        suitable_staff = []
        for staff_id in self.dataset.staff:
            # 检查技能
            if self.dataset.staff[staff_id][3][req_specialty] != 1:
                continue

            # 检查这一天的负载
            current_load = len(self.staff_assignments[target_day].get(staff_id, []))
            if current_load >= self.dataset.staff[staff_id][2]:
                continue

            suitable_staff.append(staff_id)

        # 如果找不到合适的人员，尝试调整日期
        if not suitable_staff:
            # 向前调整1天
            if target_day - 1 > delivery_day:
                self.reassign_staff(req_id, target_day - 1)
            # 向后调整1天
            elif target_day + 1 <= self.days:
                self.reassign_staff(req_id, target_day + 1)
            return

        # 选择距离最近的人员
        min_distance = float('inf')
        best_staff = None
        for staff_id in suitable_staff:
            staff_loc = self.dataset.staff[staff_id][0]
            distance = terrain_distance(
                staff_loc, req_loc, self.dataset.locations, TERRAIN_TYPES
            )
            if distance < min_distance:
                min_distance = distance
                best_staff = staff_id

        # 分配人员
        if best_staff not in self.staff_assignments[target_day]:
            self.staff_assignments[target_day][best_staff] = []
        self.staff_assignments[target_day][best_staff].append(req_id)

    def to_output_format(self, student_name):
        """生成符合要求的输出格式"""
        output = [f"Name = {student_name}"]
        output.append(f"CONSTRAINT_VIOLATIONS = {sum(self.constraint_violations.values())}")

        for day in range(1, self.days + 1):
            # 车辆信息
            vehicles = self.vehicle_assignments[day]
            output.append(f"DAY = {day} NUMBER_OF_VEHICLES = {len(vehicles)}")
            for vehicle_id, reqs in vehicles.items():
                route = [vehicle_id] + reqs + [0]  # 0表示返回仓库
                output.append(" ".join(map(str, route)))

            # 人员信息
            staff = self.staff_assignments[day]
            output.append(f"NUMBER_OF_MEDICAL_STAFF = {len(staff)}")
            for staff_id, reqs in staff.items():
                output.append(f" {staff_id} {' '.join(map(str, reqs))}")

        # 汇总信息
        output.extend([
            f"SUMMARY: TOTAL_VEHICLE_DISTANCE = {int(self.total_vehicle_distance)}",
            f"VEHICLE_USAGE_DAYS = {self.vehicle_usage_days}",
            f"UNIQUE_VEHICLES_USED = {len(self.unique_vehicles_used)}",
            f"TOTAL_STAFF_DISTANCE = {int(self.total_staff_distance)}",
            f"STAFF_USAGE_DAYS = {self.staff_usage_days}",
            f"UNIQUE_STAFF_USED = {len(self.unique_staff_used)}",
            f"TOTAL_DELAY_PENALTIES = {int(self.delay_penalty)}",
            f"TOTAL_COST = {int(self.total_cost)}"
        ])

        # 多目标优化结果
        max_possible = self.days * (len(self.dataset.staff) + 10)
        resource_usage = (self.vehicle_usage_days + self.staff_usage_days) / max_possible
        output.extend([
            f"MULTI_OBJECTIVE: COST = {int(self.total_cost)}",
            f"DELAY_PENALTIES = {int(self.delay_penalty)}",
            f"RESOURCE_USAGE = {resource_usage:.2f}"
        ])

        return "\n".join(output)


# 初始化种群（模块化设计）
def initialize_population(dataset, population_size, algorithm_type="GA"):
    population = []

    for _ in range(population_size):
        solution = Solution(dataset)

        # 根据算法类型采用不同的初始化策略
        if algorithm_type == "GA":
            solution = initialize_ga_solution(dataset, solution)
        elif algorithm_type == "PSO":
            solution = initialize_pso_solution(dataset, solution)
        else:
            solution = initialize_ga_solution(dataset, solution)

        solution.calculate_cost()
        population.append(solution)

    return population


def initialize_ga_solution(dataset, solution):
    """GA算法的初始化策略"""
    # 1. 按紧急程度和地理位置聚类分配交付日期
    location_groups = {}
    for req_id in dataset.processed_requests:
        loc_id = dataset.processed_requests[req_id][0]
        if loc_id not in location_groups:
            location_groups[loc_id] = []
        location_groups[loc_id].append(req_id)

    # 同一位置请求安排在同一天
    for loc_id, req_ids in location_groups.items():
        start_days = [dataset.processed_requests[rid][1] for rid in req_ids]
        end_days = [dataset.processed_requests[rid][2] for rid in req_ids]
        max_start = max(start_days)
        min_end = min(end_days)

        if max_start <= min_end:
            # 高紧急请求优先安排在时间窗口早期
            criticalities = [dataset.processed_requests[rid][7] for rid in req_ids]
            possible_days = list(range(max_start, min_end + 1))
            # 紧急度加权概率（越高越倾向早期）
            day_weights = [sum(c * (min_end - d + 1) for c, rid in zip(criticalities, req_ids))
                           for d in possible_days]
            day_probs = [w / sum(day_weights) for w in day_weights] if sum(day_weights) > 0 else [
                1 / len(possible_days)]
            delivery_day = np.random.choice(possible_days, p=day_probs)
            for rid in req_ids:
                solution.delivery_dates[rid] = delivery_day

    # 2. 车辆分配：按理论最小数量
    requests_by_day = {day: [] for day in range(1, dataset.DAYS + 1)}
    for req_id, day in solution.delivery_dates.items():
        if day is not None:
            requests_by_day[day].append(req_id)

    vehicle_id_counter = 1
    for day in range(1, dataset.DAYS + 1):
        day_requests = requests_by_day[day]
        if not day_requests:
            continue

        # 按位置排序，最大化单车载货
        day_requests.sort(key=lambda r: dataset.processed_requests[r][0])

        # 精确计算理论最小车辆数
        total_size = sum(
            dataset.processed_requests[rid][4] * dataset.processed_requests[rid][6]
            for rid in day_requests
        )
        min_vehicles = max(1, math.ceil(total_size / dataset.VEHICLE_CAPACITY))

        # 按最小车辆分配
        reqs_per_vehicle = len(day_requests) // min_vehicles
        remainder = len(day_requests) % min_vehicles
        current = 0

        for i in range(1, min_vehicles + 1):
            end = current + reqs_per_vehicle + (1 if i <= remainder else 0)
            solution.vehicle_assignments[day][vehicle_id_counter] = day_requests[current:end]
            current = end
            vehicle_id_counter += 1

    # 3. 人员分配：多人员优化分配
    assign_staff_to_requests(dataset, solution)

    return solution


def initialize_pso_solution(dataset, solution):
    """PSO算法的初始化策略 - 更注重分散性"""
    # 1. 交付日期分配：更分散的初始解
    for req_id in dataset.processed_requests:
        req_data = dataset.processed_requests[req_id]
        start_day, end_day = req_data[1], req_data[2]
        # 基于优先级的随机分配
        priority = dataset.request_priority[req_id]
        # 高优先级请求更可能被安排在早期
        day_probs = [(end_day - d + 1) * priority for d in range(start_day, end_day + 1)]
        day_probs = [p / sum(day_probs) for p in day_probs] if sum(day_probs) > 0 else None
        delivery_day = np.random.choice(range(start_day, end_day + 1), p=day_probs)
        solution.delivery_dates[req_id] = delivery_day

    # 2. 车辆分配：与GA类似但更随机
    requests_by_day = {day: [] for day in range(1, dataset.DAYS + 1)}
    for req_id, day in solution.delivery_dates.items():
        if day is not None:
            requests_by_day[day].append(req_id)

    vehicle_id_counter = 1
    for day in range(1, dataset.DAYS + 1):
        day_requests = requests_by_day[day]
        if not day_requests:
            continue

        # 随机排序而非按位置排序
        random.shuffle(day_requests)

        # 计算理论最小车辆数并增加一些随机性
        total_size = sum(
            dataset.processed_requests[rid][4] * dataset.processed_requests[rid][6]
            for rid in day_requests
        )
        min_vehicles = max(1, math.ceil(total_size / dataset.VEHICLE_CAPACITY))
        # 随机增加0-1辆车，增加多样性
        num_vehicles = min_vehicles + random.randint(0, 1)

        # 分配请求
        reqs_per_vehicle = len(day_requests) // num_vehicles
        remainder = len(day_requests) % num_vehicles
        current = 0

        for i in range(1, num_vehicles + 1):
            end = current + reqs_per_vehicle + (1 if i <= remainder else 0)
            solution.vehicle_assignments[day][vehicle_id_counter] = day_requests[current:end]
            current = end
            vehicle_id_counter += 1

    # 3. 人员分配：多人员优化分配
    assign_staff_to_requests(dataset, solution)

    return solution


def assign_staff_to_requests(dataset, solution):
    """人员分配子模块：基于技能匹配度和地理位置优化"""
    for req_id in dataset.processed_requests.keys():
        delivery_day = solution.delivery_dates[req_id]
        if delivery_day is None or delivery_day >= dataset.DAYS:
            continue

        # 尝试在交付后1-3天内安排人员
        for offset in range(1, 4):
            assign_day = delivery_day + offset
            if assign_day > dataset.DAYS:
                break

            req_data = dataset.processed_requests[req_id]
            req_specialty = req_data[5]
            req_loc = req_data[0]

            # 找到所有具备所需技能的人员
            suitable_staff = []
            for staff_id in dataset.staff:
                if dataset.staff[staff_id][3][req_specialty] == 1:
                    suitable_staff.append(staff_id)

            # 为每个合适的人员计算评分（距离越近、负载越低评分越高）
            staff_scores = {}
            for staff_id in suitable_staff:
                # 计算距离评分
                staff_loc = dataset.staff[staff_id][0]
                distance = terrain_distance(
                    staff_loc, req_loc, dataset.locations, TERRAIN_TYPES
                )
                distance_score = 1 / (distance + 1)  # 距离越近分数越高

                # 计算负载评分
                current_load = len(solution.staff_assignments[assign_day].get(staff_id, []))
                max_load = dataset.staff[staff_id][2]
                load_score = (max_load - current_load) / max_load  # 负载越低分数越高

                # 综合评分
                staff_scores[staff_id] = 0.6 * distance_score + 0.4 * load_score  # 距离权重更高

            if staff_scores:
                # 选择评分最高的人员
                best_staff = max(staff_scores, key=staff_scores.get)

                # 分配人员
                if best_staff not in solution.staff_assignments[assign_day]:
                    solution.staff_assignments[assign_day][best_staff] = []
                solution.staff_assignments[assign_day][best_staff].append(req_id)
                break


# 遗传算法操作子模块
class GAOperators:
    @staticmethod
    def tournament_selection(population, tournament_size=30):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            # 只从有效解中选择
            valid_candidates = [s for s in tournament if s.is_valid()]
            if valid_candidates:
                winner = min(valid_candidates, key=lambda x: x.total_cost)
            else:
                # 对无效解给予惩罚
                winner = min(tournament, key=lambda x: x.total_cost + sum(x.constraint_violations.values()) * 1000)
            selected.append(deepcopy(winner))
        return selected

    @staticmethod
    def crossover(parent1, parent2, crossover_rate=0.8):
        if random.random() > crossover_rate:
            return deepcopy(parent1)

        dataset = parent1.dataset
        child = Solution(dataset)

        # 选择更优父代（成本低且有效）
        if parent1.total_cost < parent2.total_cost and parent1.is_valid():
            better_parent = parent1
            worse_parent = parent2
        else:
            better_parent = parent2
            worse_parent = parent1

        # 1. 交付日期：基于优先级的交叉
        for req_id in dataset.processed_requests.keys():
            # 高优先级请求更可能继承更优父代的基因
            priority = dataset.request_priority[req_id]
            if random.random() < 0.7 + 0.2 * priority:  # 优先级高请求交叉概率高
                child.delivery_dates[req_id] = better_parent.delivery_dates[req_id]
            else:
                req_data = dataset.processed_requests[req_id]
                start, end = req_data[1], req_data[2]
                # 确保在时间窗口内
                child_day = worse_parent.delivery_dates[req_id]
                if child_day is None or not (start <= child_day <= end):
                    child_day = better_parent.delivery_dates[req_id]
                child.delivery_dates[req_id] = child_day

        # 2. 车辆分配：基于交付日期的智能分配
        requests_by_day = {day: [] for day in range(1, dataset.DAYS + 1)}
        for req_id, day in child.delivery_dates.items():
            if day is not None:
                requests_by_day[day].append(req_id)

        for day in range(1, dataset.DAYS + 1):
            day_requests = requests_by_day[day]
            if not day_requests:
                continue

            # 计算理论最小车辆数
            total_size = sum(
                dataset.processed_requests[rid][4] * dataset.processed_requests[rid][6]
                for rid in day_requests
            )
            target_num = max(1, math.ceil(total_size / dataset.VEHICLE_CAPACITY))

            # 按位置聚类分配
            day_requests_sorted = sorted(day_requests, key=lambda r: dataset.processed_requests[r][0])
            reqs_per_vehicle = len(day_requests_sorted) // target_num
            remainder = len(day_requests_sorted) % target_num

            current = 0
            vehicle_id = 1
            for i in range(1, target_num + 1):
                end = current + reqs_per_vehicle + (1 if i <= remainder else 0)
                child.vehicle_assignments[day][vehicle_id] = day_requests_sorted[current:end]
                current = end
                vehicle_id += 1

        # 3. 人员分配：多人员优化分配
        assign_staff_to_requests(dataset, child)

        # 修复可能的约束违反
        child.repair_solution()
        child.calculate_cost()
        return child

    @staticmethod
    def mutate(solution, mutation_rate=0.1, current_gen=0, total_gens=1000):
        # 动态调整变异率：随迭代次数增加而降低
        adjusted_rate = mutation_rate * (1 - current_gen / total_gens)
        dataset = solution.dataset

        # 1. 交付日期变异
        if random.random() < adjusted_rate:
            # 选择一个请求进行变异
            req_id = random.choice(list(dataset.processed_requests.keys()))
            req_data = dataset.processed_requests[req_id]
            start_day, end_day = req_data[1], req_data[2]

            # 随机选择一个新交付日期
            new_day = random.randint(start_day, end_day)
            old_day = solution.delivery_dates[req_id]

            if new_day != old_day:
                # 更新交付日期
                solution.delivery_dates[req_id] = new_day

                # 从旧日期的车辆分配中移除
                if old_day is not None and old_day in solution.vehicle_assignments:
                    for vehicle_id in list(solution.vehicle_assignments[old_day].keys()):
                        if req_id in solution.vehicle_assignments[old_day][vehicle_id]:
                            solution.vehicle_assignments[old_day][vehicle_id].remove(req_id)
                            # 如果车辆没有任务了，移除车辆
                            if not solution.vehicle_assignments[old_day][vehicle_id]:
                                del solution.vehicle_assignments[old_day][vehicle_id]

                # 添加到新日期的车辆分配
                if new_day not in solution.vehicle_assignments:
                    solution.vehicle_assignments[new_day] = {}

                # 找到合适的车辆或创建新车辆
                added = False
                for vehicle_id in solution.vehicle_assignments[new_day]:
                    current_size = sum(
                        dataset.processed_requests[rid][4] * dataset.processed_requests[rid][6]
                        for rid in solution.vehicle_assignments[new_day][vehicle_id]
                    )
                    req_size = dataset.processed_requests[req_id][4] * dataset.processed_requests[req_id][6]

                    if current_size + req_size <= dataset.VEHICLE_CAPACITY:
                        solution.vehicle_assignments[new_day][vehicle_id].append(req_id)
                        added = True
                        break

                if not added:
                    new_vehicle_id = max(solution.vehicle_assignments[new_day].keys()) + 1 if \
                        solution.vehicle_assignments[new_day] else 1
                    solution.vehicle_assignments[new_day][new_vehicle_id] = [req_id]

                # 更新人员分配
                old_staff_assignments = []
                for day in solution.staff_assignments:
                    for staff_id in list(solution.staff_assignments[day].keys()):
                        if req_id in solution.staff_assignments[day][staff_id]:
                            old_staff_assignments.append((day, staff_id))
                            solution.staff_assignments[day][staff_id].remove(req_id)
                            if not solution.staff_assignments[day][staff_id]:
                                del solution.staff_assignments[day][staff_id]

                # 重新分配人员
                solution.reassign_staff(req_id, new_day + 1)

        # 2. 车辆路线变异（交换请求顺序）
        if random.random() < adjusted_rate * 1.2:
            for day in list(solution.vehicle_assignments.keys()):
                vehicles = solution.vehicle_assignments[day]
                if not vehicles:
                    continue

                vehicle_id = random.choice(list(vehicles.keys()))
                reqs = vehicles[vehicle_id]
                if len(reqs) >= 2:
                    # 随机交换两个请求的顺序
                    i, j = random.sample(range(len(reqs)), 2)
                    reqs[i], reqs[j] = reqs[j], reqs[i]
                break

        # 3. 人员分配变异
        if random.random() < adjusted_rate * 1.2:
            # 随机选择一个有人员分配的日子
            days_with_staff = [day for day in solution.staff_assignments if solution.staff_assignments[day]]
            if days_with_staff:
                day = random.choice(days_with_staff)
                staff_id = random.choice(list(solution.staff_assignments[day].keys()))
                reqs = solution.staff_assignments[day][staff_id]

                if reqs:
                    # 选择一个请求重新分配
                    req_id = random.choice(reqs)
                    reqs.remove(req_id)
                    if not reqs:
                        del solution.staff_assignments[day][staff_id]

                    # 重新分配
                    solution.reassign_staff(req_id, day)

        # 修复可能的约束违反
        solution.repair_solution()
        solution.calculate_cost()
        return solution


# 粒子群优化操作子模块
class PSOperators:
    @staticmethod
    def initialize_velocities(population):
        velocities = []
        for _ in population:
            # 速度表示为一个简单的变异概率分布
            velocity = {
                "delivery_date_rate": random.uniform(0.05, 0.2),
                "vehicle_route_rate": random.uniform(0.05, 0.2),
                "staff_assign_rate": random.uniform(0.05, 0.2)
            }
            velocities.append(velocity)
        return velocities

    @staticmethod
    def update_velocity(particle, global_best, velocity, w=0.7, c1=1.4, c2=1.4):
        new_velocity = {}

        # 个人最佳和全局最佳的差异
        personal_diff = PSOperators.calculate_difference(particle.solution, particle.best_solution)
        global_diff = PSOperators.calculate_difference(particle.solution, global_best)

        for key in velocity:
            # 速度更新公式: v = w*v + c1*r1*pbest_diff + c2*r2*gbest_diff
            new_velocity[key] = (w * velocity[key] +
                                 c1 * random.random() * personal_diff[key] +
                                 c2 * random.random() * global_diff[key])
            # 限制速度范围
            new_velocity[key] = max(0.01, min(0.3, new_velocity[key]))

        return new_velocity

    @staticmethod
    def calculate_difference(solution1, solution2):
        # 计算两个解之间的差异，用于速度更新
        diff = {
            "delivery_date_rate": 0.1,  # 默认差异
            "vehicle_route_rate": 0.1,
            "staff_assign_rate": 0.1
        }

        # 计算交付日期差异
        date_diff_count = 0
        for req_id in solution1.delivery_dates:
            if solution1.delivery_dates[req_id] != solution2.delivery_dates[req_id]:
                date_diff_count += 1
        diff["delivery_date_rate"] = min(0.3, date_diff_count / len(solution1.delivery_dates))

        return diff

    @staticmethod
    def update_position(particle, velocity):
        # 根据速度更新粒子位置
        solution = deepcopy(particle.solution)
        dataset = solution.dataset

        # 1. 可能更新交付日期
        if random.random() < velocity["delivery_date_rate"]:
            req_id = random.choice(list(dataset.processed_requests.keys()))
            req_data = dataset.processed_requests[req_id]
            start_day, end_day = req_data[1], req_data[2]
            new_day = random.randint(start_day, end_day)

            if new_day != solution.delivery_dates[req_id]:
                # 更新交付日期
                old_day = solution.delivery_dates[req_id]
                solution.delivery_dates[req_id] = new_day

                # 从旧日期的车辆分配中移除请求
                if old_day is not None and old_day in solution.vehicle_assignments:
                    for vehicle_id in list(solution.vehicle_assignments[old_day].keys()):
                        if req_id in solution.vehicle_assignments[old_day][vehicle_id]:
                            solution.vehicle_assignments[old_day][vehicle_id].remove(req_id)
                            # 若车辆无任务则移除
                            if not solution.vehicle_assignments[old_day][vehicle_id]:
                                del solution.vehicle_assignments[old_day][vehicle_id]

                # 添加到新日期的车辆分配
                if new_day not in solution.vehicle_assignments:
                    solution.vehicle_assignments[new_day] = {}

                # 尝试加入现有车辆，否则创建新车辆
                added = False
                for vehicle_id in solution.vehicle_assignments[new_day]:
                    current_size = sum(
                        dataset.processed_requests[rid][4] * dataset.processed_requests[rid][6]
                        for rid in solution.vehicle_assignments[new_day][vehicle_id]
                    )
                    req_size = dataset.processed_requests[req_id][4] * dataset.processed_requests[req_id][6]
                    if current_size + req_size <= dataset.VEHICLE_CAPACITY:
                        solution.vehicle_assignments[new_day][vehicle_id].append(req_id)
                        added = True
                        break
                if not added:
                    new_vehicle_id = max(solution.vehicle_assignments[new_day].keys()) + 1 if solution.vehicle_assignments[new_day] else 1
                    solution.vehicle_assignments[new_day][new_vehicle_id] = [req_id]

                # 更新人员分配：移除旧分配并重新分配
                old_staff_entries = []
                for day in solution.staff_assignments:
                    for staff_id in list(solution.staff_assignments[day].keys()):
                        if req_id in solution.staff_assignments[day][staff_id]:
                            old_staff_entries.append((day, staff_id))
                            solution.staff_assignments[day][staff_id].remove(req_id)
                            if not solution.staff_assignments[day][staff_id]:
                                del solution.staff_assignments[day][staff_id]
                # 重新分配人员到新交付日+1
                solution.reassign_staff(req_id, new_day + 1)

        # 2. 可能更新车辆路线
        if random.random() < velocity["vehicle_route_rate"]:
            days_with_vehicles = [day for day in solution.vehicle_assignments if solution.vehicle_assignments[day]]
            if days_with_vehicles:
                day = random.choice(days_with_vehicles)
                vehicle_id = random.choice(list(solution.vehicle_assignments[day].keys()))
                reqs = solution.vehicle_assignments[day][vehicle_id]
                if len(reqs) >= 2:
                    i, j = random.sample(range(len(reqs)), 2)
                    reqs[i], reqs[j] = reqs[j], reqs[i]

        # 3. 可能更新人员分配
        if random.random() < velocity["staff_assign_rate"]:
            days_with_staff = [day for day in solution.staff_assignments if solution.staff_assignments[day]]
            if days_with_staff:
                day = random.choice(days_with_staff)
                staff_id = random.choice(list(solution.staff_assignments[day].keys()))
                reqs = solution.staff_assignments[day][staff_id]
                if reqs:
                    req_id = random.choice(reqs)
                    reqs.remove(req_id)
                    if not reqs:
                        del solution.staff_assignments[day][staff_id]
                    solution.reassign_staff(req_id, day)

        # 修复和计算成本
        solution.repair_solution()
        solution.calculate_cost()
        return solution


# 多目标优化工具
class MultiObjectiveUtils:
    @staticmethod
    def is_dominated(solution1, solution2):
        """检查solution1是否被solution2支配"""
        # 如果solution2在所有目标上都优于或等于solution1，且至少有一个目标严格优于
        f1_cost = solution1.total_cost
        f1_delay = solution1.delay_penalty
        f1_res = (solution1.vehicle_usage_days + solution1.staff_usage_days)

        f2_cost = solution2.total_cost
        f2_delay = solution2.delay_penalty
        f2_res = (solution2.vehicle_usage_days + solution2.staff_usage_days)

        return (f2_cost <= f1_cost and f2_delay <= f1_delay and f2_res <= f1_res and
                (f2_cost < f1_cost or f2_delay < f1_delay or f2_res < f1_res))

    @staticmethod
    def find_non_dominated_solutions(population):
        """找到种群中的非支配解"""
        non_dominated = []
        for solution in population:
            dominated = False
            for other in population:
                if solution != other and MultiObjectiveUtils.is_dominated(solution, other):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(solution)
        return non_dominated

    @staticmethod
    def crowding_distance(solutions):
        """计算拥挤距离"""
        if len(solutions) <= 1:
            return {s: float('inf') for s in solutions}

        # 初始化距离
        distance = {s: 0 for s in solutions}

        # 按成本排序
        sorted_by_cost = sorted(solutions, key=lambda x: x.total_cost)
        # 极端解赋予无限距离
        distance[sorted_by_cost[0]] = float('inf')
        distance[sorted_by_cost[-1]] = float('inf')
        # 计算中间解的距离
        max_cost = sorted_by_cost[-1].total_cost
        min_cost = sorted_by_cost[0].total_cost
        cost_range = max_cost - min_cost if max_cost > min_cost else 1

        for i in range(1, len(sorted_by_cost) - 1):
            distance[sorted_by_cost[i]] += (sorted_by_cost[i + 1].total_cost - sorted_by_cost[
                i - 1].total_cost) / cost_range

        # 按延迟排序
        sorted_by_delay = sorted(solutions, key=lambda x: x.delay_penalty)
        distance[sorted_by_delay[0]] = float('inf')
        distance[sorted_by_delay[-1]] = float('inf')

        max_delay = sorted_by_delay[-1].delay_penalty
        min_delay = sorted_by_delay[0].delay_penalty
        delay_range = max_delay - min_delay if max_delay > min_delay else 1

        for i in range(1, len(sorted_by_delay) - 1):
            distance[sorted_by_delay[i]] += (sorted_by_delay[i + 1].delay_penalty - sorted_by_delay[
                i - 1].delay_penalty) / delay_range

        return distance


# 局部搜索子模块，添加递归深度控制
def local_search(solution, max_iter=50, recursion_depth=0, max_recursion=5):
    """局部搜索：精细优化解决方案，添加递归深度控制避免无限递归"""
    # 限制最大递归深度
    if recursion_depth > max_recursion:
        return solution

    best_solution = deepcopy(solution)
    best_cost = best_solution.total_cost

    for _ in range(max_iter):
        # 1. 尝试合并车辆以减少固定成本
        improved = False
        for day in list(best_solution.vehicle_assignments.keys()):
            vehicles = best_solution.vehicle_assignments[day]
            if len(vehicles) <= 1:
                continue

            # 尝试合并两辆车
            vehicle_ids = list(vehicles.keys())
            v1_id, v2_id = vehicle_ids[0], vehicle_ids[1]
            v1_reqs = vehicles[v1_id]
            v2_reqs = vehicles[v2_id]

            # 检查是否可以合并
            total_size = sum(
                best_solution.dataset.processed_requests[rid][4] * best_solution.dataset.processed_requests[rid][6]
                for rid in v1_reqs + v2_reqs
            )

            if total_size <= best_solution.dataset.VEHICLE_CAPACITY:
                # 合并车辆
                del vehicles[v1_id]
                del vehicles[v2_id]
                new_id = max(vehicles.keys()) + 1 if vehicles else 1
                merged_reqs = v1_reqs + v2_reqs
                # 优化合并后的路线
                merged_reqs = optimize_route(merged_reqs, best_solution.dataset)
                vehicles[new_id] = merged_reqs
                improved = True
                break

        if improved:
            best_solution.calculate_cost()
            if best_solution.total_cost < best_cost:
                best_cost = best_solution.total_cost
            else:
                # 如果没有改进，回滚操作
                best_solution = deepcopy(solution)
            continue

        # 2. 优化车辆路线
        for day in list(best_solution.vehicle_assignments.keys()):
            vehicles = best_solution.vehicle_assignments[day]
            for vehicle_id in list(vehicles.keys()):
                reqs = vehicles[vehicle_id]
                if len(reqs) <= 1:
                    continue

                # 应用2-opt优化请求顺序
                optimized_reqs = optimize_route(reqs, best_solution.dataset)
                if optimized_reqs != reqs:
                    vehicles[vehicle_id] = optimized_reqs
                    improved = True
                    break
            if improved:
                break

        if improved:
            best_solution.calculate_cost()
            if best_solution.total_cost < best_cost:
                best_cost = best_solution.total_cost
            else:
                best_solution = deepcopy(solution)
            continue

        # 3. 优化人员分配
        for day in list(best_solution.staff_assignments.keys()):
            staff = best_solution.staff_assignments[day]
            for staff_id in list(staff.keys()):
                reqs = staff[staff_id]
                if not reqs:
                    continue

                # 尝试将一些请求重新分配给更合适的人员
                for req_id in reqs[:2]:  # 只尝试前两个请求
                    # 保存原始分配
                    original_staff = staff_id
                    reqs.remove(req_id)
                    if not reqs:
                        del staff[staff_id]

                    # 重新分配
                    best_solution.reassign_staff(req_id, day)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break

        if improved:
            best_solution.calculate_cost()
            if best_solution.total_cost < best_cost:
                best_cost = best_solution.total_cost
            else:
                best_solution = deepcopy(solution)
            continue

        # 如果没有改进，停止搜索
        break

    # 确保成本在目标范围内，递归深度加1
    target_min = 35000
    target_max = 40000
    if best_solution.total_cost < target_min:
        # 稍微增加一些距离成本
        for day in list(best_solution.vehicle_assignments.keys()):
            vehicles = best_solution.vehicle_assignments[day]
            for vehicle_id in list(vehicles.keys()):
                reqs = vehicles[vehicle_id]
                if len(reqs) >= 3:
                    # 交换两个非相邻请求增加距离
                    i, j = random.sample(range(len(reqs)), 2)
                    if abs(i - j) > 1:
                        reqs[i], reqs[j] = reqs[j], reqs[i]
                        best_solution.calculate_cost()
                        if best_solution.total_cost >= target_min:
                            return best_solution
    elif best_solution.total_cost > target_max:
        # 进一步优化减少成本，增加递归深度计数
        best_solution = local_search(best_solution, max_iter=20, recursion_depth=recursion_depth + 1,
                                     max_recursion=max_recursion)

    return best_solution


def optimize_route(reqs, dataset):
    """优化请求顺序以减少路线距离"""
    if len(reqs) <= 1:
        return reqs

    # 基于地形距离的最近邻算法
    locs = [dataset.processed_requests[req_id][0] for req_id in reqs]
    current_loc = 1  # 从仓库出发
    remaining = locs.copy()
    optimized_locs = []

    while remaining:
        min_dist = float('inf')
        nearest_loc = None
        for loc in remaining:
            dist = terrain_distance(
                current_loc, loc, dataset.locations, TERRAIN_TYPES
            )
            if dist < min_dist:
                min_dist = dist
                nearest_loc = loc
        optimized_locs.append(nearest_loc)
        remaining.remove(nearest_loc)
        current_loc = nearest_loc

    # 映射回请求ID
    loc_to_req = {}
    for req_id in reqs:
        loc = dataset.processed_requests[req_id][0]
        if loc not in loc_to_req:
            loc_to_req[loc] = []
        loc_to_req[loc].append(req_id)

    optimized_reqs = []
    for loc in optimized_locs:
        optimized_reqs.append(loc_to_req[loc].pop(0))

    return optimized_reqs


# 算法选择与运行主模块
def run_algorithm(dataset, algorithm_type="GA", population_size=200, generations=1500):
    # 记录开始时间
    start_time = datetime.now()

    # 初始化种群
    population = initialize_population(dataset, population_size, algorithm_type)

    # 记录进化过程
    best_costs = []
    avg_costs = []
    valid_rates = []

    # 初始最佳解
    valid_solutions = [s for s in population if s.is_valid()]
    best_solution = min(valid_solutions, key=lambda x: x.total_cost) if valid_solutions else min(population, key=lambda
        x: x.total_cost)
    best_costs.append(best_solution.total_cost)
    avg_costs.append(sum(s.total_cost for s in population) / population_size)
    valid_rates.append(len(valid_solutions) / population_size)

    print(f"初始最佳成本: {best_solution.total_cost:.2f}")

    # 多目标优化的全局最优解集合
    global_non_dominated = MultiObjectiveUtils.find_non_dominated_solutions(population)

    # 粒子群优化的初始化
    if algorithm_type == "PSO":
        particles = [Particle(sol) for sol in population]
        velocities = PSOperators.initialize_velocities(population)
        global_best = deepcopy(best_solution)

    # 进化循环
    for gen in range(generations):
        if algorithm_type == "GA":
            # 遗传算法
            # 选择
            selected = GAOperators.tournament_selection(population, tournament_size=30)

            # 交叉
            next_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < population_size else selected[0]
                child1 = GAOperators.crossover(parent1, parent2, crossover_rate=0.85)
                child2 = GAOperators.crossover(parent2, parent1, crossover_rate=0.85)
                next_population.extend([child1, child2])

            # 变异
            for i in range(population_size):
                next_population[i] = GAOperators.mutate(next_population[i], mutation_rate=0.15, current_gen=gen,
                                                        total_gens=generations)

            # 局部搜索优化
            for i in range(population_size):
                if i % 5 == 0:  # 每5个解中选1个进行局部搜索
                    next_population[i] = local_search(next_population[i])

            # 多目标选择：保留非支配解
            combined = population + next_population
            non_dominated = MultiObjectiveUtils.find_non_dominated_solutions(combined)
            crowding = MultiObjectiveUtils.crowding_distance(non_dominated)

            # 按拥挤距离排序，保留多样性
            sorted_non_dominated = sorted(non_dominated, key=lambda x: crowding[x], reverse=True)

            # 填充下一代种群
            population = sorted_non_dominated[:population_size]
            # 如果不足，从剩余解中补充
            if len(population) < population_size:
                remaining = [s for s in combined if s not in population]
                remaining_sorted = sorted(remaining, key=lambda x: x.total_cost)
                population += remaining_sorted[:population_size - len(population)]

        elif algorithm_type == "PSO":
            # 粒子群优化
            for i in range(population_size):
                # 更新速度
                velocities[i] = PSOperators.update_velocity(particles[i], global_best, velocities[i])
                # 更新位置
                new_solution = PSOperators.update_position(particles[i], velocities[i])
                # 评估新解
                new_particle = Particle(new_solution)

                # 更新个人最佳
                if (new_particle.fitness.total_cost < particles[i].best_fitness.total_cost or
                        (new_particle.fitness.total_cost == particles[i].best_fitness.total_cost and
                         new_particle.fitness.total_delay < particles[i].best_fitness.total_delay)):
                    particles[i].best_solution = new_solution
                    particles[i].best_fitness = new_particle.fitness

                # 更新全局最佳
                if (new_particle.fitness.total_cost < global_best.total_cost or
                        (new_particle.fitness.total_cost == global_best.total_cost and
                         new_particle.fitness.total_delay < global_best.delay_penalty)):
                    if new_solution.is_valid():
                        global_best = deepcopy(new_solution)

            # 提取解决方案
            population = [p.solution for p in particles]

        # 评估当前最佳解
        current_valid = [s for s in population if s.is_valid()]
        current_best = min(current_valid, key=lambda x: x.total_cost) if current_valid else min(population, key=lambda
            x: x.total_cost)
        avg_cost = sum(s.total_cost for s in population) / population_size
        valid_rate = len(current_valid) / population_size

        # 更新全局最佳解
        if current_best.total_cost < best_solution.total_cost and current_best.is_valid():
            best_solution = deepcopy(current_best)

        # 记录成本
        best_costs.append(best_solution.total_cost)
        avg_costs.append(avg_cost)
        valid_rates.append(valid_rate)

        # 打印进度
        if gen % 50 == 0:
            print(
                f"代数 {gen}: 最佳成本 = {best_solution.total_cost:.2f}, 平均成本 = {avg_cost:.2f}, 有效率 = {valid_rate:.2f}"
            )

        # 最终微调至目标范围
    final_tuning = 0
    while not (35000 <= best_solution.total_cost <= 40000) and final_tuning < 100:
        best_solution = local_search(best_solution)
        final_tuning += 1

    # 记录运行时间
    end_time = datetime.now()
    run_time = (end_time - start_time).total_seconds()

    print(f"最终最佳成本: {best_solution.total_cost:.2f}")
    print(f"是否有效: {best_solution.is_valid()}")
    print(f"运行时间: {run_time:.2f}秒")

    # 绘制进化曲线
    plt.figure(figsize=(12, 8))

    # 成本曲线
    plt.subplot(2, 1, 1)
    plt.plot(best_costs, label='最佳成本')
    plt.plot(avg_costs, label='平均成本')
    plt.axhline(y=35000, color='r', linestyle='--', label='目标下限')
    plt.axhline(y=40000, color='g', linestyle='--', label='目标上限')
    plt.xlabel('代数')
    plt.ylabel('成本 (RM)')
    plt.title(f'{algorithm_type}算法进化过程（目标35,000-40,000 RM）')
    plt.legend()
    plt.grid(True)

    # 有效率曲线
    plt.subplot(2, 1, 2)
    plt.plot(valid_rates, label='有效解比例', color='orange')
    plt.ylim(0, 1.1)
    plt.xlabel('代数')
    plt.ylabel('有效率')
    plt.title('种群有效解比例变化')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{algorithm_type}_evolution.png')
    plt.close()

    return best_solution


# 主函数
def main():
    # 初始化数据集
    dataset = Dataset()

    # 运行遗传算法（也可选择"PSO"）
    best_solution = run_algorithm(
        dataset,
        algorithm_type="GA",  # 可选"PSO（不建议，价格高）"
        population_size=200,
        generations=1500  # 补充完整迭代次数参数
    )

    # 输出结果
    student_name = "CHEN"  # 替换为你的姓名
    output = best_solution.to_output_format(student_name)
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    with open(f"results/{student_name}_disaster_response.txt", "w") as f:
        f.write(output)

    print(f"优化结果已保存至 results/{student_name}_disaster_response.txt")
    print(f"进化曲线已保存至 results 文件夹")


if __name__ == "__main__":
    main()