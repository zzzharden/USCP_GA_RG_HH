import copy
import random
import time
from collections import defaultdict

import numpy as np
from plot import plot_schedule, plot_teacher_schedule, plot_class_schedule, plot_class_heatmap, plot_schedule1
import random
from data_loader import data,data1,data2
from obj_function import obj_func_all, obj_func_all_print, obj_func_detail
import matplotlib.pyplot as plt

total_weeks = 16
days_per_week = 5
timeslots_per_day = 5
total_timeslots = days_per_week * timeslots_per_day
classrooms = data2(r'data\classrooms.xlsx')

def schedule_courses(courses_task, classes, teachers, schedule_all,col):
    # 初始化排课表
    schedule = np.full((len(classrooms), total_timeslots), None)
    # 按班级数量降序排序课程
    courses_task = sorted(courses_task, key=lambda x: -len(x['class']))

    while True:
        # 初始化班级和教师的排课时间表
        class_schedule = {i: set() for i in classes}
        teacher_schedule = {i['name']: set() for i in teachers}

        # 初始化可用时间段的字典
        available_timeslots_dict = {}
        for classroom in classrooms:
            classroom_type = classroom['type']
            if classroom_type not in available_timeslots_dict:
                available_timeslots_dict[classroom_type] = []
            for i in range(total_timeslots):
                if schedule_all[classrooms.index(classroom)][i] is None:
                    available_timeslots_dict[classroom_type].append((i, classrooms.index(classroom)))

        success = True
        for course in courses_task:
            remaining_duration = course['duration']
            while remaining_duration > 0:
                dtime = min(remaining_duration, total_weeks)
                allowed_types = course['allowed_classrooms']
                available_timeslots = []
                for classroom_type in allowed_types:
                    if classroom_type in available_timeslots_dict:
                        for timeslot, classroom_index in available_timeslots_dict[classroom_type]:
                            all_classes_available = all(
                                timeslot not in class_schedule[class_id] for class_id in course['class'])
                            if all_classes_available and timeslot not in teacher_schedule[course['teacher']]:
                                available_timeslots.append((timeslot, classroom_index))

                if not available_timeslots:
                    print(f"无法安排课程: {course}")
                    success = False
                    break

                probability = random.random()
                used_classroom_timeslots = [(ts, ci) for ts, ci in available_timeslots if
                                            any(schedule_all[ci])]
                if probability < 0.95 and used_classroom_timeslots:
                    start_timeslot, classroom_index = random.choice(used_classroom_timeslots)
                else:
                    start_timeslot, classroom_index = random.choice(available_timeslots)

                # 更新排课表
                schedule[classroom_index][start_timeslot] = {
                    'id': course['id'], 'course': course['course'], 'class': course['class'],
                    'teacher': course['teacher'], 'duration': course['duration'],
                    'classroom': classrooms[classroom_index]['id'],'allowed_classrooms':course['allowed_classrooms'],
                    'time': dtime
                }
                schedule_all[classroom_index][start_timeslot] = {
                    'id': col+'-'+str(course['id']), 'course': course['course'], 'class': course['class'],
                    'teacher': course['teacher'], 'duration': course['duration'],
                    'classroom': classrooms[classroom_index]['id'],'allowed_classrooms':course['allowed_classrooms'],
                    'time': dtime
                }

                # 更新班级和教师的排课时间表
                for class_id in course['class']:
                    class_schedule[class_id].add(start_timeslot)
                teacher_schedule[course['teacher']].add(start_timeslot)

                # 从可用时间段字典中移除已使用的时间段
                classroom_type = classrooms[classroom_index]['type']
                available_timeslots_dict[classroom_type].remove((start_timeslot, classroom_index))

                remaining_duration -= dtime
            if not success:
                break

        if not success:
            print("排课失败，正在重新排课...")
            # 清空排课表
            for i in range(len(schedule)):
                for j in range(len(schedule[i])):
                    if schedule[i][j]:
                        schedule[i][j] = None
                        schedule_all[i][j] = None
        else:
            return schedule, class_schedule, teacher_schedule


# 生成初始解的函数，具体实现需根据问题定义
def generate_initial_solution(paths):
    # 这里只是示例，需要根据实际问题调整
    pop = []
    pop.append([random.randint(0, 8) for _ in range(5)])
    schedule_all = np.full((len(classrooms), total_timeslots), None)
    for i, path in enumerate(paths):
        college_name = f'学院{i + 1}'
        courses_task, teachers, classes, courses = data1(path)
        res, cr, tr = schedule_courses(courses_task, classes, teachers, schedule_all,str(i+1))
        pop.append([res, cr, tr, teachers, courses, schedule_all])
    return pop

def selection(population, fitness, tournament_size=5):
    """
    锦标赛选择函数，适应度值越小被选中的概率越高
    :param population: 种群，一个包含多个个体的列表
    :param fitness: 每个个体对应的适应度值列表
    :param tournament_size: 锦标赛规模，默认为5
    :return: 选中的两个个体
    """
    selected_individuals = []
    for _ in range(2):  # 选择两个个体
        # 随机选择 tournament_size 个个体的索引
        tournament_indices = random.sample(range(len(population)), tournament_size)
        # 获取这些个体的适应度值
        tournament_fitness = [fitness[i] for i in tournament_indices]
        # 找到适应度值最小的个体的索引（因为适应度越小越好）
        min_fitness_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        # 添加到选中列表
        selected_individuals.append(copy.deepcopy(population[min_fitness_index]))
    return selected_individuals[0], selected_individuals[1]

# 单点交叉策略函数，具体实现需根据问题定义
def one_point_crossover(parent1, parent2):
    # 随机选择一个交叉点
    crossover_point = random.randint(1, len(parent1) - 1)
    # 生成子代个体，交叉点之前的部分来自 parent1，之后的部分来自 parent2
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


# 单点变异策略函数，具体实现需根据问题定义
def one_point_mutation(solution):
    # 这里只是示例，需要根据实际问题调整
    mutation_point = random.randint(0, len(solution) - 1)
    new_solution = solution.copy()
    new_solution[mutation_point] = random.randint(0, 8)
    return new_solution


# 根据低层次启发式向量LVa更新服务组合向量SVa，具体实现需根据问题定义
def update_sva(lva, sol):
    # 这里只是示例，需要根据实际问题调整
    for i in lva:
        for v in sol:
            s = [v[0], v[1], v[2]]
            if i == 0:
                s = A_swap_random_timeslots(v[0], v[1], v[2],v[5])
            elif i == 1:
                s = A_reverse_consecutive_timeslots(v[0], v[1], v[2],v[5])
            elif i == 2:
                s = A_swap_classroom_type_timeslots(v[0], v[1], v[2],v[5])
            elif i == 3:
                s = A_random(v[0], v[1], v[2],v[5])
            elif i == 4:
                s = B_night(v[0], v[1], v[2],v[5])
            elif i == 5:
                s = B_course(v[0], v[1], v[2],v[5])
            elif i == 6:
                s = B_c_course(v[0], v[1], v[2],v[4],v[5])
            elif i==7:
                s = B_c_teacher(v[0], v[1], v[2],v[3],v[5])
            elif i==8:
                s = B_method(v[0], v[1], v[2],v[3],v[5])

            s.append(v[3])
            s.append(v[4])
            s.append(v[5])
            v = s

    return sol


def A_swap_random_timeslots(schedule, class_schedule, teacher_schedule,schedule_all):

    # 找出使用过的教室的索引
    used_classroom_indices = []
    for classroom_index in range(len(schedule_all)):
        if any(schedule_all[classroom_index]):
            used_classroom_indices.append(classroom_index)

    """
    随机交换排课表中的两列元素（两个时间段），并同步更新班级和教师的时间表

    参数:
    schedule (list): 排课表，二维数组，格式为 [教室][时间段]
    class_schedule (dict): 班级时间表，{班级ID: 已排课的时间段集合}
    teacher_schedule (dict): 教师时间表，{教师姓名: 已排课的时间段集合}

    返回:
    tuple: (交换后的排课表, 班级时间表, 教师时间表, 被交换的两个时间段索引)
    """


    # 随机选择两个不同的时间段
    timeslot1, timeslot2 = random.sample(range(total_timeslots), 2)

    # 1. 更新班级和教师时间表
    # 收集所有课程涉及的班级和教师
    classes_involved = set()
    teachers_involved = set()

    # 遍历所有教室的这两个时间段
    for classroom_idx in range(len(schedule)):
        # 处理时间段1的课程
        course1 = schedule[classroom_idx][timeslot1]
        if course1:
            classes_involved.update(course1['class'])
            teachers_involved.add(course1['teacher'])

        # 处理时间段2的课程
        course2 = schedule[classroom_idx][timeslot2]
        if course2:
            classes_involved.update(course2['class'])
            teachers_involved.add(course2['teacher'])

    # 更新班级时间表：交换两个时间段的记录
    for class_id in classes_involved:
        slots = class_schedule[class_id]
        has_slot1 = timeslot1 in slots
        has_slot2 = timeslot2 in slots

        if has_slot1 and not has_slot2:
            slots.remove(timeslot1)
            slots.add(timeslot2)
        elif has_slot2 and not has_slot1:
            slots.remove(timeslot2)
            slots.add(timeslot1)
        elif has_slot1 and has_slot2:
            pass  # 两个时间段都有课，无需修改
        else:
            pass  # 两个时间段都没课，无需修改

    # 更新教师时间表：交换两个时间段的记录
    for teacher_id in teachers_involved:
        slots = teacher_schedule[teacher_id]
        has_slot1 = timeslot1 in slots
        has_slot2 = timeslot2 in slots

        if has_slot1 and not has_slot2:
            slots.remove(timeslot1)
            slots.add(timeslot2)
        elif has_slot2 and not has_slot1:
            slots.remove(timeslot2)
            slots.add(timeslot1)
        elif has_slot1 and has_slot2:
            pass  # 两个时间段都有课，无需修改
        else:
            pass  # 两个时间段都没课，无需修改

    # 2. 交换排课表中的两个时间段
    for classroom in schedule:
        classroom[timeslot1], classroom[timeslot2] = classroom[timeslot2], classroom[timeslot1]
        for i in used_classroom_indices:
            if classroom[timeslot1] and classrooms[i]['id']==classroom[timeslot1]['classroom']:
                schedule_all[i][timeslot1],schedule_all[i][timeslot2]=schedule_all[i][timeslot2],schedule_all[i][timeslot1]
            elif classroom[timeslot2] and classrooms[i]['id']==classroom[timeslot2]['classroom']:
                schedule_all[i][timeslot1],schedule_all[i][timeslot2]=schedule_all[i][timeslot2],schedule_all[i][timeslot1]

    return [schedule, class_schedule, teacher_schedule]

def A_reverse_consecutive_timeslots(schedule, class_schedule, teacher_schedule, schedule_all):
    """
    随机选择2到4个连续的时间段并将它们倒序排列，同时更新班级、教师和全局的时间表

    参数:
    schedule (list): 排课表，二维数组，格式为 [教室][时间段]
    class_schedule (dict): 班级时间表，{班级ID: 已排课的时间段集合}
    teacher_schedule (dict): 教师时间表，{教师姓名: 已排课的时间段集合}
    schedule_all (list): 全局排课表，二维数组，格式为 [教室索引][时间段]
    classrooms (list): 教室信息列表，包含教室ID等信息

    返回:
    tuple: (交换后的排课表, 班级时间表, 教师时间表, 全局排课表, 被倒序的时间段范围)
    """

    # 随机确定连续时间段的长度（2-4）
    block_size = min(random.randint(2, 4), total_timeslots)

    # 随机选择起始位置
    start_idx = random.randint(0, total_timeslots - block_size)
    end_idx = start_idx + block_size - 1

    # 创建时间段的倒序列表
    reversed_indices = list(range(start_idx, end_idx + 1))[::-1]

    # 1. 更新班级和教师时间表
    # 收集所有课程涉及的班级和教师
    classes_involved = set()
    teachers_involved = set()

    # 遍历所有教室的这个时间段块
    for classroom_idx in range(len(schedule)):
        for original_idx in range(start_idx, end_idx + 1):
            course = schedule[classroom_idx][original_idx]
            if course:
                classes_involved.update(course['class'])
                teachers_involved.add(course['teacher'])

    # 创建新旧时间段的映射
    time_mapping = {original: new_idx for original, new_idx in
                    zip(range(start_idx, end_idx + 1), reversed_indices)}

    # 更新班级时间表
    for class_id in classes_involved:
        slots = class_schedule[class_id]
        # 找出在倒序块中的时间段
        slots_in_block = [slot for slot in slots if start_idx <= slot <= end_idx]

        # 移除这些时间段
        for slot in slots_in_block:
            slots.remove(slot)

        # 添加倒序后的时间段
        for slot in slots_in_block:
            slots.add(time_mapping[slot])

    # 更新教师时间表
    for teacher_id in teachers_involved:
        slots = teacher_schedule[teacher_id]
        # 找出在倒序块中的时间段
        slots_in_block = [slot for slot in slots if start_idx <= slot <= end_idx]

        # 移除这些时间段
        for slot in slots_in_block:
            slots.remove(slot)

        # 添加倒序后的时间段
        for slot in slots_in_block:
            slots.add(time_mapping[slot])

    # 2. 倒序排课表中的时间段块
    for classroom_idx in range(len(schedule)):
        # 提取需要倒序的时间段
        block = [schedule[classroom_idx][i] for i in range(start_idx, end_idx + 1)]
        # 倒序后放回原位置
        for i, course in zip(range(start_idx, end_idx + 1), reversed(block)):
            schedule[classroom_idx][i] = course

    # 3. 更新schedule_all中的对应记录
    # 建立教室ID到schedule_all索引的映射
    classroom_id_to_index = {classroom['id']: idx for idx, classroom in enumerate(classrooms)}

    # 对每个教室处理倒序块
    for classroom in schedule:
        # 遍历倒序块中的每个时间段
        for original_idx, new_idx in zip(range(start_idx, end_idx + 1), reversed_indices):
            course = classroom[original_idx]
            if course:
                # 找到该课程对应的教室在schedule_all中的索引
                classroom_id = course.get('classroom')
                if classroom_id and classroom_id in classroom_id_to_index:
                    schedule_all_idx = classroom_id_to_index[classroom_id]
                    # 更新schedule_all中对应位置
                    schedule_all[schedule_all_idx][original_idx] = course.copy()
                    # 更新课程中的时间信息
                    schedule_all[schedule_all_idx][original_idx]['time'] = new_idx - start_idx + 1

    return [schedule, class_schedule, teacher_schedule]

def A_swap_classroom_type_timeslots(schedule, class_schedule, teacher_schedule, schedule_all):
    """
    随机交换两个时间段中某一类教室的所有元素，同时更新班级、教师和全局的时间表

    参数:
    schedule (list): 排课表，二维数组，格式为 [教室][时间段]
    class_schedule (dict): 班级时间表，{班级ID: 已排课的时间段集合}
    teacher_schedule (dict): 教师时间表，{教师姓名: 已排课的时间段集合}
    schedule_all (list): 全局排课表，二维数组，格式为 [教室索引][时间段]
    classrooms (list): 教室信息列表，包含教室ID和类型等信息

    返回:
    tuple: (交换后的排课表, 班级时间表, 教师时间表, 全局排课表, 被交换的时间段索引, 教室类型)
    """
    # 获取所有教室类型
    classroom_types = list({classroom['type'] for classroom in classrooms})

    # 确保有可用的教室类型
    if not classroom_types:
        print("警告: 没有找到可用的教室类型")
        return schedule, class_schedule, teacher_schedule, schedule_all, (None, None), None

    # 随机选择一种教室类型
    selected_type = random.choice(classroom_types)

    # 获取该类型教室在schedule中的索引
    type_indices = [i for i, classroom in enumerate(classrooms) if classroom['type'] == selected_type]

    # 确保该类型有教室
    if not type_indices:
        print(f"警告: 教室类型 {selected_type} 没有对应的教室")
        return schedule, class_schedule, teacher_schedule, schedule_all, (None, None), None

    # 随机选择两个不同的时间段
    timeslot1, timeslot2 = random.sample(range(total_timeslots), 2)

    # 1. 更新班级和教师时间表
    # 收集所有课程涉及的班级和教师
    classes_involved = set()
    teachers_involved = set()

    # 遍历该类型的所有教室
    for classroom_idx in type_indices:
        # 处理时间段1的课程
        course1 = schedule[classroom_idx][timeslot1]
        if course1:
            classes_involved.update(course1['class'])
            teachers_involved.add(course1['teacher'])

        # 处理时间段2的课程
        course2 = schedule[classroom_idx][timeslot2]
        if course2:
            classes_involved.update(course2['class'])
            teachers_involved.add(course2['teacher'])

    # 更新班级时间表：交换两个时间段的记录
    for class_id in classes_involved:
        slots = class_schedule[class_id]
        has_slot1 = timeslot1 in slots
        has_slot2 = timeslot2 in slots

        if has_slot1 and not has_slot2:
            slots.remove(timeslot1)
            slots.add(timeslot2)
        elif has_slot2 and not has_slot1:
            slots.remove(timeslot2)
            slots.add(timeslot1)
        elif has_slot1 and has_slot2:
            pass  # 两个时间段都有课，无需修改
        else:
            pass  # 两个时间段都没课，无需修改

    # 更新教师时间表：交换两个时间段的记录
    for teacher_id in teachers_involved:
        slots = teacher_schedule[teacher_id]
        has_slot1 = timeslot1 in slots
        has_slot2 = timeslot2 in slots

        if has_slot1 and not has_slot2:
            slots.remove(timeslot1)
            slots.add(timeslot2)
        elif has_slot2 and not has_slot1:
            slots.remove(timeslot2)
            slots.add(timeslot1)
        elif has_slot1 and has_slot2:
            pass  # 两个时间段都有课，无需修改
        else:
            pass  # 两个时间段都没课，无需修改

    # 2. 交换排课表中该类型教室的两个时间段
    for classroom_idx in type_indices:
        schedule[classroom_idx][timeslot1], schedule[classroom_idx][timeslot2] = \
            schedule[classroom_idx][timeslot2], schedule[classroom_idx][timeslot1]

    # 3. 更新schedule_all中的对应记录
    # 建立教室ID到schedule_all索引的映射
    classroom_id_to_index = {classroom['id']: idx for idx, classroom in enumerate(classrooms)}

    # 对该类型的每个教室处理交换
    for classroom_idx in type_indices:
        classroom_id = classrooms[classroom_idx]['id']
        if classroom_id in classroom_id_to_index:
            schedule_all_idx = classroom_id_to_index[classroom_id]
            # 交换schedule_all中的对应时间段
            schedule_all[schedule_all_idx][timeslot1], schedule_all[schedule_all_idx][timeslot2] = \
                schedule_all[schedule_all_idx][timeslot2], schedule_all[schedule_all_idx][timeslot1]

    return [schedule, class_schedule, teacher_schedule]

def A_random(res, cr, tr,schedule_all):
    for x in range(1):
        used_classroom_indices = []
        for classroom_index in range(len(res)):
            if any(res[classroom_index]):
                used_classroom_indices.append(classroom_index)

        while True:
            p1 = random.randint(0, 24)
            M = random.choice(used_classroom_indices)
            if res[M][p1]:
                break

        used_classroom_indices1 = []
        for classroom_index in used_classroom_indices:
            if classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                used_classroom_indices1.append(classroom_index)
        n = 0
        while n < 100: #检查的最大次数
            n = n + 1
            # 从列表中随机选择一个整数
            p = random.randint(0, 24)
            r = random.choice(used_classroom_indices1)
            if (res[r][p] and (res[r][p]['course'] != res[M][p1]['course'] and classrooms[M]['type'] in res[r][p][
                'allowed_classrooms'])) or schedule_all[r][p] is None:
                if check(r, p, M, p1, res, cr, tr):
                    res[r][p], res[M][p1] = res[M][p1], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    break
    return [res, cr, tr]



def check(r1, p1, r2, p2, res, cr, tr):
    if p1 in tr[res[r2][p2]['teacher']]:
        # print("教师时间冲突2")
        return False
    elif res[r1][p1] and p2 in tr[res[r1][p1]['teacher']]:
        # print("教师时间冲突1")
        return False

    if res[r1][p1]:
        for class_id in res[r1][p1]['class']:
            if p2 in cr[class_id]:
                # print("班级时间冲突1")
                return False

    for class_id in res[r2][p2]['class']:
        if p1 in cr[class_id]:
            # print("班级时间冲突2")
            return False

    if p2 in tr[res[r2][p2]['teacher']]:
        tr[res[r2][p2]['teacher']].remove(p2)
        tr[res[r2][p2]['teacher']].add(p1)

    if res[r1][p1]:
        if p1 in tr[res[r1][p1]['teacher']]:
            tr[res[r1][p1]['teacher']].remove(p1)
            tr[res[r1][p1]['teacher']].add(p2)

    for class_id in res[r2][p2]['class']:
        if p2 in cr[class_id]:
            cr[class_id].remove(p2)
            cr[class_id].add(p1)
    if res[r1][p1]:
        for class_id in res[r1][p1]['class']:
            if p1 in cr[class_id]:
                cr[class_id].remove(p1)
                cr[class_id].add(p2)

    return cr, tr


def B_night(res, cr, tr,schedule_all):
    for x in range(5):
        p1 = random.choice([4, 9, 14, 19, 24])
        M = 0
        V = 0
        for i in range(len(classrooms)):
            if res[i][p1] is not None and res[i][p1]['time'] * len(res[i][p1]['class']) > V:
                M = i
                V = res[i][p1]['time'] * len(res[i][p1]['class'])
        if res[M][p1] is None:
            continue

        nums = [i for i in range(25) if i not in [4, 9, 14, 19, 24]]
        used_classroom_indices = []
        for classroom_index in range(len(res)):
            if any(res[classroom_index]) and classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                used_classroom_indices.append(classroom_index)
        sign = True
        n = 0
        while sign and n < 100:
            n = n + 1
            # 从列表中随机选择一个整数
            p = random.choice(nums)
            r = random.choice(used_classroom_indices)

            if (res[r][p] and res[r][p]['time'] * len(res[r][p]['class']) < V) or schedule_all[r][p] is None:
                if check(r, p, M, p1, res, cr, tr):
                    res[r][p], res[M][p1] = res[M][p1], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    break
    return [res, cr, tr]


def B_course(res, cr, tr,schedule_all):
    for x in range(2):
        class_course_days = defaultdict(lambda: defaultdict(set))
        re_list = []
        for ord, classroom in enumerate(res):
            for timeslot, course_info in enumerate(classroom):
                if course_info:
                    day = timeslot // timeslots_per_day
                    sign = 0
                    for class_id in course_info['class']:
                        for c in class_course_days[class_id][day]:
                            if course_info['course'] == c[0]:
                                sign = 1
                                re_list.append([ord, timeslot])
                        if sign == 0:
                            class_course_days[class_id][day].add((course_info['course'], course_info['time']))
        if len(re_list)==0:
            return [res,cr,tr]

        first = random.choice(re_list)
        M, p1 = first[0], first[1]
        used_classroom_indices = []
        for classroom_index in range(len(res)):
            if any(res[classroom_index]) and classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                used_classroom_indices.append(classroom_index)
        n = 0
        while n < 100:
            n = n + 1
            # 从列表中随机选择一个整数
            p = random.randint(0, 24)
            r = random.choice(used_classroom_indices)
            if (res[r][p] and (res[r][p]['course'] != res[M][p1]['course'] and classrooms[M]['type'] in res[r][p][
                'allowed_classrooms'])) or schedule_all[r][p] is None:
                if check(r, p, M, p1, res, cr, tr):
                    res[r][p], res[M][p1] = res[M][p1], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    break
    return [res, cr, tr]


def B_c_course(res, cr, tr,courses,schedule_all):

    for x in range(10):
        while True:
            used_classroom_indices=[]
            used_indices=[]
            for i in range(len(res)):
                for j in range(0,25):
                    if res[i][j]:
                        used_indices.append([i,j])
                        used_classroom_indices.append(i)

            u = random.choice(used_indices)
            M,p1=u[0],u[1]
            used_classroom_indices1 = []
            for classroom_index in used_classroom_indices:
                if classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                    used_classroom_indices1.append(classroom_index)

            like_time = []
            for course in courses:
                if res[M][p1]['course'] == course['name']:
                    if course['like'][p1 % 5] == -1:
                        for index, c in enumerate(course['like']):
                            if c == 1 or c == 0:
                                like_time.append(index)
                    elif course['like'][p1 % 5] == 0:
                        for index, c in enumerate(course['like']):
                            if c == 1:
                                like_time.append(index)
                    break
            if len(like_time) > 0:
                break

        nums = [i for i in range(25) if i % 5 in like_time]

        n = 0
        while n < 100:
            n = n + 1
            # 从列表中随机选择一个整数
            p = random.choice(nums)
            r = random.choice(used_classroom_indices1)
            if (res[r][p] and (res[r][p]['course'] != res[M][p1]['course'] and classrooms[M]['type'] in res[r][p][
                'allowed_classrooms'])) or schedule_all[r][p] is None:
                if check(r, p, M, p1, res, cr, tr):
                    res[r][p], res[M][p1] = res[M][p1], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    break
    return [res, cr, tr]



def B_c_teacher(res, cr, tr, teachers, schedule_all):
    # 初始化数据结构和缓存
    teacher_like_cache = {}
    course_room_cache = {}
    time_period_to_day = {i: i // 5 for i in range(25)}

    # 按课程和教师分组的时间段索引，加速候选查找
    course_slots = defaultdict(set)  # 使用集合代替列表
    teacher_slots = defaultdict(set)
    used_indices = []

    # 初始化索引缓存
    for i in range(len(res)):
        for j in range(25):
            if res[i][j]:
                used_indices.append((i, j))
                course_slots[res[i][j]['course']].add((i, j))
                teacher_slots[res[i][j]['teacher']].add((i, j))

    # 预计算教室类型
    classroom_types = [c['type'] for c in classrooms]

    # 预计算教师信息映射
    teacher_map = {t['name']: t for t in teachers}

    for _ in range(5):
        if not used_indices:
            break

        M, p1 = random.choice(used_indices)
        day_p1 = time_period_to_day[p1]
        current_course = res[M][p1]['course']
        teacher_name = res[M][p1]['teacher']

        # 获取或缓存课程允许的教室类型
        if current_course not in course_room_cache:
            course_room_cache[current_course] = set(res[M][p1]['allowed_classrooms'])

        allowed_rooms_M = course_room_cache[current_course]
        room_type_M = classroom_types[M]  # 使用预计算的教室类型

        # 获取教师喜欢的时间段
        if teacher_name not in teacher_like_cache:
            teacher = teacher_map[teacher_name]  # 使用预计算的教师映射
            like1 = teacher['like1']
            current_preference = like1[day_p1]

            if current_preference == -1:
                like_days = [i for i, t in enumerate(like1) if t >= 0]
            elif current_preference == 0:
                like_days = [i for i, t in enumerate(like1) if t >= 1]
            else:
                like_days = []

            teacher_like_cache[teacher_name] = like_days
        else:
            like_days = teacher_like_cache[teacher_name]

        if not like_days:
            continue

        # 从教师喜欢的天中随机选择一天
        target_day = next((d for d in like_days if d != day_p1), None)
        if target_day is None:
            continue

        # 从目标天获取候选时间段
        target_slots = [
            (i, j) for i, j in teacher_slots.get(teacher_name, [])
            if time_period_to_day[j] == target_day and
               res[i][j]['course'] != current_course
        ]

        if not target_slots:
            continue

        # 随机打乱并限制尝试次数
        random.shuffle(target_slots)

        for r, p in target_slots[:20]:  # 进一步减少尝试次数
            other_course = res[r][p]['course']

            # 检查教室类型兼容性
            if classroom_types[r] not in allowed_rooms_M:  # 使用预计算的教室类型
                continue

            # 缓存并检查对方课程允许的教室类型
            if other_course not in course_room_cache:
                course_room_cache[other_course] = set(res[r][p]['allowed_classrooms'])

            if room_type_M not in course_room_cache[other_course]:
                continue

            # 执行检查并交换
            if check(r, p, M, p1, res, cr, tr):
                # 交换课程
                res[r][p], res[M][p1] = res[M][p1], res[r][p]
                schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]

                # 更新索引缓存
                course_slots[current_course].remove((M, p1))
                course_slots[current_course].add((r, p))
                course_slots[other_course].remove((r, p))
                course_slots[other_course].add((M, p1))

                break

    return [res, cr, tr]


def B_method(res, cr, tr, teachers, schedule_all):
    teacher_like3_cache = {}  # 缓存教师like3相关数据
    used_classrooms = [i for i in range(len(res)) if any(res[i])]
    pairs = [(0, 1), (2, 3), (5, 6), (7, 8), (10, 11), (12, 13), (15, 16), (17, 18), (20, 21), (22, 23)]

    for _ in range(2):
        if not used_classrooms:
            break

        M = random.choice(used_classrooms)
        used_times = [i for i in range(25) if res[M][i]]

        if not used_times:
            continue

        p1 = random.choice(used_times)
        teacher_name = res[M][p1]['teacher']

        # 获取教师like3相关数据
        if teacher_name not in teacher_like3_cache:
            teacher = next(t for t in teachers if t['name'] == teacher_name)
            like3 = teacher['like3']
            teacher_tr = tr[teacher_name]

            if like3 == 1:
                like_time = []
                no = []

                for pair in pairs:
                    a, b = pair
                    if a in teacher_tr and b not in teacher_tr and res[M][b]:
                        like_time.append(b)
                    elif a not in teacher_tr and b in teacher_tr and res[M][a]:
                        like_time.append(a)
                    elif a in teacher_tr and b in teacher_tr:
                        no.append(a)
                        no.append(b)
            else:
                like_time = []
                no = []

            teacher_like3_cache[teacher_name] = (like_time, no)
        else:
            like_time, no = teacher_like3_cache[teacher_name]

        if not like_time:
            continue

        sign = True
        while like_time and sign:
            p = random.choice(like_time)
            like_time.remove(p)

            # 更新no列表
            for pair in pairs:
                if pair[0] == p:
                    no.append(pair[1])
                elif pair[1] == p:
                    no.append(pair[0])

            # 生成swap_list
            swap_list = [t for t in tr[teacher_name] if t not in no]

            while swap_list:
                p1_swap = random.choice(swap_list)
                swap_list.remove(p1_swap)

                # 找到包含p1_swap的教室
                classroom_index = next(
                    (i for i in used_classrooms if res[i][p1_swap] and res[i][p1_swap]['teacher'] == teacher_name),
                    None
                )

                if classroom_index is None:
                    continue

                M = classroom_index

                # 优化：预先筛选可用教室索引
                used_classroom_indices1 = [
                    i for i in used_classrooms
                    if res[i][p] and
                       classrooms[i]['type'] in res[M][p1_swap]['allowed_classrooms']
                ]

                if not used_classroom_indices1:
                    continue

                r = random.choice(used_classroom_indices1)

                if check(r, p, M, p1_swap, res, cr, tr):
                    # 交换课程
                    res[r][p], res[M][p1_swap] = res[M][p1_swap], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1_swap] = schedule_all[M][p1_swap], schedule_all[r][p]
                    sign = False
                    break

            if not sign:
                break

    return [res, cr, tr]


# 遗传的超启发式算法主函数
def GA_RG_HH(paths, max_iterations=10, population_size=20, crossover_rate=0.8, mutation_rate=0.2):

    print("GA_RG_HH生成初始解中...")
    population = [generate_initial_solution(paths) for _ in range(population_size)]
    print("GA_RG_HH生成完毕")
    best_fitness_per_iteration = []

    # 终止条件需根据实际情况定义，这里先以迭代次数为例
    best_v = float('inf')
    fitness = np.full(population_size, np.inf)  # 初始适应度设为无穷大

    for ind,pop in enumerate(population):
        obj_value=0
        for i, path in enumerate(paths):
            obj_value = obj_func_all(pop[i+1][0], path, classrooms, pop[i+1][2]) + obj_value
        fitness[ind]=obj_value
        if obj_value<best_v:
            best_v = obj_value
            near_optimal_solution = copy.deepcopy(population[0])

    num=0
    for index, classroom in enumerate(classrooms):
        if (any(near_optimal_solution[1][5][index])):
            num = num + 1
    best_fitness_per_iteration.append(best_v + num * 50)

    print(fitness)
    print("GA_RG_HH初始最优解：", best_v + num * 50)


    for iteration in range(max_iterations):
        new_population = []
        # indexed_lst = list(enumerate(fitness))
        # sorted_lst = sorted(indexed_lst, key=lambda x: x[1])
        # solution_a, solution_b = population[sorted_lst[0][0]], population[sorted_lst[1][0]]
        #
        # if iteration % (max_iterations // 5) == 0 and iteration != 0 and len(solution_a[0]) > 1:
        #     solution_a[0] = solution_a[0][0:len(solution_a[0]) - 1]
        #     solution_b[0] = solution_b[0][0:len(solution_b[0]) - 1]

        for index in range(population_size):
            # 选择操作
            solution_a, solution_b = selection(population,fitness)
            if iteration % (max_iterations // 10) == 0 and iteration != 0 and len(solution_a[0]) > 1:
                solution_a[0] = solution_a[0][0:len(solution_a[0]) - 1]
                solution_b[0] = solution_b[0][0:len(solution_b[0]) - 1]


            lva = copy.deepcopy(solution_a[0])  #这里假设低层次启发式向量就是解本身，需根据实际调整
            lva1=copy.deepcopy(solution_b[0])


            # 交叉操作
            if(len(lva)!=1 and len(lva1)!=1):

                if random.random() < crossover_rate:
                    lva = one_point_crossover(lva, lva1)
                # 变异操作
                if random.random() < mutation_rate:
                    lva = one_point_mutation(lva)
            else:
                # lva[0] = random.randint(0, 8)
                if iteration < 2*max_iterations/3:
                    lva[0]=random.randint(3,8)
                else:
                    lva[0]=3


            # 生成新解
            sva = update_sva(lva, copy.deepcopy(solution_a[1:len(solution_a)]))
            new_solution = copy.deepcopy(sva)

            new_v=0
            for i, path in enumerate(paths):
                new_v = obj_func_all(new_solution[i][0], path, classrooms, new_solution[i][2]) + new_v
            # print(lva)
            if new_v < fitness[index]:

                fitness[index] = new_v
                sign=[lva]
                for new in new_solution:
                    sign.append(new)
                new_population.append(sign)

                if new_v < best_v:
                    best_v = new_v
                    near_optimal_solution = copy.deepcopy(new_population[index])
            else:
                new_population.append(copy.deepcopy(solution_a))

        num = 0
        for index, classroom in enumerate(classrooms):
            if (any(near_optimal_solution[1][5][index])):
                num = num + 1

        print("GA_RG_HH_iter:", iteration, ",best_v:", best_v+num*50, ",best_o:", near_optimal_solution[0])
        best_fitness_per_iteration.append(best_v+num*50)
        population = new_population

    detail=[0,0,0,0,0,0]
    for i, path in enumerate(paths):
        de = obj_func_detail(near_optimal_solution[i+1][0], path, classrooms, near_optimal_solution[i+1][2])
        for j,deta in enumerate(detail):
            detail[j]=detail[j]+de[j]
    detail.append(num)
    detail[1]=detail[1]/len(paths)
    print(detail)
    return near_optimal_solution[1:len(near_optimal_solution)], best_v+num*50, best_fitness_per_iteration,num,detail


if __name__ == "__main__":
    paths = [
        r'data\large\data1_4.xlsx',
        # r'data\large\data2.xlsx',
    ]
    # paths = [
    #     r'data\small\l_data05.xlsx',
    # ]

    start = time.time()
    best_path, best_v, x,num,detail = GA_RG_HH(paths,300)
    end = time.time()
    print("生成解的时间：", end - start)
    print("最短距离:", best_v)
    print("教室数量：", num)
    for i, path in enumerate(paths):
        obj_func_all_print(best_path[i][0], path, classrooms, best_path[i][2])

    for i, path in enumerate(paths):
        plot_schedule(best_path[i][0],classrooms)


    plt.figure(figsize=(10, 6))
    plt.plot(x, label='GA_RG_HH')

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.title('不同算法的适应度值迭代过程')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.legend()
    plt.show()


