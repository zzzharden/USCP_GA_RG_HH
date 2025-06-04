import copy
import random
import math
import time
from collections import defaultdict

from obj_function import obj_func_all, obj_func_all_print, obj_func_detail, obj_func_time
import numpy as np
from data_loader import data,data1,data2
from plot import plot_schedule
import matplotlib.pyplot as plt

total_weeks = 16
days_per_week = 5
timeslots_per_day = 5
total_timeslots = days_per_week * timeslots_per_day
classrooms = data2(r'C:\Users\zhuzizhuang\Desktop\classrooms.xlsx')
def schedule_courses(courses_task, classes, teachers, schedule_all):
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
                    'allowed_classrooms': course['allowed_classrooms'],
                    'time': dtime
                }
                schedule_all[classroom_index][start_timeslot] = {
                    'id': course['id'], 'course': course['course'], 'class': course['class'],
                    'teacher': course['teacher'], 'duration': course['duration'],
                    'allowed_classrooms': course['allowed_classrooms'],
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

def initialize_population(paths):
    pop = []
    schedule_all = np.full((len(classrooms), total_timeslots), None)
    for i, path in enumerate(paths):
        college_name = f'学院{i + 1}'
        courses_task, teachers, classes, courses = data1(path)
        res, cr, tr = schedule_courses(courses_task, classes, teachers, schedule_all)
        pop.append([res, cr, tr, teachers, courses, schedule_all])
    return pop

def check(r1, p1, r2, p2, res,cr,tr):

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

    return cr,tr

def A_random_new(res,cr,tr,schedule_all):
    for x in range(1):
        count_dict = {}
        for index, r in enumerate(res):
            num = sum(1 for t in r if t)
            if num > 0:
                count_dict[index] = num

        # 根据使用次数排序，选择最少的5个
        sorted_items = sorted(count_dict.items(), key=lambda x: x[1])
        used_classroom_indices = [item[0] for item in sorted_items[:5]]

        r1=random.choice(used_classroom_indices)
        T1=[]
        for j in range(0,25):
            if res[r1][j]:
                T1.append(j)
        t1=random.choice(T1)

        used_classroom_indices1 = []
        for classroom_index in count_dict:
            if classrooms[classroom_index]['type'] in res[r1][t1]['allowed_classrooms']:
                used_classroom_indices1.append(classroom_index)

        T2=[j for j in range(25)]
        sign=False
        while len(T2)>0:
            t2=random.choice(T2)
            R2=copy.deepcopy(used_classroom_indices1)
            while len(R2)>0:
                r2=random.choice(R2)
                if (res[r2][t2] and (res[r2][t2]['course'] != res[r1][t1]['course'] and classrooms[r1]['type'] in res[r2][t2][
                'allowed_classrooms'])) or schedule_all[r2][t2] is None:
                    if check(r2, t2, r1, t1, res, cr, tr):
                        res[r1][t1], res[r2][t2] = res[r2][t2], res[r1][t1]
                        schedule_all[r1][t1], schedule_all[r2][t2] = schedule_all[r2][t2], schedule_all[r1][t1]
                        sign = True
                        break
                R2.remove(r2)
            if sign:
                break
            T2.remove(t2)
    return [res,cr,tr]

def A_random(res,cr,tr,schedule_all):
    for x in range(1):
        count_dict = {}
        for index, r in enumerate(res):
            num = sum(1 for t in r if t)
            if num > 0:
                count_dict[index] = num

        # 根据使用次数排序，选择最少的5个
        sorted_items = sorted(count_dict.items(), key=lambda x: x[1])
        used_classroom_indices = [item[0] for item in sorted_items[:5]]

        M=random.choice(used_classroom_indices)
        time=[]
        for j in range(0,25):
            if res[M][j]:
                time.append(j)
        p1=random.choice(time)

        used_classroom_indices1 = []
        for classroom_index in count_dict:
            if classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                used_classroom_indices1.append(classroom_index)

        n = 0
        while  n<100:
            n=n+1
            # 从列表中随机选择一个整数
            p = random.randint(0,24)
            r = random.choice(used_classroom_indices1)
            if (res[r][p] and (res[r][p]['course'] != res[M][p1]['course'] and classrooms[M]['type'] in res[r][p][
                'allowed_classrooms'])) or schedule_all[r][p] is None:
                if check(r, p, M, p1, res,cr,tr):
                    res[r][p],res[M][p1]=res[M][p1],res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    break
    return [res,cr,tr]

def B_night(res, cr, tr,schedule_all):
    for x in range(5):
        t1 = random.choice([4, 9, 14, 19, 24])
        r1=0
        V = 0
        for i in range(len(classrooms)):
            if res[i][t1] is not None and res[i][t1]['time'] * len(res[i][t1]['class']) > V:
                r1 = i
                V = res[i][t1]['time'] * len(res[i][t1]['class'])
        if res[r1][t1] is None:
            continue

        used_classroom_indices = []
        for classroom_index in range(len(res)):
            if any(res[classroom_index]) and classrooms[classroom_index]['type'] in res[r1][t1]['allowed_classrooms']:
                used_classroom_indices.append(classroom_index)

        T2=[i for i in range(25) if i not in [4, 9, 14, 19, 24]]

        sign=False
        while len(T2)>0:
            t2=random.choice(T2)
            R2=copy.deepcopy(used_classroom_indices)
            while len(R2)>0:
                r2=random.choice(R2)
                if (res[r2][t2] and res[r2][t2]['time'] * len(res[r2][t2]['class']) < V) or schedule_all[r2][t2] is None:
                    if check(r2, t2, r1, t1, res, cr, tr):
                        res[r1][t1], res[r2][t2] = res[r2][t2], res[r1][t1]
                        schedule_all[r1][t1], schedule_all[r2][t2] = schedule_all[r2][t2], schedule_all[r1][t1]
                        sign = True
                        break
                R2.remove(r2)
            if sign:
                break
            T2.remove(t2)

    return [res, cr, tr]


def B_course(res, cr, tr,schedule_all):
    for x in range(5):
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
        r1, t1 = first[0], first[1]
        used_classroom_indices = []
        for classroom_index in range(len(res)):
            if any(res[classroom_index]) and classrooms[classroom_index]['type'] in res[r1][t1]['allowed_classrooms']:
                used_classroom_indices.append(classroom_index)

        T2 = [j for j in range(25)]
        sign = False
        while len(T2) > 0:
            t2 = random.choice(T2)
            R2=copy.deepcopy(used_classroom_indices)
            while len(R2) > 0:
                r2 = random.choice(R2)
                if (res[r2][t2] and (
                        res[r2][t2]['course'] != res[r1][t1]['course'] and classrooms[r1]['type'] in res[r2][t2][
                    'allowed_classrooms'])) or schedule_all[r2][t2] is None:
                    if check(r2, t2, r1, t1, res, cr, tr):
                        res[r1][t1],res[r2][t2]=res[r2][t2],res[r1][t1]
                        schedule_all[r1][t1], schedule_all[r2][t2] = schedule_all[r2][t2], schedule_all[r1][t1]
                        sign=True
                        break
                R2.remove(r2)
            if sign:
                break
            T2.remove(t2)
    return [res, cr, tr]


def B_c_course(res, cr, tr,courses,schedule_all):

    for x in range(1):
        while True:
            used_classroom_indices=[]
            used_indices=[]
            for i in range(len(res)):
                for j in range(0,25):
                    if res[i][j]:
                        used_indices.append([i,j])
                        used_classroom_indices.append(i)

            u = random.choice(used_indices)
            r1,t1=u[0],u[1]
            used_classroom_indices1 = []
            for classroom_index in used_classroom_indices:
                if classrooms[classroom_index]['type'] in res[r1][t1]['allowed_classrooms']:
                    used_classroom_indices1.append(classroom_index)

            like_time = []
            for course in courses:
                if res[r1][t1]['course'] == course['name']:
                    if course['like'][t1 % 5] == -1:
                        for index, c in enumerate(course['like']):
                            if c == 1 or c == 0:
                                like_time.append(index)
                    elif course['like'][t1 % 5] == 0:
                        for index, c in enumerate(course['like']):
                            if c == 1:
                                like_time.append(index)
                    break
            if len(like_time) > 0:
                break


        T2 = [i for i in range(25) if i % 5 in like_time]
        sign = False
        while len(T2) > 0:
            t2 = random.choice(T2)
            R2=copy.deepcopy(used_classroom_indices1)
            while len(R2) > 0:
                r2 = random.choice(R2)
                if (res[r2][t2] and (
                        res[r2][t2]['course'] != res[r1][t1]['course'] and classrooms[r1]['type'] in res[r2][t2][
                    'allowed_classrooms'])) or schedule_all[r2][t2] is None:
                    if check(r2, t2, r1, t1, res, cr, tr):
                        res[r1][t1], res[r2][t2] = res[r2][t2], res[r1][t1]
                        schedule_all[r1][t1], schedule_all[r2][t2] = schedule_all[r2][t2], schedule_all[r1][t1]
                        sign = True
                        break
                R2.remove(r2)
            if sign:
                break
            T2.remove(t2)
    return [res, cr, tr]


def B_c_teacher(res, cr, tr,teachers,schedule_all):
    for x in range(1):
        while True:
            used_classroom_indices=[]
            used_indices=[]
            for i in range(len(res)):
                for j in range(0,25):
                    if res[i][j]:
                        used_indices.append([i,j])
                        used_classroom_indices.append(i)
            u = random.choice(used_indices)

            r1,t1=u[0],u[1]
            used_classroom_indices1 = []
            for classroom_index in used_classroom_indices:
                if classrooms[classroom_index]['type'] in res[r1][t1]['allowed_classrooms']:
                    used_classroom_indices1.append(classroom_index)

            like_time = []
            for teacher in teachers:
                if res[r1][t1]['teacher'] == teacher['name']:
                    if teacher['like1'][t1 // 5] == -1:
                        for index1, lt in enumerate(teacher['like1']):
                            if lt >= 0:
                                like_time.append(index1)

                    elif teacher['like1'][t1 // 5] == 0:
                        for index1, lt in enumerate(teacher['like1']):
                            if lt  >= 1:
                                like_time.append(index1)

                    break
            if len(like_time) > 0:
                break
        T2 = [i for i in range(25) if i // 5 in like_time]
        sign = False
        while len(T2) > 0:
            t2 = random.choice(T2)
            R2=copy.deepcopy(used_classroom_indices1)
            while len(R2) > 0:
                r2 = random.choice(R2)
                if (res[r2][t2] and (
                        res[r2][t2]['course'] != res[r1][t1]['course'] and classrooms[r1]['type'] in res[r2][t2][
                    'allowed_classrooms'])) or schedule_all[r2][t2] is None:
                    if check(r2, t2, r1, t1, res, cr, tr):
                        res[r1][t1], res[r2][t2] = res[r2][t2], res[r1][t1]
                        schedule_all[r1][t1], schedule_all[r2][t2] = schedule_all[r2][t2], schedule_all[r1][t1]
                        sign = True
                        break
                R2.remove(r2)
            if sign:
                break
            T2.remove(t2)
    return [res, cr, tr]


def B_method(res, cr, tr,teachers,schedule_all):
    used_classroom_indices = []
    pairs = [(0, 1), (2, 3), (5, 6), (7, 8), (10, 11), (12, 13), (15, 16), (17, 18), (20, 21), (22, 23)]
    for classroom_index in range(len(res)):
        if any(res[classroom_index]):
            used_classroom_indices.append(classroom_index)
    # print("cr:",used_classroom_indices)
    for x in range(5):
        while True:
            M = random.choice(used_classroom_indices)
            used_classroom_time_indices = []
            for i in range(25):
                if res[M][i]:
                    used_classroom_time_indices.append(i)
            if used_classroom_time_indices is None:
                print(classrooms['id'][M])
                plot_schedule(res,classrooms)
                break

            p1 = random.choice(used_classroom_time_indices)
            tt = res[M][p1]['teacher']

            like_time = []
            no = []
            swap_list = []
            for teacher in teachers:
                if res[M][p1]['teacher'] == teacher['name']:

                    if teacher['like3'] == 1:
                        for pair in pairs:
                            # 检查集合中是否同时包含括号内的两个数字
                            if pair[0] in tr[teacher['name']] and pair[1] not in tr[teacher['name']]:
                                if res[M][pair[1]]:
                                    like_time.append(pair[1])
                            elif pair[0] not in tr[teacher['name']] and pair[1] in tr[teacher['name']]:
                                if res[M][pair[0]]:
                                    like_time.append(pair[0])
                            elif pair[0]  in tr[teacher['name']] and pair[1] in tr[teacher['name']]:
                                no.append(pair[0])
                                no.append(pair[1])
                        break

            if len(like_time) > 0:
                break

        sign = True
        while len(like_time) > 0 and sign:

            p = random.choice(like_time)
            for pair in pairs:
                if pair[0] == p:
                    no.append(pair[1])
                elif pair[1] == p:
                    no.append(pair[0])

            for t in tr[teacher['name']]:
                if t not in no:
                    swap_list.append(t)
            like_time.remove(p)

            while len(swap_list) > 0:
                p1 = random.choice(swap_list)
                swap_list.remove(p1)

                for classroom_index in used_classroom_indices:
                    if res[classroom_index][p1] and res[classroom_index][p1]['teacher'] == tt:
                        M = classroom_index
                        break
                if res[M][p1] is None:
                    break
                used_classroom_indices1 = []
                for classroom_index in used_classroom_indices:
                    if classrooms[classroom_index]['type'] in res[M][p1]['allowed_classrooms']:
                        used_classroom_indices1.append(classroom_index)

                # 从列表中随机选择一个整数
                r = random.choice(used_classroom_indices1)
                if res[r][p] and check(r, p, M, p1, res, cr, tr):
                    res[r][p], res[M][p1] = res[M][p1], res[r][p]
                    schedule_all[r][p], schedule_all[M][p1] = schedule_all[M][p1], schedule_all[r][p]
                    sign = False
                    break

    return [res, cr, tr]


paths = [
    r'data\large\data1_4.xlsx',

]
start=time.time()
res=initialize_population(paths)
end=time.time()
print("生成时间:",end-start)

a=0
b=0
for i in range(10000):
    a=a+A_random(res[0][0],res[0][1],res[0][2],res[0][5])
    b=b+A_random_new(res[0][0],res[0][1],res[0][2],res[0][5])
print(a,b)



# obj_value = 0
# for i, path in enumerate(paths):
#     obj_value = obj_func_all_print(res[i][0], path, classrooms, res[i][2]) + obj_value
# print(obj_value)
#
# B_c_teacher(res[0][0],res[0][1],res[0][2],res[0][3],res[0][5])
#
# obj_value = 0
# for i, path in enumerate(paths):
#     obj_value = obj_func_all_print(res[i][0], path, classrooms, res[i][2]) + obj_value
# print(obj_value)

# p7 = 0
# for index in range(0, len(res[0][0])):
#     if any(res[0][0][index]):
#         p7 = p7 + 1
# print("初始教室数量----------------------------------------：",p7)
#
# i=0
# while True:
#     i=i+1
#     A_random(res[0][0],res[0][1],res[0][2],res[0][4])
#     cr = 0
#     for index in range(0, len(res[0][0])):
#         if any(res[0][0][index]):
#             cr = cr + 1
#     print("次数：", i, ",当前教室数量:", cr)
#     if cr<p7:
#         print("-------------------------次数：",i,",当前教室数量:",cr)
#         break



