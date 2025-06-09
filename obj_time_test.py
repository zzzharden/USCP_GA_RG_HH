import copy
import random
import math
import time

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
                if probability < 0.97 and used_classroom_timeslots:
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
    c_sol = []
    schedule_all = np.full((len(classrooms), total_timeslots), None)
    for i, path in enumerate(paths):
        college_name = f'学院{i + 1}'
        courses_task, teachers, classes, _ = data1(path)
        res, cr, tr = schedule_courses(courses_task, classes, teachers,schedule_all)
        c_sol.append([res, cr, tr, path,schedule_all])
    return c_sol

paths = [
    r'large\data1_4.xlsx',
    # r'large\data2.xlsx',
    # r'large\data3.xlsx',
    # r'large\data4.xlsx',
    # r'large\data5.xlsx',
    # r'large\data6.xlsx',
    # r'large\data7.xlsx',
    # r'large\data8.xlsx',
    # r'large\data9.xlsx',
    # r'large\data10.xlsx',
]
start=time.time()
res=initialize_population(paths)
end=time.time()

print("生成时间:",end-start)

start=time.time()
obj_value = 0
res_t=[0,0,0,0,0,0]
for i, path in enumerate(paths):
    obj_value = obj_func_time(res[0][0],paths[0],res[0][2]) + obj_value

end=time.time()

print("计算时间:",end-start)
