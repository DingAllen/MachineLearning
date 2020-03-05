score = int(input('学生的成绩为：'))
grade = ''

if (not isinstance(score,int)) or score > 100 or score < 0:
    print('请输入1-100以内的整数')
    exit(0)
elif score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
elif score >= 60:
    grade = 'D'
else:
    grade = 'E'

print('学生的成绩等级为：' + grade)