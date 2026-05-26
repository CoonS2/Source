print("안녕하세요!")

# 줄 단위 실행 > 특정 행에서 실행 오류가 나는 경우 프로그램은 멈추게 된다.
# 파이썬 자료형 = 정수형, 문자형, 불린형 , 리스트 , 튜플 , 딕셔너리, set(x)

# 변수 : 프로그램 내부에 값을 담아놓기위한 공간(이름 사용)
a = 10  # 정수형 (Integer)
b = 3.14 # 실수형 (Floating-point number)
c = "Hello, World!" # 문자열형 (String)
d = True # 불리언형 (Boolean) - 참(True) 또는 거짓(False) 값을 가질 수 있다.
e = [1, 2, 3, 4, 5] # 리스트형 (List) - 여러 값을 순서대로 저장할 수 있는 자료형
f = (1, 2, 3) # 튜플형 (Tuple) - 리스트와 유사하지만, 한 번 생성된 후에는 변경할 수 없는 자료형
g = {1, 2, 3} # 집합형 (Set) - 중복되지 않는 요소들의 모음으로, 순서가 없다.
h = {"name": "Alice", "age": 30} # 딕셔너리형 (Dictionary) - 키-값 쌍으로 데이터를 저장하는 자료형

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)

# \n == enter
multiline = "Life is too short\n you need python"
print(multiline)

# import mod1
# print(mod1.add(5,3))
# print(mod1.sub(5,3))

# from mod1 import add
# print(add(3,7))

# * : 모두
from mod1 import *
print(add(3,7))
print(sub(3,7))