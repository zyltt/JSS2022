# coding=utf-8

import queue
class GenetypeQueue:
    q=queue.Queue()
    def push(self,a):
        self.q.put(a)
    def count(self):
        return self.q.qsize()
    def pop(self):
        return self.q.get()
    def empty(self):
        return self.q.empty()
#调用测试的代码
# Q=genetypeQueue()
# g=Genetype(level=6)
# Q.push(g)
# print(Q.count())
# print(Q.empty())
# g2=Q.pop()
# print(Q.count())
# print(Q.empty())