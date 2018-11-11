# -*- coding:utf-8 -*-
"""
    using heap.
"""


# heapq.heappush(heap,item)

class Heap():

    def __init__(self):

        pass

    def insert(self, heap_m, value):
        heap_m.append(value)

        i = len(heap_m) // 2

        self.max_heap(heap_m, i)
        # if heap_m =='':
        #     heap_m.node = value
        #     return heap_m
        #
        # if heap_m.node < value:
        #     heap_m.append(self.insert(heap_m.right_node, value))
        # else:
        #     heap_m.append(self.insert(heap_m.left_node, value))

        return heap_m

    def adjust_heap(self, heap_m):
        # find v, then from bottom to up to adjust heap
        pass

    def max_heap(self, lst, idx):
        largest = idx
        left = 2 * idx + 1
        right = 2 * idx + 2

        if left < largest and lst[left] > lst[largest]:
            largest = left
        if right < largest and lst[right] > lst[largest]:
            largest = left

        if largest != idx:
            lst[idx], lst[largest] = lst[largest], lst[idx]
            self.max_heap(lst, largest)

        return lst

    def build_heap(self, in_lst):
        self.heap_m = ''
        # for i, v in enumerate(in_lst):
        #     self.insert(self.heap_m, v)
        #     self.adjust_heap(self.heap_m,v)
        for i in range(len(in_lst), -1, -2):
            self.heap_m = self.max_heap(in_lst, i)

    # def max_heapify(A, i):
    #     left = 2 * i + 1
    #     right = 2 * i + 2
    #     largest = i
    #     if left < len(A) and A[left] > A[largest]:
    #         largest = left
    #     if right < len(A) and A[right] > A[largest]:
    #         largest = right
    #     if largest != i:
    #         A[i], A[largest] = A[largest], A[i]
    #         max_heapify(A, largest)
    #
    # def build_max_heap(A):
    #     for i in range(len(A) // 2, -1, -1):
    #         max_heapify(A, i)

    def pop(self):
        pass


if __name__ == '__main__':
    in_lst = [3, 2, 4, 1, 6]
    heap = Heap()
    heap.build_heap(in_lst)

    heap.insert(heap.heap_m, 10)
