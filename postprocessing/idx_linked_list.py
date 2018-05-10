class ListNode:
    def __init__(self,idx,offset):
        self.idx = idx
        self.offset = offset
        self.next = None

class IdxList:
    def __init__(self):
        self.root = None

    def add(self,idx,offset): # новый индекс, смещение от нового к старому
        node_to_add = ListNode(idx,offset)
        if self.root is None:
            self.root = node_to_add
        else:
            node = self.root
            while node is not None and node.idx < idx+offset: # ищем по старому индексу
                prev_node = node
                node = node.next
            prev_node.next = node_to_add
            node_to_add.next = node
            while node is not None:
                node.idx -= offset
                node = node.next
            # дописать условие про отрицительный оффсет

    def find_old_idx(self,new_idx):
        total_offset = 0
        node = self.root
        while node is not None and node.idx <= new_idx:
            total_offset += node.offset
            node = node.next
        return new_idx + total_offset

    def __repr__(self):
        lst = []
        node = self.root
        while node is not None:
            lst.append(' '.join([str(node.idx),str(node.offset)]))
            node = node.next
        return ', '.join(lst)


    def add_batch(self,idxs):
        idxs = sorted(idxs,key=lambda x: x[0])
        for idx,offset in idxs:
            self.add(idx,offset)



# ОТРИЦАТЕЛЬНЫЕ ОФФСЕТЫ
# как найти оффсет
if __name__ == '__main__':
    a = IdxList()
    a.add_batch([(5,-2),(430,-3),(800,3)])
    a.add_batch([(347,-4),(200,5)])
    print(a)
    print(a.find_old_idx(900))
