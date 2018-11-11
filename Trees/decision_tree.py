class TreeNode:

    def __init__(self, value_nd=''):
        self.node_v = value_nd
        self.left_node = ''
        self.right_node = ''


class DecisionTree:

    def __init__(self, in_lst=[]):
        # self.tree_m = TreeNode(value_nd='')  # init tree
        pass

    def add_node(self, tree_m='', value=''):

        if tree_m == '':
            tree_m = TreeNode(value)
            # tree_m.node = value

            return tree_m

        if tree_m.node_v < value:
            tree_m.right_node = self.add_node(tree_m.right_node, value)
        else:
            tree_m.left_node = self.add_node(tree_m.left_node, value)

        return tree_m
        #
        #
        # while self.tree_m.node!='':
        #     if self.tree_m.node < value:
        #         self.tree_m = self.tree_m.right_node
        #     else:
        #         self.tree_m = self.tree_m.left_node
        #
        # return self.tree_m

    def print_tree(self, tree_m):

        # depth-first search (DFS)
        # if tree_m == '':
        #     return
        #
        # if tree_m.node != '':
        #     print(tree_m.node)
        #     self.print_tree(tree_m.left_node)
        #     self.print_tree(tree_m.right_node)
        # else:
        #     return

        # breadth-first search (BFS)
        queue = [tree_m]
        while queue != []:
            node = queue.pop(0)
            if node.node_v != '':
                print(node.node_v)
                if node.left_node:
                    queue.append(node.left_node)
                if node.right_node:
                    queue.append(node.right_node)

    def build_tree(self, in_lst=[]):
        self.tree_m = ''
        for i, value in enumerate(in_lst):
            self.tree_m = self.add_node(self.tree_m, value)

        self.print_tree(self.tree_m)


if __name__ == '__main__':
    in_lst = [1, 3, 4, 1, 2, 0]
    decision_tree_m = DecisionTree()
    decision_tree_m.build_tree(in_lst=in_lst)
