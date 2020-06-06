from matplotlib import pyplot as plt
import numpy as np
import networkx as nx


class DAGMeta:
    def __init__(self, layer_sizes, bbox=(.1, .1, .9, .9)):
        '''
        参数
        ==========
        bbox: DAG 所在矩形区域
        layer_sizes: DAG 每层的节点数
        '''
        self.bbox = bbox
        self.layer_sizes = layer_sizes

    @property
    def w(self):
        '''DAG 的画布宽度'''
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self):
        '''DAG 的画布高度'''
        return self.bbox[3] - self.bbox[1]

    @property
    def x_center(self):
        '''DAG 的画布水平中心'''
        return (self.bbox[2] + self.bbox[0])/2

    @property
    def y_center(self):
        '''DAG 的画布竖直中心'''
        return (self.bbox[3] + self.bbox[1])/2

    def __len__(self):
        '''DAG 的层数'''
        return len(self.layer_sizes)

    @property
    def x_spacing(self):
        '''DAG 水平方向的留白间隙'''
        return self.w/(len(self) - 1)

    @property
    def y_spacing(self):
        '''DAG 竖直方向的留白间隙'''
        return self.h/max(self.layer_sizes)


class DAG(DAGMeta):
    def __init__(self, layer_sizes, bbox=(.1, .1, .9, .9), name='DAG'):
        super().__init__(layer_sizes, bbox)
        self._dag = nx.DiGraph(name=name)  # 可通过 self.name 获取名称

    def node_position(self, m, n):
        '''节点的位置
        参数
        ============
        m: DAG 的层序号
        n: DAG 该层的节点序号
        '''
        x = self.bbox[0] + m * self.x_spacing
        layer_top = self.y_spacing*(self.layer_sizes[m]-1)/2. + self.y_center
        y = layer_top - n*self.y_spacing
        return x, y

    def layout_nodes(self):
        for m, layer in enumerate(self.layer_sizes):
            for n in range(layer):
                self._dag.add_node(
                    f"$x^{m}_{n}$", pos=self.node_position(m, n))

    @property
    def pairs(self):
        sizes = self.layer_sizes.copy()
        edgelist = []
        n_layer = 0
        for size in sizes[1:]:
            x, y = np.meshgrid(np.arange(sizes[0]), np.arange(sizes[1]))
            paris = np.stack([x.flatten(), y.flatten()], axis=1)
            edgelist.extend(
                [f"$x^{n_layer}_{i}$", f"$x^{n_layer+1}_{j}$"] for i, j in paris)
            del sizes[0]
            n_layer += 1
        return edgelist

    def plot(self):
        self.layout_nodes()
        pos = nx.get_node_attributes(self._dag, 'pos')
        nodes = nx.draw_networkx_nodes(
            self._dag, pos, node_size=500, alpha=0.5)
        nx.draw_networkx_edges(self._dag, pos,
                               edgelist=self.pairs,
                               width=1, alpha=0.2, edge_color='g')
        nx.draw_networkx_labels(self._dag, pos, font_size=14)


if __name__ == "__main__":
    bbox = .1, .1, .9, .9  # 网络所在矩形区域
    layer_sizes = [5, 7, 5, 3, 1]  # 网络每层的节点数
    self = DAG(layer_sizes, bbox)
    self.plot()
    plt.axis('off')
    plt.show()
