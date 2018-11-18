# -*- coding: utf-8 -*-
r"""
    display data
"""

import matplotlib.animation as manimation
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from utilities.load_data import load_data


def plot_hierarchy(X, y):
    r"""
        Matplotlib Object Hierarchy
            Figure->(multiple)Axes-> indiviual plot -> trick marks, line, legends, ..

    :return:
    """

    # fig = plt.figure()  # less frquenetly used, it just creates a Figure, with no Axes
    # fig.add_subplot()   # add Axes to fig.
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()  # flatten a 2d Numpy array to 1d
    """
        when we call plot(), matplotlib calls gca() to get the current axes and gca in turn calls gcf() to get 
        the current figure. If there is none it calls figures() to make one, strictly speaking, to make a subplot(1,1,1).
    """
    ax1.plot(X, y)
    ax2.scatter(X, y, c=y, cmap='RdYlGn')
    ax2.set_title('demo', fontsize=10)

    plt.show()

def plot_hierarchy_2(X,y,title='demo'):

    # not recommend
    # fig = plt.figure()
    # ax=fig.add_subplot(111)

    # recommend to use, plt.subplots() default parameter is (111)
    fig, ax = plt.subplots()  # combine figure and fig.add_subplots(111)
    ax.plot(X,y)
    ax.set_title(title)
    # ax.set_ylabel()
    plt.show()


def test_plot_hierarchy():
    plot_hierarchy()

def live_plot(X, y):
    global idx
    step = 10
    idx = 0
    fig, axes = plt.subplots(nrows=5, ncols=2)  # create fig and add subplots (axes) into it.
    ax_lst = []
    # X_tmp = []
    # y_tmp = []
    # for ax_i in axes.flatten():
    #     # ax_i.clear()
    #     scat_i=ax_i.scatter(X_tmp, y_tmp, c=y_tmp, s=10)
    #     # line,=ax_i.plot(X, y, animated=True)
    #     # ax_i.set_xlabel('weight')
    #     # ax_i.set_ylabel('Frequency')
    #     # ax_i.set_xlim(-100, 10000)
    #     # ax_i.set_ylim(-100, 10000)
    #     ax_lst.append(scat_i)

    def update(frame_data):
        X_tmp, y_tmp, idx = frame_data
        print(X_tmp, y_tmp)
        for ax_i in axes.flatten():
            ax_i.clear()  # clear the previous data, then redraw the new data.
            ax_i.scatter(X_tmp, y_tmp, c=y_tmp, s=10)
            # ax_i.plot(X, y, animated=True)
            ax_i.set_xlabel('weight')
            ax_i.set_ylabel('Frequency')
            ax_i.set_xlim(-100, 10000)
            ax_i.set_ylim(-100, 10000)

        print('start_idx=%d, %dth frame.' % (idx, idx // step))
        fig.suptitle('start_idx=%d, %dth frame.' % (idx, idx // step))

        return ax_lst

    frames_num = len(y) // step
    print('Number of frames:', frames_num)

    def new_data():
        global idx
        for i in range(len(y) // step):
            print('len(y)', len(y), idx)
            yield X[idx:idx + step], y[idx:idx + step], idx
            idx += step

    anim = FuncAnimation(fig, update, frames=new_data, repeat=False, interval=2000, blit=False)  # interval : ms
    anim.save('dynamic.mp4', writer='ffmpeg', fps=None, dpi=400)
    idx = 0
    anim.save('dynamic.gif', writer='imagemagick')
    plt.show()


def live_plot_2(X,y):
    r"""
        refer to:
            https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

    :param X:
    :param y:
    :return:
    """
    step = 10
    numframes = len(y)//step
    numpoints = 10
    color_data = np.random.random((numframes, numpoints))
    # x, y, c = np.random.random((3, numpoints))
    fig, axes = plt.subplots(nrows=5, ncols=2)  # create fig and add subplots (axes) into it.
    # scat = plt.scatter(x, y, c=c, s=100)

    def update_plot(i, data, x,y, step, scat):
        # scat.set_array(data[i])
        # scat.set_array(x)
        # scat.set_array(y)
        for ax_i in axes.flatten():
            ax_i.clear()
            scat_i = ax_i.scatter(x[i*step:(i+1)*step], y[i*step:(i+1)*step], c=data[i], s=100)
            ax_i.set_xlabel('weight')
            ax_i.set_ylabel('Frequency')
            ax_i.set_xlim(-100, 10000)
            ax_i.set_ylim(-100, 10000)
        print('start_idx=%d, %dth frame.' % (i*step, i))
        fig.suptitle('start_idx=%d, %dth frame.' % (i*step, i))
        return scat_i,


    anim =FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, X, y, step, ''),repeat=False,blit=True)
    anim.save('dynamic.mp4', writer='ffmpeg', fps=None, dpi=400)
    plt.show()

def live_plot_3():
    """
        refer to :
            https://stackoverflow.com/questions/9867889/matplotlib-scatterplot-removal
    :return:
    """
    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    def animate(i):
        # x.append(np.random.rand(1) * 10)
        x=[np.random.rand(1) * 10 for i in range(10)]
        y=[np.random.rand(1) * 10 for i in range(10)]
        # sc.clear()
        # print(i)
        ax.clear()
        ax.scatter(x, y)
        # if i > 0:
        #     sc.remove()
        # sc.set_offsets(np.c_[x, y])

    anim = FuncAnimation(fig, animate,  frames=200, interval=100, repeat=False)
    anim.save('dynamic.mp4', writer='ffmpeg', fps=None, dpi=400)

    plt.show()
    plt.show()


def dynamic_plot(X, y, title='dynamic_plot'):
    r"""
        must install ffmpeg (sudo apt-get install ffmpeg), then pip3 install ffmpeg

        Note:
            pycharm cannot show animation. so it needs to save animation to local file.

    :param input_f:
    :return:
    """
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    plt.title(title)
    #
    # def update_figure(X, y):
    #     # plt.scatter(X, y)
    #     plt.plot(X,y,'k-o')
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)

    with writer.saving(fig, "writer_test.mp4", dpi=100):
        for k in range(10):
            # Create a new plot object
            ax = plt.scatter(X[:k * 2], y[:k * 2])
            # update_figure(X,y)
            # ax = plt.plot(X,y)
            writer.grab_frame()


def drip_drop_demo():
    """
        simulate drip drop
    :return:
    """
    global position, color, size

    # New figure with white background
    fig = plt.figure(figsize=(6, 6), facecolor='white')

    # New axis over the whole figure, no frame and a 1:1 aspect ratio
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)

    num = 50
    size_max = 20 * 20
    size_min = 20

    # Ring position
    position = np.random.uniform(0, 1, (num, 2))

    # Ring colors
    color = np.ones((num, 4)) * (0, 0, 0, 1)
    # Alpha color channel goes from 0 (transparent) to 1 (opaque)
    color[:, 3] = np.linspace(0, 1, num)

    # Ring sizes
    size = np.linspace(size_min, size_max, num)

    # scatter plot
    scat = ax.scatter(position[:, 0], position[:, 1], s=size, lw=0.5, edgecolors=color, facecolors='None')

    # Ensure limit are [0,1] and remove ticks
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])

    def update(frame):
        global position, color, size
        # print(color, size)
        color[:, 3] = np.maximum(0, color[:, 3] - 1.0 / num)
        size += (size_max - size_min) / num

        # reset ring
        i = frame % num
        position[i] = np.random.uniform(0, 1, 2)
        size[i] = size_min
        color[i, 3] = 1

        # update scatter object
        scat.set_edgecolors(color)
        scat.set_sizes(size)
        scat.set_offsets(position)

        # returen the modified object
        # return scat # TypeError: 'PathCollection' object is not iterable
        return scat,

    anim = FuncAnimation(fig, update, interval=100, blit=True, frames=200)
    anim.save('rain.gif', writer='imagemagick', fps=30, dpi=40)
    anim.save('rain.mp4', writer='ffmpeg', fps=30, dpi=40)
    plt.show()


def histogram(x, num_bins=5, title='histogram', x_label='Values.', y_label='Frequency'):
    # x = [21, 22, 23, 4, 5, 6, 77, 8, 9, 10, 31, 32, 33, 34, 35, 36, 37, 18, 49, 50, 100]
    # num_bins = 5
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

# from matplotlib import style
#
# def dynamic_plot(input_f):
#     pass
#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import time
#
# style.use('fivethirtyeight')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
#
# def animate(i):
#     pullData = open(r'../data/attack_demo.csv', "r").read()
#     # pullData = open('../data/samples.csv', "r").read()
#     dataArray = pullData.split('\n')
#     xar = []
#     yar = []
#     for eachLine in dataArray:
#         if len(eachLine) > 1:
#             x, y = eachLine.split(',')[0:2]
#             xar.append(variables_n_data_types(x))
#             yar.append(variables_n_data_types(y))
#     ax1.clear()
#     ax1.plot(xar, yar)
#
# anim = animation.FuncAnimation(fig, animate, interval=50, blit=True)
# plt.show()
# anim.save('noise.gif', writer='ffmpeg', fps=10, dpi=100, metadata={'title':'test'})
#
#


# import matplotlib.pyplot as plt
# import time
# import random
#
# ysample = random.sample(range(-50, 50), 100)
#
# xdata = []
# ydata = []
#
# plt.show()
#
# axes = plt.gca()
# axes.set_xlim(0, 100)
# axes.set_ylim(-50, +50)
# line, = axes.plot(xdata, ydata, 'r-')
#
# for i in range(100):
#     xdata.append(i)
#     ydata.append(ysample[i])
#     line.set_xdata(xdata)
#     line.set_ydata(ydata)
#     plt.draw()
#     plt.pause(1e-17)
#     time.sleep(0.1)
#
# # add this if you don't want the window to disappear at the end
# plt.show()


#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, MovieWriter
#
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro', animated=True)
#
# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
# def update_figure(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
# ani = FuncAnimation(fig, update_figure, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=True)
# plt.show()
#
# n = 10
# moviewriter = MovieWriter()
# # moviewriter.setup(fig=fig, 'my_movie.ext', dpi=100)
# with moviewriter.saving(fig, 'myfile.mp4', dpi=100):
#     for j in range(n):
#         update_figure(n)
#         moviewriter.grab_frame()

# from python3.tricks import load_data



#
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
#
# def data_gen(t=0):
#     cnt = 0
#     while cnt < 1000:
#         cnt += 1
#         t += 0.1
#         yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)
#
#
# def init():
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(0, 10)
#     # del xdata[:]
#     # del ydata[:]
#     line.set_data(xdata, ydata)
#     return line,
#
# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)
# ax.grid()
# xdata, ydata = [], []
#
#
# def run(data):
#     # update the data
#     t, y = data
#     xdata.append(t)
#     print(len(xdata))
#     ydata.append(y)
#     xmin, xmax = ax.get_xlim()
#
#     if t >= xmax:
#         ax.set_xlim(xmin, xmax)
#         ax.figure.canvas.draw()
#         # xdata =xdata[-10:]
#         # ydata=ydata[-10:]
#     line.set_data(xdata, ydata)
#
#
#     return line,
#
# ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
#                               repeat=False, init_func=init)
# plt.show()

if __name__ == '__main__':
    input_fil = '../data/attack_demo.csv'
    data = load_data(input_fil)
    data = np.asarray(data, dtype=float)
    plot_hierarchy_2(X=data[:,0],y=data[:,1])
    # live_plot(X=data[:, 0], y=data[:, 1])
    # live_plot_2(X=data[:,0],y=data[:,1])
    live_plot_3()
    # # drip_drop()
    # # dynamic_plot(X=data[:,0],y=data[:,1])
