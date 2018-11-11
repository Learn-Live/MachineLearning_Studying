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
#             xar.append(int(x))
#             yar.append(int(y))
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

import matplotlib.animation as manimation
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dynamic_plot(X, y):
    r"""
        must install ffmpeg, then pip3 install ffmpeg

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
    #
    # def update_figure(X, y):
    #     # plt.scatter(X, y)
    #     plt.plot(X,y,'k-o')
    #     plt.xlim(0,100)
    #     plt.ylim(0,100)

    with writer.saving(fig, "writer_test.mp4", dpi=100):
        for k in range(10):
            # Create a new plot object
            plt.scatter(range(X), range(y))
            # update_figure(X,y)
            writer.grab_frame()

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

# if __name__ == '__main__':
#     input_f = '../data/attack_demo.csv'
#     X,y = load_data(input_f)
#     dynamic_plot(X,y)
