#!/usr/bin/env python
# coding: utf-8

import argparse
# import scipy.stats as sps
# import numba as numba
# from numba import njit
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import datetime

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--particles', required=True)
parser.add_argument('--engine', required=True)
parser.add_argument('--output_folder', default='output')
parser.add_argument('--video_name', default='movie')
parser.add_argument('--log', default=True, type=bool)  # как часто выводить сообщения на экран
parser.add_argument('--speed', default=300)
parser.add_argument('--duration', default=60, type=int)
parser.add_argument('--bitrate', default=20000, type=int)

args = parser.parse_args()


# finaly work version: https://pastebin.com/pj8yNtSSs

def maxwell_pdf(x, temperature, mass_of_molecule, k=0.0000000000000000000000138065):
    return 4 * np.pi * (x ** 2) * ((mass_of_molecule / (2 * np.pi * k * temperature)) ** 1.5) * np.exp(
        -mass_of_molecule * (x ** 2) / (2 * k * temperature))


def last_second_mean(data, frames=100):
    n = len(data)
    if n % frames == 0 and n >= frames:
        return np.nanmean(data[n - frames: n])
    else:
        return None


def beauty_plot():
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 12
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 24

class SimulatorEngine:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.process = None
        self.sim_res = self.run_simulation()

        self.atom_cnt = int(next(self.sim_res))
        self.mass_of_molecule = next(self.sim_res)
        self.k = next(self.sim_res)

        self.X = np.zeros(self.atom_cnt)
        self.Y = np.zeros(self.atom_cnt)
        self.Z = np.zeros(self.atom_cnt)
        self.V = np.zeros(self.atom_cnt)

        # next main block
        nfr = args.duration * 50  # Number of frames
        fps = 100  # Frame per sec

        # next main block

        self.ps = []
        self.bs = []
        self.ts = []
        self.ss = np.arange(1, nfr, 0.5)
        i = 0
        fig = plt.figure(figsize=(23, 23), dpi=100)

        self.ax = fig.add_subplot(221, projection='3d')
        self.sct, = self.ax.plot([], [], [], "o", markersize=2)


        self.ax2 = fig.add_subplot(222)
        hst = self.ax2.hist([], density=True)
        # maxwell_plot, = ax2.plot([], [])

        self.ax3 = fig.add_subplot(223)
        self.pres_plt, = self.ax3.plot([], [])

        self.ax4 = fig.add_subplot(224)
        self.tmp_plt, = self.ax4.plot([], [])

        self.progress = tqdm(total=nfr * 2 - 3)

        last_frm = [-1, ]

        # next main block
        self.pressure_by_seconds = []
        self.t_by_sec = []

        # next main block
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
        self.ax3.set_xlabel('Time, s')
        self.ax4.set_xlabel('Time, s')
        self.ax3.set_ylabel('Pressure, Pa')
        self.ax4.set_ylabel('Temperature, K')
        beauty_plot()
        ani = animation.FuncAnimation(fig, self.update, nfr * 2 - 3, fargs=(last_frm,), interval=1000 / fps)

        output_folder = args.output_folder  # folder name

        writer = FFMpegWriter(fps=fps, metadata=dict(artist='nature'), bitrate=args.bitrate)
        # ani.save(args.output, writer=writer)
        try:
            os.mkdir(output_folder)
        except OSError:
            print("Creation of the directory %s failed" % output_folder)

        # fig.rcParam.update(beauty_plot())
        beauty_plot()

        ani.save("{}/{}.mp4".format(output_folder, args.video_name), writer=writer)

        self.progress.close()

        with open('{}/v_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/v_dist.csv".format(output_folder), self.V, delimiter=",")
        with open('{}/x_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/x_dist.csv".format(output_folder), self.X, delimiter=",")
        with open('{}/y_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/y_dist.csv".format(output_folder), self.Y, delimiter=",")
        with open('{}/z_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/z_dist.csv".format(output_folder), self.Z, delimiter=",")
        with open('{}/p_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/p_dist.csv".format(output_folder), self.ps, delimiter=",")
        with open('{}/t_dist.csv'.format(output_folder), 'w', newline='') as csvfile:
            np.savetxt("{}/t_dist.csv".format(output_folder), self.ts, delimiter=",")

        if args.log:
            with open("{}/log.txt".format(output_folder), "a") as myfile:
                myfile.write('start_time: {} \nexecution time: {}\n'.format(self.start_time,
                                                                            datetime.datetime.now() - self.start_time))  # записываем время в секундах

    def run_simulation(self):
        self.process = subprocess.Popen([args.engine,
                                         args.particles,
                                         str(int(float(args.speed)))],
                                        stdout=subprocess.PIPE)
        for c in iter(lambda: self.process.stdout.readline(), b''):  # replace '' with b'' for Python 3
            yield float(c.decode('ascii'))

    def step(self):
        for i in range(self.atom_cnt):
            x = next(self.sim_res)
            y = next(self.sim_res)
            z = next(self.sim_res)
            v_x = next(self.sim_res)
            v_y = next(self.sim_res)
            v_z = next(self.sim_res)
            a_x = next(self.sim_res)
            a_y = next(self.sim_res)
            a_z = next(self.sim_res)

            self.X[i] = x
            self.Y[i] = y
            self.Z[i] = z
            self.V[i] = np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)
        bumps = next(self.sim_res)
        pressure = next(self.sim_res)
        return bumps, pressure

    def update(self, ifrm, last_frm):
        if ifrm > last_frm[0]:
            bumps, pressure = self.step()
            self.bs.append(bumps)
            self.ps.append(pressure)
            temp = (np.nanmean(self.V ** 2) * self.mass_of_molecule / 3.0) / self.k
            self.ts.append(temp)
            last_frm[0] += 1
        self.progress.n = ifrm
        self.progress.refresh()
        self.sct.set_data(self.X, self.Y)
        self.sct.set_3d_properties(self.Z)
        self.ax.set_title('{} collisions'.format(self.bs[ifrm]), fontdict={'fontsize': 18})

        self.ax2.clear()
        x = np.linspace(0, 2 * float(args.speed), 200)
        self.ax2.hist(self.V, density=True, bins=np.linspace(0, 2 * float(args.speed), num=30), alpha=0.75,
                      edgecolor='black')

        self.ax2.plot(x, maxwell_pdf(x, self.ts[ifrm], self.mass_of_molecule), linestyle='--', linewidth=3,
                      label='Theoretical distribution')
        beauty_plot()
        self.ax2.set_xticks([min(x), max(x)])
        self.ax2.set_title('Mean Square Speed: {} m/s'.format(str(round(np.sqrt(np.nanmean(self.V ** 2)), 2))))
        self.ax2.legend()

        new_pres = last_second_mean(self.ps)
        if new_pres is not None:
            self.pressure_by_seconds.append(new_pres)

        self.pres_plt.set_data(np.arange(1, len(self.pressure_by_seconds) + 1), self.pressure_by_seconds)
        if len(self.pressure_by_seconds) > 0:
            pbs_mean = np.mean(self.pressure_by_seconds)
            self.ax3.set_ylim(pbs_mean / 2, pbs_mean + pbs_mean / 2)
            self.ax3.set_xlim(1, len(self.pressure_by_seconds))
            self.ax3.set_title('Pressure: {} Pa'.format(str(self.pressure_by_seconds[-1])))

        timeline = np.linspace(0, float(ifrm + 1) / 100, ifrm + 1)

        new_t = last_second_mean(self.ts)
        if new_t is not None:
            self.t_by_sec.append(new_t)
        self.tmp_plt.set_data(np.arange(1, len(self.t_by_sec) + 1), self.t_by_sec)
        if len(self.t_by_sec) > 0:
            self.ax4.set_ylim(self.t_by_sec[-1] / 2, self.t_by_sec[-1] + self.t_by_sec[-1] / 2)
            self.ax4.set_xlim(1, len(self.t_by_sec))
            self.ax4.set_title('Temperature: {} K'.format(str(self.t_by_sec[-1])))


s = SimulatorEngine()
