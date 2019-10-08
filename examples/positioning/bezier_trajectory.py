# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2019 Bitcraze AB
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA  02110-1301, USA.
"""
Example of how to generate trajectories for the High Level commander using
Bezier curves. The output from this script is intended to be pasted into the
autonomous_sequence_high_level.py example.

This code uses Bezier curves of degree 7, that is with 8 control points.
See https://en.wikipedia.org/wiki/B%C3%A9zier_curve

All coordinates are (x, y, z, yaw)
"""

# Enable this if you have Vispy installed and want a visualization of the
# trajectory

# Import here to avoid problems for users that do not have Vispy
# from vispy import scene
# from vispy.scene import XYZAxis, LinePlot, TurntableCamera, Markers      
# visualizer = Visualizer()

import math
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

draggable_points = []
circles = []

def findDraggablePoint(plot_name, point_id):
    for draggable_point in draggable_points:
        if draggable_point.plot_name is plot_name and draggable_point.point_id is point_id:
            return draggable_point
    return False

class DraggablePoint:
    lock = None #only one can be animated at a time
    def __init__(self, plot_name, point_id, point):
        self.plot_name = plot_name
        self.point_id = point_id
        self.point = point
        self.press = None
        self.background = None
        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.onPress)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.onRelease)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.onMotion)

    
    def onPress(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def onMotion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)
        print(self.point_id, self.point.center)

        other_plot_name = None
        if self.plot_name is 'XY':
            other_plot_name = 'XZ'

        elif self.plot_name is 'XZ':
            other_plot_name = 'XY'
        else:
            assert "Invalid plot name" + self.plot_name

        other_draggable_point = findDraggablePoint(other_plot_name, self.point_id)
        # print(other_draggable_point)
        if other_draggable_point is not False:
            # other_draggable_point.point.center = self.point.center
            other_draggable_point.point.center = (self.point.center[0], other_draggable_point.point.center[1])

        
        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def onRelease(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)




class Node:
    """
    A node represents the connection point between two Bezier curves
    (called Segments).
    It holds 4 control points for each curve and the positions of the control
    points are set to join the curves with continuity in c0, c1, c2, c3.
    See https://www.cl.cam.ac.uk/teaching/2000/AGraphHCI/SMEG/node3.html

    The control points are named
    p4, p5, p6 and p7 for the tail of the first curve
    q0, q1, q2, q3 for the head of the second curve
    """

    def __init__(self, node_id, q0, q1=None, q2=None, q3=None):
        """
        Create a Node. Pass in control points to define the shape of the
        two sequence that share the Node. The control points are for the
        second segment, that is the four first control points of the Bezier
        curve after the node. The control points for the Bezier curve before
        the node are calculated from the existing control points.
        The control points are for scale = 1, that is if the Bezier curve
        after the node has scale = 1 it will have exactly these handles. If the
        curve after the node has a different scale the handles will be moved
        accordingly when the Segment is created.

        q0 is required, the other points are optional.
        if q1 is missing it will be set to generate no velocity in q0.
        If q2 is missing it will be set to generate no acceleration in q0.
        If q3 is missing it will be set to generate no jerk in q0.

        If only q0 is set, the node will represent a point where the Crazyflie
        has no velocity. Good for starting and stopping.

        To get a fluid motion between sequence, q1 must be set.

        :param q0: The position of the node
        :param q1: The position of the first control point
        :param q2: The position of the second control point
        :param q3: The position of the third control point
        """
        self.node_id = node_id
        self._control_points = np.zeros([2, 4, 4])

        q0 = np.array(q0)

        if q1 is None:
            q1 = q0
        else:
            q1 = np.array(q1)
            # print('q1 generated:', q1)

        d = q1 - q0

        if q2 is None:
            q2 = q0 + 2 * d
            # print('q2 generated:', q2)
        else:
            q2 = np.array(q2)

        e = 3 * q2 - 2 * q0 - 6 * d

        if q3 is None:
            q3 = e + 3 * d
            # print('q3 generated:', q3)
        else:
            q3 = np.array(q3)

        p7 = q0
        p6 = q1 - 2 * d
        p5 = q2 - 4 * d
        p4 = 2 * e - q3

        self._control_points[0][0] = q0
        self._control_points[0][1] = q1
        self._control_points[0][2] = q2
        self._control_points[0][3] = q3

        self._control_points[1][3] = p7
        self._control_points[1][2] = p6
        self._control_points[1][1] = p5
        self._control_points[1][0] = p4

    def get_head_points(self):
        return self._control_points[0]

    def get_tail_points(self):
        return self._control_points[1]

    def draw_unscaled_controlpoints(self, visualizer, color=(0.8, 0.8, 0.8)):
        for p in self._control_points[0]:
            visualizer.marker(p[0:3], color=color)
        for p in self._control_points[1]:
            visualizer.marker(p[0:3], color=color)

    
            
    def draw_controlpoints_matplot(self, ax, ax1, ax2,  color='k'):
    
        # control_points = self._control_points[0] + self._control_points[1]
        for h,p in enumerate (self._control_points[0]): #headnode
            
            point_id =  'N' + str(self.node_id) + '_Q' + str(h)
            # print(point_id)
            # p = list(p)
            # print(p[0], p[1],p[2])
            ax.scatter(p[0], p[1],p[2], color = color)

            plot_name = 'XY' 
            other_draggable_point = findDraggablePoint(plot_name,point_id)
            # print(other_draggable_point)
            if other_draggable_point is False:
                # ax1.scatter(p[0], p[1], color = color)
                circ = patches.Circle((p[0], p[1]), 0.05, fc=color, alpha=0.25)
                circles.append(circ)
                ax1.add_patch(circ)
                draggable_point = DraggablePoint(plot_name ,point_id , circ )
                draggable_points.append(draggable_point)
                print(plot_name,point_id)
                
                
            plot_name = 'XZ' 
            other_draggable_point = findDraggablePoint(plot_name,point_id)
            # print(other_draggable_point)
            if other_draggable_point is False:
                # ax2.scatter(p[1],p[2], color = color)
                circ = patches.Circle((p[1], p[2]), 0.05, fc=color, alpha=0.25)
                circles.append(circ)
                ax2.add_patch(circ)
                draggable_point = DraggablePoint(plot_name , point_id , circ )
                draggable_points.append(draggable_point)
                print(plot_name,point_id)

            
        
        for t,p in enumerate (self._control_points[1]): #tailnode -> self-run
            point_id = 'N' + str(self.node_id) + '_P' + str(t)
            # print(point_id)
            # p = list(p)
            # print(p[0], p[1],p[2])
            ax.scatter(p[0], p[1],p[2], color = color)

            plot_name = 'XY' 
            other_draggable_point = findDraggablePoint(plot_name,point_id)
            # print(other_draggable_point)
            if other_draggable_point is False:
                # ax1.scatter(p[0], p[1], color = color)
                circ = patches.Circle((p[0], p[1]), 0.025, fc=color, alpha=0.25)
                circles.append(circ)
                ax1.add_patch(circ)
                draggable_point = DraggablePoint('XY' ,point_id , circ )
                draggable_points.append(draggable_point)

            plot_name = 'XZ' 
            other_draggable_point = findDraggablePoint(plot_name,point_id)
            # print(other_draggable_point)
            if other_draggable_point is False:
                # ax2.scatter(p[1],p[2], color = color)
                circ = patches.Circle((p[1], p[2]), 0.025, fc=color, alpha=0.25)
                circles.append(circ)
                ax2.add_patch(circ)
                draggable_point = DraggablePoint('XZ' , point_id , circ )
                draggable_points.append(draggable_point)            
        

    def print(self):
        print('Node ---')
        print('Tail:')
        for c in self._control_points[1]:
            print(c)
        print('Head:')
        for c in self._control_points[0]:
            print(c)


class Segment:
    """
    A Segment represents a Bezier curve of degree 7. It uses two Nodes to
    define the shape. The scaling of the segment will move the handles compared
    to the Node to maintain continuous position, velocity, acceleration and
    jerk through the Node.
    A Segment can generate a polynomial that is compatible with the High Level
    Commander, either in python to be sent to the Crazyflie, or as C code to be
    used in firmware.
    A Segment can also be rendered in Vispy.
    """

    def __init__(self, head_node, tail_node, scale):
        self._head_node = head_node
        self._tail_node = tail_node
        self._scale = scale

        unscaled_points = np.concatenate(
            [self._head_node.get_head_points(), self._tail_node.get_tail_points()])

        self._points = self._scale_control_points(unscaled_points, self._scale)

        polys = self._convert_to_polys()
        self._polys = self._stretch_polys(polys, self._scale)

        self._vel = self._deriv(self._polys)
        self._acc = self._deriv(self._vel)
        self._jerk = self._deriv(self._acc)

    def _convert_to_polys(self):
        n = len(self._points) - 1
        result = np.zeros([4, 8])

        for d in range(4):
            for j in range(n + 1):
                s = 0.0
                for i in range(j + 1):
                    s += ((-1) ** (i + j)) * self._points[i][d] / (
                        math.factorial(i) * math.factorial(j - i))

                c = s * math.factorial(n) / math.factorial(n - j)
                result[d][j] = c

        return result

    def draw_trajectory(self, visualizer, color='black'):
        self._draw(self._polys, color, visualizer)

    def draw_trajectory_matplot(self, ax, color='black'):
        self._draw_matplot(self._polys, color, ax)

    def draw_vel(self, visualizer):
        self._draw(self._vel, 'green', visualizer)

    def draw_acc(self, visualizer):
        self._draw(self._acc, 'red', visualizer)

    def draw_jerk(self, visualizer):
        self._draw(self._jerk, 'blue', visualizer)

    def draw_control_points(self, visualizer):
        for p in self._points:
            visualizer.marker(p[0:3])

    def _draw(self, polys, color, visualizer):
        step = self._scale / 32
        prev = None
        for t in np.arange(0.0, self._scale + step, step):
            p = self._eval_xyz(polys, t)

            if prev is not None:
                visualizer.line(p, prev, color=color)
                
            prev = p

    def _draw_matplot(self, polys, color, ax):
        step = self._scale / 32
        prev = None
        for t in np.arange(0.0, self._scale + step, step):
            # print(polys)
            p = self._eval_xyz(polys, t)
            # print(p)    

            if prev is not None:
                ax.plot([prev[0], p[0]], [prev[1],p[1]],zs=[prev[2],p[2]],color = color)
                
            prev = p

    def velocity(self, t):
        return self._eval_xyz(self._vel, t)

    def acceleration(self, t):
        return self._eval_xyz(self._acc, t)

    def jerk(self, t):
        return self._eval_xyz(self._jerk, t)

    def _deriv(self, p):
        result = []
        for i in range(3):
            result.append([
                1 * p[i][1],
                2 * p[i][2],
                3 * p[i][3],
                4 * p[i][4],
                5 * p[i][5],
                6 * p[i][6],
                7 * p[i][7],
                0
            ])

        return result

    def _eval(self, p, t):
        result = 0
        for part in range(8):
            result += p[part] * (t ** part)
        return result

    def _eval_xyz(self, p, t):
        return np.array(
            [self._eval(p[0], t), self._eval(p[1], t), self._eval(p[2], t)])

    def print_poly_python(self):
        s = '  [' + str(self._scale) + ', '

        for axis in range(4):
            for d in range(8):
                s += str(self._polys[axis][d]) + ', '

        s += '],  # noqa'
        print(s)

    def print_poly_c(self):
        s = ''

        for axis in range(4):
            for d in range(8):
                s += str(self._polys[axis][d]) + ', '

        s += str(self._scale)
        print(s)

    def print_points(self):
        print(self._points)

    def print_peak_vals(self):
        peak_v = 0.0
        peak_a = 0.0
        peak_j = 0.0

        step = 0.05
        for t in np.arange(0.0, self._scale + step, step):
            peak_v = max(peak_v, np.linalg.norm(self._eval_xyz(self._vel, t)))
            peak_a = max(peak_a, np.linalg.norm(self._eval_xyz(self._acc, t)))
            peak_j = max(peak_j, np.linalg.norm(self._eval_xyz(self._jerk, t)))

        print('Peak v:', peak_v, 'a:', peak_a, 'j:', peak_j)

    def _stretch_polys(self, polys, time):
        result = np.zeros([4, 8])

        recip = 1.0 / time

        for p in range(4):
            scale = 1.0
            for t in range(8):
                result[p][t] = polys[p][t] * scale
                scale *= recip

        return result

    def _scale_control_points(self, unscaled_points, scale):
        s = scale
        l_s = 1 - s
        p = unscaled_points

        result = [None] * 8

        result[0] = p[0]
        result[1] = l_s * p[0] + s * p[1]
        result[2] = l_s ** 2 * p[0] + 2 * l_s * s * p[1] + s ** 2 * p[2]
        result[3] = l_s ** 3 * p[0] + 3 * l_s ** 2 * s * p[
            1] + 3 * l_s * s ** 2 * p[2] + s ** 3 * p[3]
        result[4] = l_s ** 3 * p[7] + 3 * l_s ** 2 * s * p[
            6] + 3 * l_s * s ** 2 * p[5] + s ** 3 * p[4]
        result[5] = l_s ** 2 * p[7] + 2 * l_s * s * p[6] + s ** 2 * p[5]
        result[6] = l_s * p[7] + s * p[6]
        result[7] = p[7]

        return result


class Visualizer:
    def __init__(self):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600),
                                        show=True)
        view = self.canvas.central_widget.add_view()
        view.bgcolor = '#ffffff'
        view.camera = TurntableCamera(fov=10.0, distance=40.0, up='+z',
                                      center=(0.0, 0.0, 1.0))
        XYZAxis(parent=view.scene)
        self.scene = view.scene

    def marker(self, pos, color='black', size=8):
        Markers(pos=np.array(pos, ndmin=2), face_color=color,
                parent=self.scene, size=size)

    def lines(self, points, color='black'):
        LinePlot(points, color=color, parent=self.scene)

    def line(self, a, b, color='black'):
        self.lines([a, b], color)

    def run(self):
        self.canvas.app.run()


segment_time = 2
z = 1
yaw = 0
sequence = []


####

# When setting q2 we can also control acceleration and get more action.
# Yaw also adds to the fun.

d2 = 0.2
dyaw = 2
f = -0.3
color = 'b'

n8 = Node(8, (0, 0, z, yaw))
n9 = Node(9,
    (1, 0, z, yaw),
    q1=(1 + d2, 0 + d2, z, yaw),
    q2=(1 + 2 * d2, 0 + 2 * d2 + 0*f * d2, 1, yaw))
n10 = Node(10,
    (1, 1, z, yaw + dyaw),
    q1=(1 - d2, 1 + d2, z, yaw + dyaw),
    q2=(1 - 2 * d2 + f * d2, 1 + 2 * d2 + f * d2, 1, yaw + dyaw))
n11 = Node(11,
    (0, 1, z, yaw - dyaw),
    q1=(0 - d2, 1 - d2, z, yaw - dyaw),
    q2=(0 - 2 * d2,  1 - 2 * d2,  1, yaw - dyaw))

sequence.append({'s': Segment(n8, n9, segment_time), 'c': color})
sequence.append({'s': Segment(n9, n10, segment_time), 'c': color})
sequence.append({'s': Segment(n10, n11, segment_time), 'c': color})
sequence.append({'s': Segment(n11, n8, segment_time), 'c': color})

####


# By setting the q1 control point we get velocity through the nodes
# Increase d to 0.7 to get some more action
d = 0.1
color = 'g'

n4 = Node(4, (0, 0, z, yaw), q1=(0, 0, z, yaw), q2=(0, 0, z, yaw), q3=(0, 0, z, yaw))
n5 = Node(5, (1, 0, z, yaw), q1=(1 + d, 0 + d, z, yaw))
n6 = Node(6, (1, 1, z, yaw), q1=(1 - d, 1 + d, z, yaw))
n7 = Node(7, (0, 1, z, yaw), q1=(0 - d, 1 - d, z, yaw))

sequence.append({'s': Segment(n4, n5, segment_time), 'c': color})
sequence.append({'s': Segment(n5, n6, segment_time), 'c': color})
sequence.append({'s': Segment(n6, n7, segment_time), 'c': color})
sequence.append({'s': Segment(n7, n4, segment_time), 'c': color})



####

color = 'r'


# Nodes with one control point has not velocity, this is similar to calling
# goto in the High-level commander

n0 = Node(0, (0, 0, z, yaw))
n1 = Node(1, (1, 0, z, yaw))
n2 = Node(2, (1, 1, z, yaw))
n3 = Node(3, (0, 1, z, yaw))

# n12 = n0 k
n12 = Node(12, (0.5, 0.5, z, yaw))

sequence.append({'s': Segment(n0, n1, segment_time), 'c': color})
sequence.append({'s': Segment(n1, n2, segment_time), 'c': color})
sequence.append({'s': Segment(n2, n3, segment_time), 'c': color})
sequence.append({'s': Segment(n3, n12, segment_time), 'c': color})

print('Paste this code into the autonomous_sequence_high_level.py example to '
      'see it fly')

for segment in sequence:
    segment['s'].print_poly_python()



    
fig = plt.figure(figsize=(25,20))
ax = fig.add_subplot(212,projection='3d')


plot_name = 'XY'
ax1 = fig.add_subplot(221,sharex=ax,sharey=ax)
ax1.grid(True)
        

plot_name = 'XZ'
ax2 = fig.add_subplot(222)
plt.xlim(-1,2)
plt.ylim(-1,2)
ax2.grid(True)
        

for sequence_step in sequence:
        
    # sequence_step['s'].draw_trajectory(visualizer, sequence_step['c'])
    # segment['s'].draw_vel(visualizer)
    # segment['s'].draw_control_points(visualizer)
    # sequence_step['s']._head_node.draw_unscaled_controlpoints(visualizer, sequence_step['c'])
    # sequence_step['s']._tail_node.draw_unscaled_controlpoints(visualizer, sequence_step['c'])
            
    segment = sequence_step['s']
    color = sequence_step['c']

    segment.draw_trajectory_matplot(ax, color)
 
    segment._head_node.draw_controlpoints_matplot(ax,ax1,ax2, color)
    # segment._tail_node.draw_controlpoints_matplot(ax,ax1,ax2, color)
        
    # alternative ways:
    # sequence_step['s']._head_node.draw_controlpoints_matplot(ax, sequence_step['c'])
    # sequence_step['s']._tail_node.draw_controlpoints_matplot(ax, sequence_step['c'])
    # sequence_step['s']._head_node.draw_controlpoints_matplot(ax1, sequence_step['c'])
    # sequence_step['s']._tail_node.draw_controlpoints_matplot(ax1, sequence_step['c'])
    # sequence_step['s']._head_node.draw_controlpoints_matplot(ax2, sequence_step['c'])
    # sequence_step['s']._tail_node.draw_controlpoints_matplot(ax2, sequence_step['c'])
    # for n in [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]:
    #     n.draw_unscaled_controlpoints(visualizer)

if sequence[-1]:
    segment = sequence_step['s']
    segment._tail_node.draw_controlpoints_matplot(ax,ax1,ax2, color)

plt.show()
# visualizer.run()