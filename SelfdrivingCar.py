import pyglet
from pyglet.gl import *
import numpy as np
import math
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Track:

    def __init__(self):
        self.batch = pyglet.graphics.Batch()
        self.backgroundSprite = pyglet.sprite.Sprite(pyglet.image.load('background.png'), batch=self.batch, group=pyglet.graphics.OrderedGroup(0))
        self.lines = []
        self.rewardLines = []

        vertsOut = [53,285, 58,454, 99,522, 250,569.75, 660,591, 973.94,590, 1136,577, 1173.5,547, 1181,483, 1166,442,
                    1053.5,391, 806,327.12, 607,318.25, 550.5,298, 543,265, 558,227, 650,218, 942,251.5, 1114,251,
                    1172,202, 1204,109, 1172,38, 1009,15.5, 639.5,9, 148,31, 111,67, 84,142, 53,285]
        vertsIn = [150,284.5, 150,378, 163,442, 263,483, 663,505.5, 969,500.62, 1061,505.5, 1079,499, 1070,488,
                   796,412, 596.5,401, 481,355, 455,281.5, 471,196, 491,160, 634,136, 948,178, 1065,177, 1099,165,
                   1119,128, 1109,104.5, 998,92, 639.5,87, 208,116.25, 191,136, 150,284.5]

        for i in range(0, len(vertsOut) - 2, 2):
            self.lines.append(pyglet.graphics.vertex_list(2,
                                                          ('v2f', vertsOut[i:i+4]),
                                                          ('c3B', (255, 0, 0) * 2)))

        for i in range(0, len(vertsIn) - 2, 2):
            self.lines.append(pyglet.graphics.vertex_list(2,
                                                          ('v2f', vertsIn[i:i+4]),
                                                          ('c3B', (255, 0, 0) * 2)))


        vertsRewardLines = [53,285,150,284.5, 56.08,389,150,378, 99,522,163,442, 250,569.75,263,483, 455,580.37,460.47,494.11, 660,591,663,505.5,
                            827,590.47,823.67,503.06, 973.94,590,969,500.62, 1136,577,1061,505.5, 1166,442,1070,488, 969,369.56,942,452.5,
                            806,327.12,796,412, 607,318.25,596.5,401, 550.5,298,481,355, 558,227,471,196, 650,218,634,136,
                            786.64,233.68,796,157.67, 942,251.5,948,178, 1047.29,251.19,1043.75,177.18, 1172,202,1099,165, 1172,38,1109,104.5,
                            1009,15.5,998,92, 830,12.35,828.01,89.63, 639.5,9,639.5,87, 405.75,19.46,414.33,102.26, 148,31,208,116.25]
        for i in range(0, len(vertsRewardLines), 4):
            self.rewardLines.append(vertsRewardLines[i:i+4])

    def draw(self):
        self.batch.draw()


class Car:

    # constants
    CA_F = -1.5             # cornering stiffness front wheels
    CA_R = -1.52             # cornering stiffness rear wheels
    MAX_GRIP = 2.0          # maximum (normalised) friction force
    CDRAG = 5.0             # factor for air resistance
    CRESISTANCE = 30.0      # factor for rolling resistance

    def __init__(self, x, y):

        # parameter
        self.maxEngineforce = 10000
        self.maxBrakeforce = 12000
        self.mass = 1500
        self.distanceWheels = 2.0
        self.heightCenterOfGravity = 1.0
        self.maxSteeringAngle = 1.5
        self.steeringVelocity = 5.0


        self.weight = 9.81*self.mass
        self.weightFrontAxle = 0.5*self.weight
        self.weightRearAxle = 0.5*self.weight

        self.steeringAngle = 0.0

        self.scale = 0.08
        self.image = pyglet.image.load('car-red.png')
        self.batch = pyglet.graphics.Batch()
        self.groupLines = pyglet.graphics.OrderedGroup(0)
        self.groupCircles  = pyglet.graphics.OrderedGroup(1)
        self.groupCar = pyglet.graphics.OrderedGroup(2)


        self.position = np.array([x,y])
        self.velocity = np.array([0.0,0.0])
        self.acceleration = np.array([0.0,0.0])

        self.angular_velocity = 0.0
        self.engineforce = 0.0
        self.brakeforce = 0.0
        self.angle = -90

        self.sprite = pyglet.sprite.Sprite(self.image, x=self.position[0], y=self.position[1], batch=self.batch, group=self.groupCar)
        self.sprite.scale = self.scale
        self.sprite.image.anchor_x = self.sprite.image.width // 2
        self.sprite.image.anchor_y = self.sprite.image.height // 2
        self.sprite.rotation = self.angle

        self.numberOfLines = 8
        self.maxLineLength = 400
        self.circleRadius = 3
        self.circleNumPoints = 16
        self.lines = self.createLines(self.numberOfLines, self.maxLineLength)
        self.circles = self.createCircles(self.numberOfLines)
        self.intersects = []
        for i in range(self.numberOfLines):
            self.intersects.append(np.array([0.0,0.0]))
        self.distances = np.full((self.numberOfLines, 1), np.inf)

        self.rewardIndex = 0
        self.reward = 0
        self.rewardPerLine = 1000

        self.collisionLines = []
        for i in range(4):
            self.collisionLines.append(pyglet.graphics.vertex_list(2,
                ('v2f', (-10, -10, -10, -10)),
                ('c4B', (255,0,0,0) * 2)))

    def do_physics(self, dt):
        # velocity relative to car, x is to the front of the car [0], y is to the right [1]
        velocityRelCar = self.rotate2d(self.velocity, self.angle)

        # side slip angle a.k.a. beta (angle between car orientation and movement)
        if not velocityRelCar[0]:
            sideslipAngle = 0.0
        else:
            sideslipAngle = math.atan(velocityRelCar[1]/velocityRelCar[0])

        # slip angles a.k.a. alpha (angle between wheel orientation and movement)
        yawspeed = self.distanceWheels * 0.5 * self.angular_velocity
        if not velocityRelCar[0]:
            rot_angle = 0.0
        else:
            rot_angle = math.atan(yawspeed/velocityRelCar[0])
        slipAngleFront = sideslipAngle + rot_angle - math.radians(self.steeringAngle)
        slipAngleRear = sideslipAngle - rot_angle

        # lateral forces
        fLatFront = np.array([0.0,0.0])
        fLatFront[1] = Car.CA_F * slipAngleFront
        fLatFront[1] = max(- Car.MAX_GRIP, min(fLatFront[1], Car.MAX_GRIP))
        fLatFront[1] *= self.weight*0.5
        fLatFront[1] *= math.cos(math.radians(self.steeringAngle))
        fLatRear = np.array([0.0,0.0])
        fLatRear[1] = Car.CA_R * slipAngleRear
        fLatRear[1] = max(- Car.MAX_GRIP, min(fLatRear[1], Car.MAX_GRIP))
        fLatRear[1] *= self.weight*0.5

        # longitudinal force
        fTrac = np.array([self.engineforce - self.brakeforce * math.copysign(1.0, velocityRelCar[0]),0.0])

        # resistent force (drag and rolling)
        fRes = -(Car.CRESISTANCE * velocityRelCar + Car.CDRAG * velocityRelCar * np.abs(velocityRelCar))

        # sum forces
        force = fLatFront + fLatRear + fTrac + fRes

        # torque on body from lateral force
        torque = self.distanceWheels*0.5*fLatFront[1] - self.distanceWheels*0.5*fLatRear[1]

        # acceleration
        acceleration = force / self.mass
        angular_acceleration = torque / self.mass

        self.acceleration = self.rotate2d(acceleration, -self.angle)

        self.velocity += dt*self.acceleration
        if np.linalg.norm(self.velocity) < 5 and self.engineforce < 0.5:
            self.velocity[0], self.velocity[1] = 0.0,0.0
            angular_acceleration, self.angular_velocity = 0.0,0.0

        self.position += dt*self.velocity

        self.angular_velocity += dt*angular_acceleration
        self.angle += dt*math.degrees(self.angular_velocity)

    def updateCollisionLines(self, track):
        w = self.sprite.width//2
        h = self.sprite.height//2
        self.collisionLines[0].vertices = self.rotatePointAroundPoint([self.position[0]-w, self.position[1]-h], self.position, -self.angle) + self.rotatePointAroundPoint([self.position[0]+w, self.position[1]-h], self.position, -self.angle)
        self.collisionLines[1].vertices = self.rotatePointAroundPoint([self.position[0]+w, self.position[1]-h], self.position, -self.angle) + self.rotatePointAroundPoint([self.position[0]+w, self.position[1]+h], self.position, -self.angle)
        self.collisionLines[2].vertices = self.rotatePointAroundPoint([self.position[0]+w, self.position[1]+h], self.position, -self.angle) + self.rotatePointAroundPoint([self.position[0]-w, self.position[1]+h], self.position, -self.angle)
        self.collisionLines[3].vertices = self.rotatePointAroundPoint([self.position[0]-w, self.position[1]+h], self.position, -self.angle) + self.rotatePointAroundPoint([self.position[0]-w, self.position[1]-h], self.position, -self.angle)
        collision = False
        for line in self.collisionLines:
            for trackline in track.lines:
                temp = self.get_intersect(line.vertices, trackline.vertices)
                if temp[0]>=0:
                    collision = True
        return collision


    def createCircles(self, number):
        position = [40.0,40.0]
        circles = []
        for i in range(number):
            circle, indices = self.createIndexedVertices(position[0], position[1], self.circleRadius, self.circleNumPoints)
            vertex_count = len(circle) // 2
            circles.append(self.batch.add_indexed(vertex_count, pyglet.gl.GL_TRIANGLES, self.groupCircles,
                    indices,
                    ('v2f', circle),
                    ('c4f', (1, 1, 1, 0.8) * vertex_count)))
        return circles

    def createIndexedVertices(self, x, y, radius, numPoints):
        sides = numPoints - 2
        vertices = [x, y]
        for side in range(sides):
            angle = side * 2.0 * math.pi / sides
            vertices.append(x + math.cos(angle) * radius)
            vertices.append(y + math.sin(angle) * radius)
        # Add a degenerated vertex
        vertices.append(x + math.cos(0) * radius)
        vertices.append(y + math.sin(0) * radius)

        indices = []
        for side in range(1, sides+1):
            indices.append(0)
            indices.append(side)
            indices.append(side + 1)
        return vertices, indices

    def updateCircles(self):
        number = len(self.circles)
        for i in range(number):
            verts = []
            position = self.intersects[i]
            for k in range(self.circleNumPoints):
                angle = math.radians(float(k)/self.circleNumPoints * 360.0)
                x = self.circleRadius*math.cos(angle) + position[0]
                y = self.circleRadius*math.sin(angle) + position[1]
                verts += [x,y]
            self.circles[i].vertices = verts

    def createLines(self, number, lenght):
        lines = []
        step = 180 / number
        for i in range(number):
            x1 = self.sprite.x
            y1 = self.sprite.y
            x2 = x1 + lenght * math.sin(math.radians(self.angle+11.25+i*step))
            y2 = y1 + lenght * math.cos(math.radians(self.angle+11.25+i*step))

            lines.append(self.batch.add(2, pyglet.gl.GL_LINES, self.groupLines,
                ('v2f', (x1, y1, x2, y2)),
                ('c4B', (50, 50, 50, 255) * 2)
            ))
        return lines

    def updateLines(self, track):
        step = 180 / len(self.lines)
        lenght = self.maxLineLength
        for i, line in enumerate(self.lines):
            x1 = self.sprite.x
            y1 = self.sprite.y
            x2 = x1 + lenght * math.sin(math.radians(self.angle+11.25+i*step))
            y2 = y1 + lenght * math.cos(math.radians(self.angle+11.25+i*step))
            intersect = [-1, -1]
            dist = self.maxLineLength
            for trackline in track.lines:
                temp = self.get_intersect([x1, y1, x2, y2], trackline.vertices)
                if temp[0]>=0:
                    tempDist = self.distance2([x1, y1], temp)
                    if tempDist < dist:
                        dist = tempDist
                        intersect = temp
            self.intersects[i] = intersect
            self.distances[i] = dist
            if intersect[0] < 0 and intersect[1] < 0:
                line.vertices = [x1, y1, x2, y2]
            else:
                line.vertices = [x1, y1, intersect[0], intersect[1]]

    def get_intersect(self, lineA, lineB):
        #s = np.vstack([lineA[:2], lineA[2:], lineB[:2], lineB[2:]])
        #h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        #l1 = np.cross(h[0], h[1])           # get first line
        #l2 = np.cross(h[2], h[3])           # get second line
        #x, y, z = np.cross(l1, l2)          # point of intersection
        # if z == 0:                          # lines are parallel
        #    point = [float('inf'), float('inf')]
        # else:
        #    point = [x/z, y/z]

        x1,y1,x2,y2,x3,y3,x4,y4 = lineA[0],lineA[1],lineA[2],lineA[3],lineB[0],lineB[1],lineB[2],lineB[3]
        if ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) == 0:
            point = [float('inf'), float('inf')]
        else:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            point =[px,py]
        epsilon = 0.01
        if (point[0] < lineA[0]-epsilon and point[0] < lineA[2]-epsilon) or (point[0] > lineA[0]+epsilon and point[0] > lineA[2]+epsilon):
            return [-1,-1]
        if (point[1] < lineA[1]-epsilon and point[1] < lineA[3]-epsilon) or (point[1] > lineA[1]+epsilon and point[1] > lineA[3]+epsilon):
            return [-1,-1]
        if (point[0] < lineB[0]-epsilon and point[0] < lineB[2]-epsilon) or (point[0] > lineB[0]+epsilon and point[0] > lineB[2]+epsilon):
            return [-1,-1]
        if (point[1] < lineB[1]-epsilon and point[1] < lineB[3]-epsilon) or (point[1] > lineB[1]+epsilon and point[1] > lineB[3]+epsilon):
            return [-1,-1]
        return point

    def getReward(self,track):
        for line in self.collisionLines:
            intersect = self.get_intersect(line.vertices, track.rewardLines[self.rewardIndex])
            if intersect[0] >= 0:
                self.reward = self.reward + self.rewardPerLine
                self.rewardIndex = self.rewardIndex + 1
                if self.rewardIndex >= len(track.rewardLines):
                    self.rewardIndex = 0
        return self.reward

    def draw(self):
        self.batch.draw()

    def getUnitVector(self, angle):
        return np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])

    def rotate2d(self, vector, angle):
        cos = math.cos(math.radians(angle))
        sin = math.sin(math.radians(angle))
        return np.array([vector[0]*cos - vector[1]*sin, vector[0]*sin + vector[1]*cos])

    def rotatePointAroundPoint(self, point, origin, angle):
        cos = math.cos(math.radians(angle))
        sin = math.sin(math.radians(angle))
        xr = cos*(point[0]-origin[0])-sin*(point[1]-origin[1]) + origin[0]
        yr = sin*(point[0]-origin[0])+cos*(point[1]-origin[1]) + origin[1]
        return [xr,yr]

    def distance(self, points):
        return math.hypot(points[2]-points[0], points[3]-points[1])

    def distance2(self, pointA, pointB):
        return math.hypot(pointB[0]-pointA[0], pointB[1]-pointA[1])

    def process_input(self, dt, action):
        up,down,left,right = action[0], action[1], action[2], action[3]
        # Engineforce
        if up:
            self.engineforce = self.maxEngineforce
        else:
            self.engineforce = 0.0
        if down:
            self.brakeforce = self.maxBrakeforce
        else:
            self.brakeforce = 0.0

        # Steering
        if left:
            self.steeringAngle -= self.steeringVelocity*dt
        if right:
            self.steeringAngle += self.steeringVelocity*dt
        if not left and not right:
            if self.steeringAngle:
                self.steeringAngle -= self.steeringVelocity*dt*math.copysign(1.0,self.steeringAngle)
                if math.fabs(self.steeringAngle) <= 0.5:
                    self.steeringAngle = 0.0
        self.steeringAngle = max(-self.maxSteeringAngle, min(self.steeringAngle, self.maxSteeringAngle))

    def updateCar(self):
        self.sprite.x = self.position[0]
        self.sprite.y = self.position[1]
        self.sprite.rotation = self.angle

    def update(self, dt, track):
        self.do_physics(dt)
        self.updateLines(track)
        self.updateCircles()
        self.updateCar()
        return np.reshape(self.distances, [1, self.numberOfLines]), self.getReward(track), self.updateCollisionLines(track)



# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == '__main__':


    window = pyglet.window.Window(width=1280, height=620)
    keys = pyglet.window.key.KeyStateHandler()
    window.push_handlers(keys)

    agent = DQNAgent(8,16)
    episodes = 1000
    episode = 0
    batch_size = 32

    @window.event
    def on_draw():
        window.clear()
        track.draw()
        car.draw()

    def newEpisode():
        global car, track, done, state, reward, episode, episodes

        episode = episode +1
        if episode > episodes:
            pyglet.app.exit()

        car = Car(110.0, 240.0)
        track = Track()

        state, reward, done = car.update(0, track)


    def update(dt):
        dt = dt *5
        global keys, car, done, track, state, reward, agent, batch_size

        actionInt = agent.act(state)
        action = [bool(actionInt & (1<<n)) for n in range(4)]

        #up,down,left,right = False, False, False, False    # action = funktion(state)
        #if keys[pyglet.window.key.UP]:
        #    up = True
        #if keys[pyglet.window.key.DOWN]:
        #    down = True
        #if keys[pyglet.window.key.LEFT]:
        #    left = True
        #if keys[pyglet.window.key.RIGHT]:
        #    right = True
        #car.process_input(dt, [up, down, left, right])

        car.process_input(dt, action)
        next_state, reward, done = car.update(dt, track)

        agent.remember(state, actionInt, reward, next_state, done)
        state = next_state

        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(episode, episodes, reward, agent.epsilon))
            newEpisode()

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    newEpisode()

    pyglet.clock.schedule_interval(update,1/60.0)
    pyglet.app.run()
