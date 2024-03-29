from neurosystems_mazelab.mazelib import Maze as MLMaze
from neurosystems_mazelab.generate.TrivialMaze import TrivialMaze
from neurosystems_mazelab.generate.Prims import Prims
from neurosystems_mazelab.solve.Tremaux import Tremaux
from neurosystems_mazelab.solve.BacktrackingSolver import BacktrackingSolver
import numpy as np

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import display as idp

colormap = colors.ListedColormap(["white", "black", "red", "green", "blue", "pink"])

DIRECTION_ARROW = np.array(
    [
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 1, 4, 4, 4],
        [4, 4, 1, 4, 1, 4, 4],
        [4, 1, 4, 1, 4, 1, 4],
        [4, 4, 4, 1, 4, 4, 4],
        [4, 4, 4, 1, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
    ]
)

# Define some mazes. Each maze comes with a start and end position.
MAZE_LINE = {
    "grid": np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    ),
    "start": (5, 1),
    "end": (1, 1),
}

MAZE_CORNER = {
    "grid": np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    ),
    "start": (5, 4),
    "end": (1, 1),
}

MAZE_ZIGZAG = {
    "grid": np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    "start": (1, 7),
    "end": (1, 1),
}

MAZE_LOOP = {
    "grid": np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    "start": (7, 7),
    "end": (1, 1),
}

MAZE_SPIRAL = {
    "grid": np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    "start": (1, 1),
    "end": (9, 11),
}

MAZE_EMPTY = {
    'grid': np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
    ),
    'start': (2, 2),
    'end': (3, 3)
}

MAZE_FILLED = {
    'grid': np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    ),
    'start': (1, 1),
    'end': (2, 2)
}

MAZE_INEFFICIENT = {
    'grid': np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
    ),
    'start': (3, 2),
    'end': (3, 3)
}

DIERCTONS_NESW = ["N", "E", "S", "W"]
LEFT_TURNS = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_TURNS = {"N": "E", "E": "S", "S": "W", "W": "N"}


class Maze(MLMaze):
    def __init__(self, width, height, preMaze=None, generator=None):
        self.isWaterMaze = False

        if preMaze is not None:
            self.grid = preMaze["grid"]
            self.start = preMaze["start"]
            self.end = preMaze["end"]
        else:
            if generator is None:
                self.generator = Prims(width, height)
            else:
                self.generator = generator(width, height)
            self.generate()
            self.generate_entrances(False, False)

        self.solver = BacktrackingSolver()
        self.prune = False
        self.solve()

        # create a copy of the grid with the start and end positions marked
        self.drawGrid = self.grid.copy()
        self.drawGrid[self.start[0]][self.start[1]] = 2
        self.drawGrid[self.end[0]][self.end[1]] = 3

        self.direction_arrow = DIRECTION_ARROW

    def view(self):
        plt.axis("off")
        grid = self.drawGrid.copy()
        grid[self.end[0]][self.end[1]] = 3
        if self.isWaterMaze: # make the platform bigger
            grid[self.end[0] + 1][self.end[1]] = 3
            grid[self.end[0]][self.end[1] + 1] = 3
            grid[self.end[0] + 1][self.end[1] + 1] = 3
        plt.imshow(colormap(grid))

    def waterMaze(self):
        maze = self.emptyCopy() # remove the walls
        end_x = maze['end'][0]
        end_y = maze['end'][1]
        # move platform if it is too close to the edge
        if end_x >= maze['grid'].shape[0] - 2:
            end_x -= 1
        if end_y >= maze['grid'].shape[1] - 2:
            end_y -= 1
        
        maze['end'] = (end_x, end_y)

        waterMaze = Maze(None, None, preMaze=maze)
        waterMaze.isWaterMaze = True
        return waterMaze

    def emptyCopy(self):
        emptyGrid = np.zeros(self.grid.shape, dtype=int)
        # make sides walls
        emptyGrid[0, :] = 1
        emptyGrid[-1, :] = 1
        emptyGrid[:, 0] = 1
        emptyGrid[:, -1] = 1
        preMaze = {
            "grid": emptyGrid,
            "start": self.start,
            "end": self.end
        }
        return preMaze
        
    def animate(self, steps, directions=None, goalReached=False):
        drawGrid = self.drawGrid.copy()
        if self.isWaterMaze: # hide platform
            drawGrid[self.end[0]][self.end[1]] = 0
        frames = [drawGrid]
        grid = drawGrid.copy()
        prev_step = steps[0]

        for i, step in enumerate(steps):
            if grid[step[0]][step[1]] == 4:  # if the robot is backtracking
                grid[prev_step[0]][
                    prev_step[1]
                ] = 5  # mark the previous step as backtracked
            else:  # if the robot is moving forward
                grid[prev_step[0]][
                    prev_step[1]
                ] = 4  # mark the previous step as visited

            prev_step = step

            grid_copy = grid.copy()  # copy the grid to avoid modifying the original grid
            # if self.isWaterMaze:
            # hide platform
            #     grid_copy[self.end[0] + 1][self.end[1]] = 3
            #     grid_copy[self.end[0]][self.end[1] + 1] = 3
            #     grid_copy[self.end[0] + 1][self.end[1] + 1] = 3

            if directions is None:  # if move() is used instead of move_directional()
                grid_copy[step[0]][
                    step[1]
                ] = 2  # mark the current step as the robot's position
                frames.append(grid_copy)  # append the grid to the frames
            else:  # if move_directional() is used
                expanded_grid = self.draw_grid_directional(
                    grid_copy, step, DIERCTONS_NESW.index(directions[i])
                )
                frames.append(
                    expanded_grid
                )  # append the grid with the direction of the robot

        # Create the figure and axes objects
        fig, ax = plt.subplots()
        plt.axis("off")  # Hide the axes
        if goalReached:
            plt.title("Goal Reached! Well Done!")

        fig.subplots_adjust(
            left=0, bottom=0, right=1, top=0.95, wspace=None, hspace=None
        )

        # Set the initial image
        im = ax.imshow(colormap(frames[0]), animated=True)

        def update(i):
            im.set_array(colormap(frames[i]))
            return (im,)

        animation.embed_limit = 2**128

        # Create the animation object
        animation_fig = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=100, blit=True
        )

        video = animation_fig.to_jshtml()

        html = idp.HTML(video)
        idp.display(html)

        # Good practice to close the plt object.
        plt.close()

    def draw_grid_directional(self, grid, step, direction):
        """
        Draws the grid with the direction of the robot by upsampling the grid
        and adding the direction arrow.

        Args:
            grid (np.array): the grid to be drawn
            step (tuple): the position of the robot
            direction (int): the direction of the robot (0 - forward, 1 - right, 2 - backward, 3 - left)
        """

        # Get the direction arrow and adjust upsampling factor to the size of the arrow
        direction_arrow = self.direction_arrow
        upsampling_factor = direction_arrow.shape[0]

        # Expand the grid by repeating each element upsampling_factor times
        expanded_grid = np.repeat(
            np.repeat(grid, upsampling_factor, axis=0), upsampling_factor, axis=1
        )

        # Rotate the arrow to the correct direction and add to the grid at the robot's position
        direction_arrow = np.rot90(direction_arrow, k=direction, axes=(1, 0))
        expanded_grid[
            step[0] * upsampling_factor : (step[0] + 1) * upsampling_factor,
            step[1] * upsampling_factor : (step[1] + 1) * upsampling_factor,
        ] = direction_arrow

        return expanded_grid


class Robot:
    def __init__(self, maze) -> None:
        self.maze = maze
        self.position = self.maze.start
        self.direction = "N"
        self.path = [[list(self.position)], [self.direction]]

        self.goalReached = False

        pass

    def newStart(self):
        new_direction = np.random.randint(0, 3)
        self.direction = ["N", "E", "S", "W"][new_direction]
        start_x = np.random.randint(1, self.maze.grid.shape[0] - 2)
        start_y = np.random.randint(1, self.maze.grid.shape[1] - 2)
        self.position = (start_x, start_y)
        self.path = [[list(self.position)], [self.direction]]
        self.maze.drawGrid[self.maze.start[0]][self.maze.start[1]] = 0 # remove old start
        self.maze.start = (start_x, start_y) # add new start
        self.maze.drawGrid[self.maze.start[0]][self.maze.start[1]] = 2 # add new start

    def checkGoal(self):
        if self.maze.isWaterMaze:
            if self.position[0] == self.maze.end[0] or self.position[0] == self.maze.end[0] + 1:
                if self.position[1] == self.maze.end[1] or self.position[1] == self.maze.end[1] + 1:
                    self.goalReached = True
                    return True
                return False

        if self.position == list(self.maze.end):
            self.goalReached = True
            return True

        return False

    def getPath(self):
        return np.array([np.array(step) for step in self.path[0]])

    # def checking if a position is valid
    def __isvalidPos(self, pos, step=1):
        if pos[0] < 0 or pos[0] >= self.maze.grid.shape[0]:
            return False
        if pos[1] < 0 or pos[1] >= self.maze.grid.shape[1]:
            return False

        # if there is more than one step to be taken, trace the path
        if step > 1:
            if self.maze.grid[pos[0]][pos[1]] == 1:
                return False
            return self.__isvalidPos([pos[0], pos[1]], step - 1)

        if self.maze.grid[pos[0]][pos[1]] == 1:
            return False
        return True

    # move the robot to a new position
    def __move(self, direction, step=1):
        # directions: 1 - up, 2 - right, 3 - down, 4 - left
        proposed_pos = [self.position[0], self.position[1]]
        if direction == 4:
            proposed_pos[1] -= step
        elif direction == 3:
            proposed_pos[0] += step
        elif direction == 2:
            proposed_pos[1] += step
        elif direction == 1:
            proposed_pos[0] -= step

        if self.__isvalidPos(proposed_pos, step):
            self.position = proposed_pos
            self.path[0].append(self.position)

            self.checkGoal()

            return True
        return False

    # def move_directional(self, step):
    #     # directions: 0 - forward, 1 - right, 2 - backward, 3 - left
    #     directional_move = (step + self.direction) % 4 + 1
    #     if self.move(directional_move):
    #         self.direction = step
    #         self.directions.append(self.direction)
    #         return True
    #     return False

    def turnLeft(
        self,
    ):  # Turns the direction to the left and appends new position and direction
        self.direction = LEFT_TURNS[self.direction]
        self.path[0].append(self.position)
        self.path[1].append(self.direction)
        pass

    def turnRight(
        self,
    ):  # Turns the direction to the left and appends new position and direction
        self.direction = RIGHT_TURNS[self.direction]
        self.path[0].append(self.position)
        self.path[1].append(self.direction)
        pass

    def moveForward(self):  # Moves in the current direction
        if self.__move(DIERCTONS_NESW.index(self.direction) + 1):
            self.path[1].append(self.direction)
        pass

    def lookAround(self, specificDirection=None):
        # returns a dictionary of the walls around the robot in the order of NESW
        # 0 - no wall, 1 - wall
        if specificDirection:
            return self.__isvalidPos(
                [
                    self.position[0]
                    + [-1, 0, 1, 0][DIERCTONS_NESW.index(specificDirection)],
                    self.position[1]
                    + [0, 1, 0, -1][DIERCTONS_NESW.index(specificDirection)],
                ]
            )
        else:
            walls = {}
            for i in range(4):
                walls[DIERCTONS_NESW[i]] = self.__isvalidPos(
                    [
                        self.position[0] + [-1, 0, 1, 0][i],
                        self.position[1] + [0, 1, 0, -1][i],
                    ]
                )
            return walls

    def run(self):
        self.maze.animate(self.path[0], self.path[1], self.goalReached)
