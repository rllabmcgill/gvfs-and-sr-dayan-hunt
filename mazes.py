# Import necessary packages
import curses

# Pycolab
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab import ascii_art
from pycolab import human_ui


def make_maze(maze, maze_type):
    if maze_type == 'twobytwo_R1':
        """Builds and returns a blocking maze gridworld game."""
        return ascii_art.ascii_art_to_game(
            maze, what_lies_beneath=' ',
            sprites={'P': PlayerSpriteR1})
    elif maze_type == 'twobytwo_R2':
        """Builds and returns a blocking maze gridworld game."""
        return ascii_art.ascii_art_to_game(
            maze, what_lies_beneath=' ',
            sprites={'P': PlayerSpriteR2})


class PlayerSpriteR1(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.

    This `Sprite` ties actions to going in the four cardinal directions. If we
    reach a magical location (in this example, (4, 3)), the agent receives a
    reward of 1 and the epsiode terminates.
    """

    def __init__(self, corner, position, character):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSpriteR1, self).__init__(
          corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things   # Unused.

        # Apply motion commands.
        if actions == 0:    # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)

        # See if we've found the goal:
        if self.position == (2, 2):
            the_plot.add_reward(0.0)
            the_plot.terminate_episode()
            # print("Terminating episode..")
            # time.sleep(10)

        else:
            the_plot.add_reward(-1.0)


class PlayerSpriteR2(prefab_sprites.MazeWalker):
    """A `Sprite` for our player.

    This `Sprite` ties actions to going in the four cardinal directions. If we
    reach a magical location (in this example, (4, 3)), the agent receives a
    reward of 1 and the epsiode terminates.
    """

    def __init__(self, corner, position, character):
        """Inform superclass that we can't walk through walls."""
        super(PlayerSpriteR2, self).__init__(
          corner, position, character, impassable='#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things   # Unused.

        # Apply motion commands.
        if actions == 0:    # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)

        # See if we've found the goal:
        if self.position == (2, 1):
            the_plot.add_reward(0.0)
            the_plot.terminate_episode()
            # print("Terminating episode..")
            # time.sleep(10)

        else:
            the_plot.add_reward(-1.0)


# Human control (for testing purposes):
# def main(argv=()):
# 	del argv  # Unused.
#
# 	# Build a four-rooms game.
# 	game = make_maze(BLOCKING_MAZE_INIT)
#
# 	# Make a CursesUi to play it with.
# 	ui = human_ui.CursesUi(
# 		keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
# 	                     curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
# 	                     -1: 4},
# 		delay=200)
#
# 	# Let the game begin!
# 	ui.play(game)
# if __name__ == '__main__':
# 	main(sys.argv)
