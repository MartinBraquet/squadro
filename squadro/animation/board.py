import pygame

from squadro.squadro_state import get_moves, SquadroState
from squadro.tools.constants import RESOURCE_PATH
from squadro.tools.timer import pretty_print_time

black = (255, 255, 255)
grey = (34, 34, 34)
yellow = (255, 164, 0)
red = (150, 0, 0)


class Board:
    """
    Board used for animation purposes only.
    Must not contain any logic intrinsic to the game.
    """

    def __init__(self, n_pawns, n_tiles=None, title=None):
        pygame.init()
        if title:
            pygame.display.set_caption(title)

        # Initialise screen
        self.n_pawns = n_pawns
        self.n_tiles = n_tiles if n_tiles is not None else n_pawns + 2
        self.screen = pygame.display.set_mode(
            (self.n_tiles * 100, self.n_tiles * 100),
            # pygame.RESIZABLE,
            # pygame.FULLSCREEN
        )
        self.screen.fill(0)


        # Resources
        self.tile = pygame.image.load(RESOURCE_PATH / "tile.png")
        self.corner = pygame.image.load(RESOURCE_PATH / "corner.png")
        self.start_l = [
            pygame.image.load(RESOURCE_PATH / f"start_{x}_l.png")
            for x in range(1, 4)
        ]
        self.start_b = [
            pygame.image.load(RESOURCE_PATH / f"start_{x}_b.png")
            for x in range(1, 4)
        ]
        self.start_r = [
            pygame.image.load(RESOURCE_PATH / f"start_{x}_r.png")
            for x in range(1, 4)
        ]
        self.start_t = [
            pygame.image.load(RESOURCE_PATH / f"start_{x}_t.png")
            for x in range(1, 4)
        ]
        self.yellow_pawn = pygame.image.load(RESOURCE_PATH / "yellow_pawn.png")
        self.red_pawn = pygame.image.load(RESOURCE_PATH / "red_pawn.png")
        self.yellow_pawn_ret = pygame.image.load(RESOURCE_PATH / "yellow_pawn_ret.png")
        self.red_pawn_ret = pygame.image.load(RESOURCE_PATH / "red_pawn_ret.png")
        self.yellow_pawn_fin = pygame.image.load(RESOURCE_PATH / "yellow_pawn_fin.png")
        self.red_pawn_fin = pygame.image.load(RESOURCE_PATH / "red_pawn_fin.png")

    @property
    def width(self):
        return self.screen.get_width()

    @property
    def height(self):
        return self.screen.get_height()

    def turn_draw(self, state):
        """
        Draw all the things for the current turn
        """
        self.draw_board(state)
        self.show_turn(state)
        pygame.display.flip()

    def draw_board(self, state):
        # Draw the tiles
        for i in range(1, self.n_tiles - 1):
            for j in range(1, self.n_tiles - 1):
                self.screen.blit(self.tile, (i * 100, j * 100))

        n_pixels = (self.n_pawns + 1) * 100

        self.screen.blit(self.corner, (0, 0))
        self.screen.blit(self.corner, (0, n_pixels))
        self.screen.blit(self.corner, (n_pixels, n_pixels))
        self.screen.blit(self.corner, (n_pixels, 0))

        moves = get_moves(self.n_pawns)

        for pawn in range(self.n_pawns):
            self.screen.blit(self.start_l[moves[0][pawn] - 1], (0, 100 * (pawn + 1)))
            self.screen.blit(self.start_b[moves[0][pawn] - 1], (100 * (pawn + 1), n_pixels))
            self.screen.blit(self.start_r[moves[1][pawn] - 1], (n_pixels, 100 * (pawn + 1)))
            self.screen.blit(self.start_t[moves[1][pawn] - 1], (100 * (pawn + 1), 0))

        # Draw the pawns
        pawn_images = dict(
            pawn_fin=[self.yellow_pawn_fin, self.red_pawn_fin],
            pawn_ret=[self.yellow_pawn_ret, self.red_pawn_ret],
            pawn=[self.yellow_pawn, self.red_pawn],
        )
        for pawn in range(self.n_pawns):
            for player in range(2):
                if state.is_pawn_finished(player, pawn):
                    pawn_img = pawn_images["pawn_fin"][player]
                elif state.is_pawn_returning(player, pawn):
                    pawn_img = pawn_images["pawn_ret"][player]
                else:
                    pawn_img = pawn_images["pawn"][player]
                pawn_pos = state.get_pawn_position(player, pawn)
                pawn_board_pos = pawn_to_board_position(pawn_pos)
                self.screen.blit(pawn_img, pawn_board_pos)

    def show_turn(self, state: SquadroState):
        # Draw whose turn it is
        font1 = pygame.font.Font("freesansbold.ttf", 12)
        text1 = font1.render("Current player:", True, black, grey)
        textRect1 = text1.get_rect()
        textRect1.center = (
            (self.n_tiles - 1) * 100 + 50,
            (self.n_tiles - 1) * 100 + 38
        )
        self.screen.blit(text1, textRect1)

        font2 = pygame.font.Font("freesansbold.ttf", 20)
        color = yellow if state.get_cur_player() == 0 else red
        text2 = font2.render("    ", True, color, color)
        textRect2 = text2.get_rect()
        textRect2.center = (
            (self.n_tiles - 1) * 100 + 50,
            (self.n_tiles - 1) * 100 + 62
        )
        self.screen.blit(text2, textRect2)

        font3 = pygame.font.Font("freesansbold.ttf", 12)
        text3 = font3.render("Time left:", True, black, grey)
        textRect3 = text3.get_rect()
        textRect3.center = (50, (self.n_tiles - 1) * 100 + 38)
        self.screen.blit(text3, textRect3)

        text3 = font3.render("Time left:", True, black, grey)
        textRect3 = text3.get_rect()
        textRect3.center = ((self.n_tiles - 1) * 100 + 50, 38)
        self.screen.blit(text3, textRect3)

        text3 = font3.render(f"Turn: {state.total_moves}", True, black, grey)
        textRect3 = text3.get_rect()
        textRect3.center = (50, 38)
        self.screen.blit(text3, textRect3)

    def show_timer(self, times_left):
        font = pygame.font.Font("freesansbold.ttf", 2)
        text4 = font.render(" " * 60, True, black, yellow)
        textRect4 = text4.get_rect()
        textRect4.center = (50, (self.n_tiles - 1) * 100 + 50)
        self.screen.blit(text4, textRect4)

        text4 = font.render(" " * 60, True, black, red)
        textRect4 = text4.get_rect()
        textRect4.center = ((self.n_tiles - 1) * 100 + 50, 50)
        self.screen.blit(text4, textRect4)

        font = pygame.font.Font("freesansbold.ttf", 12)
        text = font.render(f"  {pretty_print_time(times_left[0])}  ", True, black, grey)
        textRect = text.get_rect()
        textRect.center = (50, (self.n_tiles - 1) * 100 + 62)
        self.screen.blit(text, textRect)

        text = font.render(f"  {pretty_print_time(times_left[1])}  ", True, black, grey)
        textRect = text.get_rect()
        textRect.center = ((self.n_tiles - 1) * 100 + 50, 62)
        self.screen.blit(text, textRect)
        pygame.display.flip()

    def display_winner(self, state):
        """Print the winner"""
        assert state.game_over(), "Game is not over"
        font = pygame.font.Font("freesansbold.ttf", 48)

        winner = state.get_winner()
        if winner == 0:
            text = font.render(" Yellow wins! ", True, yellow, grey)
        elif winner == 1:
            text = font.render(" Red wins! ", True, red, grey)
        else:
            raise ValueError(f"Invalid winner: {winner}")

        textRect = text.get_rect()
        textRect.center = (self.width // 2, self.height // 2)
        self.screen.blit(text, textRect)

        # Print if time-out or invalid action
        if state.timeout_player is not None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent timed out", True,
                                 black, grey)
            textRect2 = text2.get_rect()
            textRect2.center = (self.width // 2, self.height // 2 + 34)
            self.screen.blit(text2, textRect2)
        elif state.invalid_player is not None:
            font2 = pygame.font.Font("freesansbold.ttf", 18)
            text2 = font2.render("The opponent made an invalid move", True,
                                 black, grey)
            textRect2 = text2.get_rect()
            textRect2.center = (self.width // 2, self.height // 2 + 34)
            self.screen.blit(text2, textRect2)

        pygame.display.flip()


def handle_events():
    # Events
    for event in pygame.event.get():
        check_quit(event)


def check_quit(event):
    # Quit when pressing the X button
    if event.type == pygame.QUIT:
        pygame.quit()
        raise SystemExit("Exiting due to pygame QUIT event")


def get_position_from_click():
    pos = pygame.mouse.get_pos()
    pos = board_to_pawn_position(pos)
    return pos


def pawn_to_board_position(pawn_pos):
    return tuple(map(lambda x: x * 100, pawn_pos))


def board_to_pawn_position(board_pos):
    """
    >>> board_to_pawn_position((101, 200))
    (1, 2)
    >>> board_to_pawn_position(pawn_to_board_position((1, 2)))
    (1, 2)
    """
    return tuple(map(lambda x: int(x / 100), board_pos))
