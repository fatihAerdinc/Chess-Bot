import pygame
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def board_to_vector(board):
    vector = np.zeros(64)
    for i, piece in enumerate(board.piece_map().values()):
        vector[i] = 1 if piece.color == chess.WHITE else -1
    return torch.tensor(vector, dtype=torch.float32)

def choose_move(board, model):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -1

    for move in legal_moves:
        board.push(move)
        board_vector = board_to_vector(board)
        score = model(board_vector).item()
        if score > best_score:
            best_score = score
            best_move = move
        board.pop()

    return best_move

def train(model, board, result):
    board_vector = board_to_vector(board)
    optimizer.zero_grad()
    prediction = model(board_vector)
    loss = criterion(prediction, torch.tensor([result], dtype=torch.float32))
    loss.backward()
    optimizer.step()

pygame.init()

WIDTH, HEIGHT = 512, 512  
SQUARE_SIZE = WIDTH // 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

pieces = {rook.png, bishop.png, knight.png, king.png, queen.png, pawn.png}
piece_symbols = ['p', 'r', 'n', 'b', 'q', 'k', 'P', 'R', 'N', 'B', 'Q', 'K']
for piece in piece_symbols:
    pieces[piece] = pygame.transform.scale(
        pygame.image.load(f'images/{piece}.png'), (SQUARE_SIZE, SQUARE_SIZE)
    )

board = chess.Board()

def draw_board(screen):
    colors = [WHITE, BLACK]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = pieces[piece.symbol()]
            col, row = square % 8, square // 8
            screen.blit(piece_image, (col * SQUARE_SIZE, (7 - row) * SQUARE_SIZE))

def main_game(model):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess Game')
    running = True
    selected_square = None

    while running:
        draw_board(screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if board.turn: 
                    mouse_x, mouse_y = event.pos
                    col = mouse_x // SQUARE_SIZE
                    row = 7 - (mouse_y // SQUARE_SIZE)
                    clicked_square = chess.square(col, row)

                    if selected_square is None and board.piece_at(clicked_square):
                        selected_square = clicked_square
                    elif selected_square is not None:
                        move = chess.Move(selected_square, clicked_square)
                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None

                            if not board.is_game_over():
                                model_move = choose_move(board, model)
                                if model_move:
                                    board.push(model_move)

        if board.is_game_over():
            print("Game Over:", board.result())
            running = False

    pygame.quit()

main_game(model)