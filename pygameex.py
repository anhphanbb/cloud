import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Move Block')

# Define the block
block_color = (0, 128, 255)
block_size = 50
block_x, block_y = width // 2, height // 2
block_speed = 1
max_speed = 10
acceleration = 1

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:
        block_x -= block_speed
        block_speed = min(max_speed, block_speed + acceleration)
    elif keys[pygame.K_RIGHT]:
        block_x += block_speed
        block_speed = min(max_speed, block_speed + acceleration)
    elif keys[pygame.K_UP]:
        block_y -= block_speed
        block_speed = min(max_speed, block_speed + acceleration)
    elif keys[pygame.K_DOWN]:
        block_y += block_speed
        block_speed = min(max_speed, block_speed + acceleration)
    else:
        block_speed = 1  # Reset speed when no arrow key is pressed

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the block
    pygame.draw.rect(screen, block_color, (block_x, block_y, block_size, block_size))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)
