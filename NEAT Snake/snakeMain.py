#inputs
    #collision dist up, down, left, right
    #fruit distance
#outputs
    #move up, down, left, right
#activation function
    #???
#fitness function
    #fruit is + 20
    #each frame is plus 1 survival
    #survival bonus goes down if fruit dist increased

#dist from fruit
    #take the lin dist from point to point of head x y and fruit x y
#collision dist is harder
    #send out raycasts in 4 directions until something is hit?


from cmath import sqrt
import pygame, sys, time, random, os, neat

speed = 500
maxScore = 0

frame_size_x = 720
frame_size_y = 720

err = pygame.init()

if(err[1] > 0):
    print("ERROR: " + err[1])
else:
    print("Game Init")

pygame.display.set_caption("SNAKE")
game_window = pygame.display.set_mode((frame_size_x , frame_size_y))

black = pygame.Color(0,0,0)
white = pygame.Color(255,255,255)
red = pygame.Color(255,0,0)
green = pygame.Color(0,255,0)
blue = pygame.Color(0,0,255)


fps_cont = pygame.time.Clock()

blockSize = 20

def init_vars():
    global headPos, snakeBody, foodPos, foodSpawn, score, direction, death, hunger
    direction = "RIGHT"
    headPos = [120,60]
    snakeBody = [[120,60]]
    foodPos = [random.randrange(1,(frame_size_x // blockSize)) * blockSize, 
                random.randrange(1,(frame_size_y // blockSize)) * blockSize]
    foodSpawn = True
    score = 0
    death = False
    hunger = 500
init_vars()

def show_score(choice, color, font, size):
    scoreFont = pygame.font.SysFont(font, size)
    scoreSurface = scoreFont.render("Score: " + str(score), True, color)
    scoreRect = scoreSurface.get_rect()
    if choice == 1:
        scoreRect.midtop = (frame_size_x / 10, 15)
    else:
        scoreRect.midtop = (frame_size_x/2, frame_size_y/1.25)
    
    game_window.blit(scoreSurface, scoreRect)

def getCollision():
    maxRight = 0
    maxLeft = 0
    maxUp = 0
    maxDown = 0
    for block in snakeBody[1:]:
        if (headPos[0] == block[0]):
            if headPos[1] > block[1] and abs(headPos[1] - block[1]) > maxUp:
                maxUp = abs(headPos[1] - block[1])
            elif abs(headPos[1] - block[1]) > maxDown:
                maxDown = abs(headPos[1] - block[1])

        if (headPos[1] == block[1]):
            if headPos[0] > block[0] and abs(headPos[0] - block[0]) > maxRight:
                maxRight = abs(headPos[0] - block[0])
            elif abs(headPos[0] - block[0]) > maxLeft:
                maxLeft = abs(headPos[0] - block[0])

    if maxRight == 0:
        maxRight = headPos[0] - frame_size_x - blockSize
    if maxLeft == 0:
        maxLeft = headPos[0]
    if maxUp == 0:
        maxUp = headPos[1]
    if maxDown == 0:
        maxDown = headPos[1] - frame_size_y - blockSize
    
    if maxRight <= blockSize:
        maxRight = 0
    else:
        maxRight = 1
    if maxLeft <= blockSize:
        maxLeft = 0
    else:
        maxLeft = 1
    if maxUp <= blockSize:
        maxUp = 0
    else:
        maxUp = 1
    if maxDown <= blockSize:
        maxDown = 0
    else:
        maxDown = 1

    return [maxRight,maxLeft,maxDown,maxUp]

def main(genomes, config):
    global headPos, snakeBody, foodPos, foodSpawn, score, direction, death, hunger, speed, maxScore

    for genome_id, g in genomes:
        #inputs = [0,0,0,0,0,0,0,0]
        inputs = [0,0,0,0,0,0]
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0
        init_vars()
        while death == False:

            #get food dist
            inputs[0] = foodPos[0] - headPos[0]
            inputs[1] = foodPos[1] - headPos[1]
            #get collisions
            collision_data = getCollision()
            inputs[2] = collision_data[0]
            inputs[3] = collision_data[1]
            inputs[4] = collision_data[2]
            inputs[5] = collision_data[3]
            # inputs[2] = headPos[0]
            # inputs[3] = headPos[1]
            # inputs[5] = 1
            #get food direction
            # if headPos[0] <= foodPos[0]:
            #     inputs[5] = inputs[5] * -1
            # if headPos[1] <= foodPos[1]:
            #     inputs[5] = inputs[5] * 2 

            output = net.activate(inputs)
            action = max(output)

            if output[0] == action:
                if direction == "DOWN":
                    g.fitness -= 2.5
                else:
                    direction = "UP"
            elif output[1] == action:
                if direction == "UP":
                    g.fitness -= 2.5
                else:
                    direction = "DOWN"
            elif output[2] == action:
                if direction == "LEFT":
                    g.fitness -= 2.5
                else:
                    direction = "RIGHT"
            else:
                if direction == "RIGHT":
                    g.fitness -= 2.5
                else:
                    direction = "LEFT"

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if( event.key == ord("s")):
                        if speed > 10:
                            speed -= 10
                            print(speed)
                    elif( event.key == ord("w")):
                        speed += 10
                        print(speed)
                # elif event.type == pygame.KEYDOWN:
                #     if ( event.key == pygame.K_UP or event.key == ord("w") and direction != "DOWN"):
                #         direction = "UP"
                #     elif ( event.key == pygame.K_LEFT or event.key == ord("a") and direction != "RIGHT"):
                #         direction = "LEFT"
                #     elif ( event.key == pygame.K_RIGHT or event.key == ord("d") and direction != "LEFT"):
                #         direction = "RIGHT"
                #     elif ( event.key == pygame.K_DOWN or event.key == ord("s") and direction != "UP"):
                #         direction = "DOWN"
            
            if direction == "UP":
                headPos[1] -= blockSize
            elif direction == "DOWN":
                headPos[1] += blockSize
            elif direction == "RIGHT":
                headPos[0] += blockSize
            elif direction == "LEFT":
                headPos[0] -= blockSize

            #check if closer
            newDist = (  ((headPos[0] - foodPos[0])**2) + ((headPos[1] - foodPos[1])**2) )**0.5
            if inputs[0] >= newDist and newDist != 0:
                g.fitness += 0.001 #* 1018/newDist 
            else:
                g.fitness -= 0.001#1 * newDist/1018

            #EAT
            snakeBody.insert(0, list(headPos))
            if headPos[0] == foodPos[0] and headPos[1] == foodPos[1]:
                score += 1
                foodSpawn = False
                g.fitness += 100#score*20
                hunger = 5000
            else:
                snakeBody.pop()
            
            #ADD FRUIT
            if not foodSpawn:
                foodPos = [random.randrange(1,(frame_size_x // blockSize)) * blockSize, 
                    random.randrange(1,(frame_size_y // blockSize)) * blockSize]
                foodSpawn = True
            
            #GFX
            game_window.fill(black)
            for pos in snakeBody:
                pygame.draw.rect(game_window, green, pygame.Rect(pos[0] + 2, pos[1] + 2, blockSize - 2, blockSize - 2))

            pygame.draw.rect(game_window, red, pygame.Rect(foodPos[0], foodPos[1], blockSize - 2, blockSize - 2))

            #Check border
            if (headPos[0] < 0) or (headPos[0] > frame_size_x - blockSize) or (headPos[1] < 0) or (headPos[1] > frame_size_y - blockSize):
                death = True
                g.fitness -= 50
            
            #Check if snake collision
            for block in snakeBody[1:]:
                if (headPos[0] == block[0] and headPos[1] == block[1]):
                    death = True
                    g.fitness -= 30

            hunger -= 1

            if hunger <= 0:
                death = True
            
            if death == True:
                g.fitness -= 10
            show_score(1, white, 'consolas', 20)
            pygame.display.update()
            fps_cont.tick(speed)
        if score > maxScore:
            maxScore = score
    print(maxScore)



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)    

    winner = p.run(main,100)
    print("Highscore: " + str(maxScore))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-ff.txt")
    run(config_path)