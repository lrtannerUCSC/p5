import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome, generation):
        mutation_rate = 0.1 / (1 + generation)

        tile_weights = {
            "-": 1,  # Empty space (low weight)
            "X": 5,  # Solid wall (high weight)
            "B": 5,  # Breakable block
            "?": 2,  # Question block
            "M": 1,  # Mushroom block
            "E": 10, # Enemy
            "o": 30, # Coin
            "T": 10,  # Pipe top
        }
        tiles = list(tile_weights.keys())
        weights = list(tile_weights.values())

        # Iterate over each cell in the grid (excluding the first and last columns)
        left = 2
        right = width - 1
        for y in range(height):
            for x in range(left, right):

                if genome[y][x] in ["T", "|"]:
                    continue  # Protect pipes from being overwritten
            
                initial_mutation_rate = mutation_rate
                # Increase mutation rate for empty spaces
                if genome[y][x] == "-":
                    mutation_rate = initial_mutation_rate * 5

                if random.random() < mutation_rate:
                    # Count neighboring blocks
                    neighbor_count = self.count_neighboring_blocks(genome, x, y)
                    # Increase the probability of placing a block if there are neighboring blocks
                    if neighbor_count > 0:
                        # Increase weights of non-empty tiles
                        adjusted_weights = [w * (1 + neighbor_count) if tile != "-" else w for tile, w in zip(tiles, weights)]
                    else:
                        adjusted_weights = weights
                    
                    # Select a new tile based on adjusted weights
                    new_tile = random.choices(tiles, weights=adjusted_weights, k=1)[0]
                    
                    # Ensure the new tile is valid
                    if self.is_valid_tile(genome, new_tile, x, y):
                        genome[y][x] = new_tile
                    else:
                        # If the placement is invalid, try to place required blocks
                        if random.random() < (0.2 / (1 + generation)):  # 10% chance to attempt fixing decreasing with each gen
                            self.place_required_blocks(genome, new_tile, x, y)
                mutation_rate = initial_mutation_rate # Reset mutation rate
        return genome
    
    # Helper function for mutations to check if new tile is valid
    def is_valid_tile(self, genome, tile, x, y):

        # Ensure tile is allowed at y level
        if not self.is_allowed_at_level(genome, tile, x, y):
            return False

        # Ensure pipe has a supported top
        if not self.is_pipe_top_allowed(genome, tile, x, y):
            return False
            
        # Ensure enemies do not spawn at the ground level
        if not self.enemy_near_ground(genome, tile, x, y):
            return False
        
        # Ensure bricks are in groups
        if not self.bricks_in_groups(genome, tile, x, y):
            return False
        
        # Ensure question mark blocks are reachable
        if not self.is_block_reachable(genome, tile, x, y):
            return False
        
        # Ensure coins are reachable
        if not self.is_coin_reachable(genome, tile, x, y):
            return False
        
        # Ensure wall blocks are supported
        if not self.is_wall_block_supported(genome, tile, x, y):
            return False
    
        # Add more constraints as needed
        return True
    
    def place_required_blocks(self, genome, tile, x, y):

        tile_weights = {
            "B": 10,  # Breakable block
            "?": 3,  # Question block
            "M": 1,  # Mushroom block
        }
        tiles = list(tile_weights.keys())
        weights = list(tile_weights.values())

        # Do not force enemy mutation
        if tile == "E":
            return False

        # Place pipe bodies below until we reach a solid block or the bottom
        if tile == "T":  # Pipe top
            
            # If too high, do not place no matter what
            if not self.is_allowed_at_level(genome, tile, x, y):
                return False

            # Do not allow stacked pipe tops
            if not self.stack_pipe_top_check(genome, tile, x, y):
                return False

            for ny in range(y + 1, height-1):
                if genome[ny][x] in ["X", "B"]:  # Stop if we hit a solid block
                    break
                genome[ny][x] = "|"
            # Ensure pipe has top at the end
            if not self.pipe_has_top(genome, "|", x, y+1):
                pipe_found = False
                for nny in range (height-1, 0, -1):
                    if genome[nny][x] == "|":
                        pipe_found = True
                    if pipe_found and genome[nny][x] != "|":
                        genome[nny][x] = "T"
                        break
            return True  # Successfully placed required blocks

        elif tile in ["B", "?", "M"]:  # Brick, question block, or mushroom block
            # Determine the length of the group (random between 1 and 7)
            group_length = random.randint(1, 7)
            # Place blocks horizontally to the left and right
            for dx in range(-group_length, group_length + 1):
                nx = x + dx
                if 1 <= nx < width-1:  # Ensure within bounds and empty
                    new_tile = random.choices(tiles, weights, k=1)[0]
                    self.place_tile_safely(genome, nx, y, new_tile)  # Add the same type of block
            return True  # Successfully placed required blocks

        return False  # No action needed for other tiles   
    
    def is_allowed_at_level(self, genome, tile, x, y):
        # Only walls and empty space allowed at ground level
        if y >= height - 1 and tile not in ["X", "-"]:
            return False

        # Do not place right next to pipe since it is 2 wide
        if genome[y][x-1] in ["T", "|"]:
            return False
        
        if tile == "T":
            # Do not place above half height or near start
            if y < height/2 or x < width/4:
                return False
        
        if tile == "|":
            # Do not place above half height + 1
            if y < height/2 + 1 or x < width/4:
                return False
            
        if tile in ["B", "M", "?"]:
            # Do not place near ground level
            if y >= height - 3:
                return False
            
        if tile == "o":
            # Do not place at ground level
            if y >= height - 1:
                return False

        if tile == "E":
            # Do not place near start
            if x <= width/4:
                return False
        
        return True
        
    def place_tile_safely(self, genome, x, y, tile):
        if tile == "|":
            # Always allow placing pipe bodies, even if the target tile is part of a pipe
            genome[y][x] = tile
            return True
        elif (genome[y][x] and genome[y][x-1]) not in ["T", "|"]:  # Do not overwrite existing pipes for other tiles
            genome[y][x] = tile
            return True
        return False 
    
    def stack_pipe_top_check(self, genome, tile, x, y):
        # Do not allow stacked tops
        for ny in range(0, height):
            if genome[ny][x] == "T":
                return False
        return True
    
    def is_pipe_top_allowed(self, genome, tile, x, y):
        if tile == "T":
            # Do not allow stacked tops
            if not self.stack_pipe_top_check(genome, tile, x, y):
                return False
            # Must have pipe body, ground or platform beneath it
            for ny in range(y + 1, height):
                if genome[ny][x] not in ["X", "B", "|"]:
                    return False
            # Must have platform nearby to be able to scale pipe
            if not self.is_pipe_scalable(genome, tile, x, y):
                return False
            return True
        return True

    def is_wall_block_supported(self, genome, tile, x, y):
        if tile == "X":  # Only apply this check to wall blocks
            if y == height - 1:  # Ground level, always valid
                return True
            # Check all tiles below this "X" block
            for ny in range(y + 1, height):
                if genome[ny][x] != "X":
                    return False  # If any tile below is not "X", the block is unsupported
            return True  # All tiles below are "X", so the block is supported
        # If the tile is not an "X" block, skip this check
        return True

    def enemy_near_ground(self, genome, tile, x, y):
        if tile == "E":
            if y >= height - 1:
                return False
            # Check if the tile below is either ground or platform
            for dy in range(1, 4):
                ny = y + dy
                if ny <= height-1:
                    if genome[ny][x] in ["X", "B", "?", "M", "T"]:
                        return True
            return False
        return True
    
    def pipe_has_top(self, genome, tile, x, y):
        if tile == "|":
            for i in range(y-1, 1, -1):  # Scan upwards
                if genome[i][x] == "T":
                    return True
                if genome[i][x] in ["X", "-"]:  # Stop at solid ground or empty space
                    return False
        return False
 
    def bricks_in_groups(self, genome, tile, x, y):
        if tile == "B":
            # Check neighboring tiles (left, right, above, below)
            neighbors = [
                (x - 1, y),  # Left
                (x + 1, y),  # Right
                # (x, y - 1),  # Above
                # (x, y + 1),  # Below
            ]

            # No bricks at the ground level
            if y >= height - 2:
                return False
            
            # Count how many neighboring tiles are also bricks
            brick_count = 0
            for nx, ny in neighbors:
                if 0 <= nx < width and 0 <= ny < height:  # Ensure the neighbor is within bounds
                    if genome[ny][nx] == ("B" or "?" or "M"):
                        brick_count += 1
            
            # Require at least one adjacent brick
            if brick_count == 0:
                return False
        
        return True
    
    def is_block_reachable(self, genome, tile, x, y):
        if tile in ["?", "M"]:  # Only apply this check to question mark blocks

            # Check if the question block is too low to headbutt
            if y >= height - 2:
                return False
        
            # Check tiles 1 and 2 spots below the question block
            for dy in range(1, 2):  # Check 1 and 2 tiles below
                ny = y + dy  # Calculate the y-coordinate of the tile below
                if ny >= height:  # If the tile is below the bottom of the level, it's invalid
                    return False
                if genome[ny][x] in ["X", "B", "?", "M", "T"]:  # Ground or platform tiles
                    return False  # The block is not reachable
            
            # Check tiles 3 and 4 spots below the question block
            for dy in range(3, 4):  # Check 2 and 3 tiles below
                ny = y + dy  # Calculate the y-coordinate of the tile below
                if ny >= height:  # If the tile is below the bottom of the level, it's invalid
                    return False
                if genome[ny][x] in ["X", "B", "?", "M", "T"]:  # Ground or platform tiles
                    return True  # The block is reachable
                
            
            # If no ground or platform is found within 2-3 tiles below, the block is unreachable
            return False
            
        # If the tile is not a question mark block, skip this check
        return True
    
    def is_coin_reachable(self, genome, tile, x, y):
        if tile == "o":
            # Check tiles 3 and 4 spots below the question block
            for dy in range(1, 4):  # Check 1 to 4 tiles below
                ny = y + dy  # Calculate the y-coordinate of the tile below
                if ny < height:
                    if genome[ny][x] in ["X", "B", "?", "M", "T"]:  # Ground or platform tiles
                        return True  # The block is reachable
            
            # If no ground or platform is found within 1-3 tiles below, the coin is unreachable
            return False
        # If the tile is not a coin, skip this check
        return True
    
    # Check if there is ground or platform within jumping distance of the pipe top
    def is_pipe_scalable(self, genome, tile, x, y):
        if tile == "T":  # Check if it's a pipe top
            for dy in range(1, 4):  # Check 1-3 blocks below
                ny = y + dy
                if ny < height:  # Ensure within level bounds
                    for dx in range(-2, 0):  # Check 2 blocks to the left (-2 and -1)
                        nx = x + dx
                        if 0 <= nx < width:  # Ensure within level bounds
                            if genome[ny][nx] in ["B", "X", "M", "?"]:  # Check for solid ground
                                return True
            return False  # No valid platform found, not scalable
        
        return True  # Ignore if not a pipe top


    def count_neighboring_blocks(self, genome, x, y):
        count = 0
        for dy in range(-1, 2):  # Check rows above, current, and below
            for dx in range(-1, 2):  # Check columns left, current, and right
                if dx == 0 and dy == 0:  # Skip the current tile
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:  # Ensure within bounds
                    if genome[ny][nx] != "-":  # Count non-empty tiles
                        count += 1
        return count


    # Create zero or more children from self and other
    def generate_children(self, other, generation):
        # Calculate fitness for both parents
        self_fitness = self.fitness()
        other_fitness = other.fitness()

        # Determine the bias based on fitness
        total_fitness = self_fitness + other_fitness
        if total_fitness == 0:
            # If both parents have zero fitness, use equal probability
            bias = 0.5
        else:
            # Probability of selecting from self is proportional to its fitness
            bias = self_fitness / total_fitness
        
        new_genome_1 = copy.deepcopy(self.genome)
        # new_genome_2 = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1

        # Take column from either parent
        for x in range(left, right):
            if random.random() < bias:
                for y in range(height):
                    new_genome_1[y][x] = self.genome[y][x]
            else:
                for y in range(height):
                    new_genome_1[y][x] = other.genome[y][x]
                
        # do mutation; note we're returning a one-element tuple here
        new_genome_1 = self.mutate(new_genome_1, generation)
        # new_genome_2 = self.mutate(new_genome_2)
        return (Individual_Grid(new_genome_1),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.6,
            negativeSpace=0.5,
            pathPercentage=0.7,
            emptyPercentage=0.5,
            linearity=-0.4,
            solvability=2.5,
            enemyDensity=-0.3,
            coinDensity=0.2,
            platformVariety=0.4
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len([de for de in self.genome if de[1] == "6_stairs"]) > 5:
            penalties -= 2
        if len([de for de in self.genome if de[1] == "3_coin"]) > 20:
            penalties -= 3
        
        self._fitness = sum(
            coefficients[m] * measurements.get(m, 0) for m in coefficients
        ) + penalties

        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.15 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x, de_type = de[:2]
            choice = random.random()
            
            if de_type == "4_block":
                y, breakable = de[2], de[3]
                if choice < 0.4:
                    x = offset_by_upto(x, width / 10, min=1, max=width - 2)
                elif choice < 0.8:
                    y = offset_by_upto(y, height / 3, min=0, max=height - 1)
                else:
                    breakable = not breakable
                new_de = (x, de_type, y, breakable)
                
            elif de_type == "5_qblock":
                y, has_powerup = de[2], de[3]
                if choice < 0.4:
                    x = offset_by_upto(x, width / 10, min=1, max=width - 2)
                elif choice < 0.8:
                    y = offset_by_upto(y, height / 3, min=0, max=height - 1)
                else:
                    has_powerup = not has_powerup
                new_de = (x, de_type, y, has_powerup)
                
            elif de_type == "6_stairs":
                h, dx = de[2], de[3]
                if choice < 0.4:
                    x = offset_by_upto(x, width / 10, min=1, max=width - 2)
                elif choice < 0.8:
                    h = offset_by_upto(h, 5, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 10, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 3, min=0, max=height - 1)
                new_de = (x, de_type, y)
                
            new_genome[to_change] = new_de
        return new_genome

    def generate_children(self, other, generation):
        # STUDENT How does this work?  Explain it in your writeup.
        if len(self.genome) < 1:
            self.genome = Individual_DE.random_individual().genome
        if len(other.genome) < 1:
            other.genome = Individual_DE.random_individual().genome
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


# Individual = Individual_DE
Individual = Individual_Grid


def generate_successors(population,  generation, selection_method="tournament"):
    results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    # Use tournament selection to choose parents
    
    # Select the appropriate selection function
    if selection_method == "tournament":
        select_parent = tournament_select_parent
    elif selection_method == "roulette":
        select_parent = roulette_wheel_select_parent
    else:
        raise ValueError("Invalid selection method. Use 'tournament' or 'roulette'.")
    
    # Generate children until we have enough for the next generation
    while len(results) < len(population):
        # Select two parents
        parent1 = select_parent(population)
        parent2 = select_parent(population)
        
        # Generate children from the parents
        children = parent1.generate_children(parent2, generation)
        
        # Add the children to the results
        results.extend(children)
    
    # If we generated more children than needed (due to generate_children returning multiple children),
    # trim the results to match the population size
    if len(results) > len(population):
        results = results[:len(population)]
    
    return results

def tournament_select_parent(population):
        # Randomly select 20 individuals and return the one with higher fitness
        candidates = random.sample(population, 20)
        return max(candidates, key=lambda ind: ind.fitness())

def roulette_wheel_select_parent(population):
    # Calculate the total fitness of the population
    total_fitness = sum(ind.fitness() for ind in population)
    
    # If total fitness is zero, select a random individual
    if total_fitness == 0:
        return random.choice(population)
    
    # Pick a random value between 0 and total_fitness
    pick = random.uniform(0, total_fitness)
    
    # Iterate through the population and accumulate fitness until the pick is reached
    current = 0
    for ind in population:
        current += ind.fitness()
        if current > pick:
            return ind
    
    # If no individual is selected (due to floating-point precision), return the last one
    return population[-1]

def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        # population = [Individual.random_individual() if random.random() < 0.9
        #               else Individual.empty_individual()
        #               for _g in range(pop_limit)]
        population = [Individual.empty_individual()
                        for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population, selection_method="tournament", generation=generation)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")