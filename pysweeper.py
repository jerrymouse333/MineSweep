#!/usr/bin/python
"""
Use machine learning to automatically play mine-sweeper games.
Contact: Hao Zhang <zhangh@cs.colostate.edu>

Use a part of Brett Smith's (<bcsmit1@engr.uky.edu>) code to first set up the game.
Minesweeper game
"""
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#
# Last revised: $Date: 2011/11/29 $

import math
import getopt
import os
import random
import sys
import traceback
import numpy as np
import itertools
import copy
import pdb

random.seed()

tmp_count = 0

def show_usage(error = None):
    """Show usage information, with an error if given, and exit appropriately.

    This function will display a standard, run-of-the-mill usage message,
    providing rudimentary help with the game's command-line switches.  If an
    error message is given, it will be printed preceding the rest of the
    message, and the program will exit with an exit code of 2; otherwise,
    the program will exit with a code of 0.
    """
    if error is not None:
        print "Error: %s." % error
    print "Usage: %s [-d] [-r,--rows ROWS]" % sys.argv[0],
    print "[-c,--cols,--columns COLUMNS] [-m,--mines MINES] [--dir DIRECTORY]"
    print "  -h,--help:           Display this usage message and exit."
    print "  -v,--version:        Display the program version and exit."
    print "  -r,--rows:           Set the number of rows in the playing field."
    print "  -c,--cols,--columns: Set the number of columns in the playing",
    print "field."
    print "  -m,--mines:          Set the number of mines in the playing",
    print "field."
    print "  --dir:               Add a directory for finding external data."
    print "  -d,--debug:          Enable debugging."
    if error is None:
        sys.exit(0)
    else:
        sys.exit(2)
        

def show_version():
    """Print version and copyright information, and exit normally."""
    print "Pysweeper 1.0 -- An implementation of the classic Minesweeper game."
    print "Copyright 2002 Brett Smith <bcsmit1@engr.uky.edu>"
    print ""
    print "This program is free software; you can redistribute it and/or modify"
    print "it under the terms of the GNU General Public License as published by"
    print "the Free Software Foundation; either version 2 of the License, or"
    print "(at your option) any later version."
    print ""
    print "This program is distributed in the hope that it will be useful, but"
    print "WITHOUT ANY WARRANTY; without even the implied warranty of"
    print "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU"
    print "General Public License for more details."
    sys.exit(0)


def fail(error, debug = 0):
    """Print an error message, with traceback if desired, and exit.

    This function will print an error message with the given body.  It will
    also print an exception traceback if debug is true.  It will then exit
    with an exit code of 1.

    error is the body of the error message.  If debug is a true value, a
    traceback will be printed.
    """
    print "Error: %s; exiting...." % error
    if debug:
        traceback.print_exc()
    sys.exit(1)


def set_option(options, name, value, minvalue = 0):
    """Set a value of a dictionary, with type and bounds checking.

    This function will set a given dictionary key to the given value, if it
    can be converted to an integer and is above a given value.  If any of
    the checks fail, the program will abort with an appropriate error
    message.

    options is the dictionary which contains the key to be set.  name is the
    name of the dictionary key to be set.  value is the value which will be
    checked and stored.  minvalue is the minimum value; the variable value
    must be larger than minvalue to be set.
    """
    try:
        value = int(value)
    except ValueError:
        show_usage("Bad argument (%s) for option %s" % (value, name))
    else:
        if value > minvalue:
            options[name] = value
        else:
            fail("bad value for option %s (too small)" % name)


def debug(objectlist):
    """Provide internal information about given objects.

    This function takes a list of objects, and then prints as much internal
    information about their state as possible.  Very much a dirty hack.

    objectlist is the list of objects to be debugged.
    """
    for object in objectlist:
        for variable in dir(object):
            attribute = getattr(object, variable)
            if not callable(attribute):
                print '%s: %s' % (variable, `attribute`)
        

def get_options():
    """Parse command-line options.

    This function reads the command-line options from sys.argv[1:] and
    returns a dict containing all configuration options with their given
    values.  It will abort the program if appropriate; for example, if
    an option has a bad argument, or a bad option is given.
    """
    game_opts = {'rows': 20, 'cols': 30, 'mines': 110, 'debug': 0}
    if os.name is 'posix':
        game_opts['paths'] = ['/usr/share/games/pysweeper',
                              '/usr/local/share/games/pysweeper', sys.path[0],
                              '.']
    else:
        game_opts['paths'] = [sys.path[0], '.']

    try:
        options = getopt.getopt(sys.argv[1:], 'hvdr:c:m:',
                                ['help', 'rows=', 'columns=', 'cols=', 'dir=',
                                 'mines=', 'version', 'debug'])[0]
    except getopt.error:
        show_usage(sys.exc_info()[1])

    set_size = 0
    set_mines = 0
    for option, argument in options:
        if option in ('-h', '--help'):
            show_usage()
        elif option in ('-v', '--version'):
            show_version()
        elif option in ('-d', '--debug'):
            game_opts['debug'] = 1
        elif option in ('-r', '--rows'):
            set_option(game_opts, 'rows', argument)
            set_size = 1
        elif option in ('-c', '--cols', '--columns'):
            set_option(game_opts, 'cols', argument)
            set_size = 1
        elif option in ('-m', '--mines'):
            set_option(game_opts, 'mines', argument)
            set_mines = 1
        elif option == '--dir':
            argument = os.path.normcase(argument)
            argument = os.path.normpath(argument)
            game_opts['paths'].insert(0, argument)
    if set_size and (not set_mines):
        game_opts['mines'] = int(round(game_opts['rows'] * game_opts['cols']
                                       * .15625))
    if game_opts['mines'] > (game_opts['rows'] * game_opts['cols']):
        show_usage("Too many mines (%i) for a %ix%i playing field" %
                   (game_opts['mines'], game_opts['rows'], game_opts['cols']))
    return game_opts




class Field:
    """Provide a playing field for a Minesweeper game.

    This class internally represents a Minesweeper playing field, and provides
    all functions necessary for the basic manipulations used in the game.
    """
    def __init__(self, rows = 16, cols = 16, mines = 40):
        """Initialize the playing field.

        This function creates a playing field of the given size, and randomly
        places mines within it.

        rows and cols are the numbers of rows and columns of the playing
        field, respectively.  mines is the number of mines to be placed within
        the field.
        """
        for var in (rows, cols, mines):
            if var < 0:
                raise ValueError, "all arguments must be > 0"
        if mines >= (rows * cols):
            raise ValueError, "mines must be < (rows * cols)"
                
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.cleared = 0
        self.flags = 0
        self.start_time = None

        minelist = []
        self.freecoords = {}
        for col in range(cols):
            self.freecoords[col] = range(rows)
        while mines > 0:
            y = random.choice(self.freecoords.keys())
            x = random.randrange(len(self.freecoords[y]))
            minelist.append((self.freecoords[y][x], y))
            del self.freecoords[y][x]
            if not self.freecoords[y]:
                del self.freecoords[y]
            mines = mines - 1

        self.board = []
        for col in range(cols):
            self.board.append([(-2, 0)] * rows)
            for row in range(rows):
                if (row, col) in minelist:
                    self.board[col][row] = (-1, 0)


    def _get_adjacent(self, x, y):
        """Provide a list of all tiles adjacent to the given tile.

        This function takes the x and y coordinates of a tile, and returns a
        list of a 2-tuples containing the coordinates of all adjacent tiles.

        x and y are the x and y coordinates of the base tile, respectively.
        """
        adjlist = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                   (x - 1, y), (x + 1, y),
                   (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
        rmlist = []
        for adjpair in adjlist:
            for value, index in [(-1, 0), (-1, 1), (self.cols, 0),
                                 (self.rows, 1)]:
                if adjpair[index] == value:
                    rmlist.append(adjpair)
                    break
        for item in rmlist:
            adjlist.remove(item)
        return adjlist


    def open_cell(self, coordlist, y = None):
        """Open one or more tiles.

        This function opens all the tiles provided, and others as
        appropriate, and returns a list indicating which tiles were opened
        and the result of the open.  If a tile is opened which has no
        adjacent mines, all adjacent tiles will be opened automatically.

        The function returns a list of 2-tuples.  The first value is a
        2-tuple with the x and y coordinates of the opened tile; the second
        value indicates the number of mines adjacent to the tile, or -1 if
        the tile contains a mine.
        e.g. return [((x,y),num_mines/-1), ((),), ...]

        If a value for y is given, the tile at (coordlist, y) will be
        opened; otherwise, the function will open the tiles whose
        coordinates are given in 2-tuples in coordlist.
        """
        if y is not None:
            coordlist = [(coordlist, y)]
        opened = []
        while len(coordlist) != 0:
            x, y = coordlist.pop()
            not_done = 1
            if (self.board[x][y][1] == 1) or (self.board[x][y][0] >= 0):
                not_done = 0
            elif self.board[x][y][0] == -1:
                # when it's a mine.
                if self.cleared > 0:
                    self.board[x][y] = (-1, -1)
                    opened.append(((x, y), -1))
                    not_done = 0
                else:
                    while self.board[x][y][0] == -1:
                        # The first opened block is a mine; move it elsewhere.
                        newx = random.choice(self.freecoords.keys())
                        newy = random.randrange(len(self.freecoords[newx]))
                        self.board[x][y] = (-2, 0)
                        self.board[newx][newy] = (-1,
                                                  self.board[newx][newy][1])
            if not_done:
                adjlist = self._get_adjacent(x, y)
                adjcount = 0
                for adjx, adjy in adjlist:
                    if self.board[adjx][adjy][0] == -1:
                        adjcount = adjcount + 1
                self.board[x][y] = (adjcount, -1)
                if self.cleared is 0:
                    del self.freecoords
                self.cleared = self.cleared + 1
                opened.append(((x, y), adjcount))
                if adjcount == 0:
                    coordlist.extend(adjlist)
        return opened


    def open_adjacent(self, x, y):
        """Open all unflagged tiles adjacent to the given one, if appropriate.

        This function counts the number of tiles adjacent to the given one
        which are flagged.  If that count matches the number of mines adjacent
        to the tile, all unflagged adjacent tiles are opened, using
        Field.open_cell().

        x and y are the x and y coordinates of the tile to be flagged,
        respectively.
        """
        adjmines = self.board[x][y][0]
        if self.board[x][y][1] != -1:
            return []
        adjlist = self._get_adjacent(x, y)
        flagcount = 0
        for adjx, adjy in adjlist:
            if self.board[adjx][adjy][1] == 1:
                flagcount = flagcount + 1
        if adjmines == flagcount:
            return self.open_cell(adjlist)
        else:
            return []
        

    def flag(self, x, y):
        """Flag or unflag an unopened tile.

        This function attempts to toggle the flagged status of the tile at
        the given coordinates, and returns a value indicating the action
        which occurred.  A return value of -1 indicates that the tile is
        opened and cannot be flagged; 0 indicates that the tile is unflagged;
        and 1 indicates that the tile was flagged.

        x and y are the x and y coordinates of the tile to be flagged,
        respectively.
        """
        if self.board[x][y][1] == -1:
            return -1 #the tile is opened and cannot be flagged
        elif self.board[x][y][1] == 0:
            self.board[x][y] = (self.board[x][y][0], 1)
            self.flags = self.flags + 1
            return 1 #the tile was flagged
#        else:
#            self.board[x][y] = (self.board[x][y][0], 0)
#            self.flags = self.flags - 1
#            return 0 #the tile was unflagged


    def get_diff(self):
        """Return a list providing mine locations.

        This function provides a list of 2-tuples.  The first value of each
        2-tuple is a 2-tuple, providing the x and y coordinates of a tile;
        the second value of the 2-tuple is either 1 or -1.  1 indicates that
        a mine is at those coordinates; -1 indicates that a mine is not at
        those coordinates, but a flag was placed there.
        """
        diff = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[x][y] == (-2, 1):
                    diff.extend([((x, y), -1)])
                elif ((self.board[x][y][0] == -1) and
                      (self.board[x][y][1] == 0)):
                    diff.extend([((x, y), 1)])
        return diff


    def playtime(self):
        """Return a string representing the current play time.

        This function returns a string which provides a human-readable
        representation of the amount of time the current game has been
        played, starting when the first tile is opened.  If the player
        takes an inordinate amount of time (9999 minutes, 0 seconds -- or
        longer), the returned string will be '9999:00+'.
        """
        if self.start_time is None:
            return '00:00'
        rawtime = int(time.time() - self.start_time)
        mins = int(math.floor(rawtime / 60.0))
        secs = rawtime % 60
        if mins > 9998:
            return '9999:00+'
        elif (mins < 10) and (secs < 10):
            return '0%i:0%i' % (mins, secs)
        elif mins < 10:
            return '0%i:%i' % (mins, secs)
        elif secs < 10:
            return '%i:0%i' % (mins, secs)
        else:
            return '%i:%i' % (mins, secs)


    def won(self):
        """Indicate whether or not the game has been won.

        This function will return a true value if the game has been won, and
        a false value otherwise.
        """
        return ((self.flags == self.mines) and
                (self.cleared == (self.rows * self.cols) - self.mines))
    
    def display_board(self):
        for r in range(self.rows):
            print ''
            for c in range(self.cols):
                if self.board[c][r][1] == 0:# unknown cell
                    print '%2s' %' #',
                elif self.board[c][r][1] == -1:
                    if self.board[c][r][0]>=0:
                        print '%2d' %(self.board[c][r][0]),# a successful open
                    else:
                        print '%2s' %' *',# a mine is opened
                elif self.board[c][r][1] == 1:# flag
                    print '%2s' %' F',

class Player:
    def __init__(self, board, row, column, mine):
        self.board = board
        self.rows = row
        self.cols = column
        self.mines = mine
        self.cboard = self.convert_board()

    def update_board(self, board):
        self.board = board
        self.cboard = self.convert_board()

    def move(self):
        moves = []
        
        for r in range(self.rows):
            for c in range(self.cols):
                if (ord(self.cboard[r,c])>=49) & (ord(self.cboard[r,c])<=56):
                    FC, UC = self.count_neighbors(r,c)
#                    print FC, UC
                    if (UC > 0) & (FC == int(self.cboard[r,c])):
                        #Situation 2: n=FC
                        moves.append(('sweep',(r,c)))
                        print '\n',moves
                        return moves
                    if (UC > 0) & (FC + UC == int(self.cboard[r,c])):
                        #Situation 1: n=FC+UC, where FC is the Flag count and UC is the Unexplored count
                        unopened_coords = self.get_unopened_cells(r,c)
                        for i,j in unopened_coords:
                            moves.append(('flag',(i,j)))
                        print '\n',moves
                        return moves
        #Situation 3: find other available moves
        moves = self.findAvailMoves()
        if moves != []:
            global tmp_count
            tmp_count += 1
            print '\n',moves
            return moves
        #Situation 4: random move based on least probability of losing the game
#        prob = float((self.mines-sum(sum(self.cboard=='F')))) / float(sum(sum(self.cboard=='#')))
        prob_mat = self.comp_prob()
        prob = 1
        nx, ny = -1, -1
        for r in range(self.rows):
            for c in range(self.cols):
                if (self.cboard[r,c]=='#'):
                    tmp_prob = 0
                    for i in range(r-1, r+2):
                        if i < 0 or i > self.rows-1: continue
                        for j in range(c-1, c+2):
                            if j < 0 or j > self.cols-1: continue
                            if i==r and j==c: continue
                            if prob_mat[i,j]>tmp_prob:
                                tmp_prob = prob_mat[i,j]

                    if tmp_prob <= prob:
                        if tmp_prob == prob:
                            if np.random.uniform() < 0.01:
                                nx, ny = r, c
                        else:
                            prob = tmp_prob
                            nx, ny = r, c
        print '\n',prob
        if prob <= 0.25 and prob != 0:
            print 'random open', nx, ny
#        if prob <= 0.9:
            return [('open', (nx,ny))]
        #Situation 5: enumerate all the possible situations when the number of remaining mine is small(<20)
        if sum(sum(self.cboard=='#')) < 20:
            moves = self.enum_all()
            if moves != []:
                print '\n',moves
                return moves
        return self.get_input()

    def enum_all(self):
        moves = []
        m = self.mines-sum(sum(self.cboard=='F')) # number of remaining mines
        n = sum(sum(self.cboard=='#')) # number of remaining cells
        state = []
        for a in itertools.combinations(range(n),m):
            state.append(a)
        i,j = np.where(self.cboard=='#')
        n_index = zip(i,j) # coordinates of the cells: a list of 2-tuples
        all_moves = []
        for elem in state:
            board = copy.copy(self.cboard)
            tmp_mark = ['0']*(n)
            for i in elem:
                board[n_index[i][0], n_index[i][1]]='F'
                tmp_mark[i]='1'
            #check if the board is valid
            fail=0
            for i in range(self.rows):
                if fail==1:
                    break
                for j in range(self.cols):
                    if board[i,j]=='#' or board[i,j]=='F':
                        continue
                    fc = self.count_flags(board, i, j)
                    if fc != int(board[i,j]):
                        fail=1
                        break
            if fail==0:
                all_moves.append(tmp_mark)
        if all_moves != []:
            current_move = all_moves[0]
            for i in all_moves:
                if current_move == None:
                    break
                current_move = self.common_elem(current_move, i)
            if current_move != None:
                for i in range(len(current_move)):
                    if current_move[i] == '1':
                        moves.append(('flag',n_index[i]))
                    elif current_move[i] == '0':
                        moves.append(('open',n_index[i]))
        return moves
        

    def count_flags(self, board, r, c):
        rows = self.rows
        cols = self.cols
        flagCount = 0 # count the cells which are marked as Flagged
        if (r>=1):
            flagCount += (board[r-1,c]=='F')
            if (c>=1):
                flagCount += (board[r-1,c-1]=='F')
            if (c<=cols-2):
                flagCount += (board[r-1,c+1]=='F')
        if (r<=rows-2):
            flagCount += (board[r+1,c]=='F')
            if (c>=1):
                flagCount += (board[r+1,c-1]=='F')
            if (c<=cols-2):   
                flagCount += (board[r+1,c+1]=='F')
        if (c>=1):
            flagCount += (board[r,c-1]=='F')
        if (c<=cols-2):
            flagCount += (board[r,c+1]=='F')
        return flagCount           


    def comp_prob(self):
        '''
        compute the probability table of losing the game
        '''
        p = np.ones((self.rows, self.cols))
        p0 = float((self.mines-sum(sum(self.cboard=='F')))) / float(sum(sum(self.cboard=='#')))
        for r in range(self.rows):
            for c in range(self.cols):
                if (ord(self.cboard[r,c])>=49) & (ord(self.cboard[r,c])<=56):
                    FC, UC = self.count_neighbors(r,c)
                    if UC == 0:
                        continue
                    p[r,c] = float((int(self.cboard[r,c])-FC)) / float(UC)
                elif self.cboard[r,c]=='#':
                    p[r,c] = p0
                elif self.cboard[r,c]=='F':
                    p[r,c] = 0
        return p
                    


    def findAvailMoves(self):
        rows = self.rows
        cols = self.cols
        moves = []
        for r in range(rows-2): #search through the whole board
            for c in range(cols-2):
                innerRegion = (r, r+2, c, c+2) #(low_row, high_row, low_col, high_col)
#                outerRegion = (r-(r!=0), r+2+(r!=rows-3), c-(c!=0), c+2+(c!=cols-3))
                unCells = self.get_unopened_cells(r+1, c+1) # unopened cells in the inner region
                if len(unCells)>4:
                    continue
                k = 0
                clusters = {}
                for i in range(innerRegion[0], innerRegion[1]+1):
                    for j in range(innerRegion[2], innerRegion[3]+1):
                        if ((i,j) in unCells) or (ord(self.cboard[i,j])==70): 
                            continue
                        fc, uc = self.count_neighbors(i,j)
                        xy = self.get_unopened_cells(i,j) # unopened cells around it
                        N = int(self.cboard[i,j]) - fc # effective number of mines around this cell
                        x = [xx for xx in unCells if xx in xy] # unopened cells in the inner region
                        if x == []:
                            continue
                        y = list(set(xy)-set(x)) # unopened cells outside of the inner region
                        mark = [0]*len(xy)
                        for tmpi in range(len(xy)):
                            if xy[tmpi] in x:
                                mark[tmpi]=1 # if the unopened cell is in the inner region, mark=1
                        minMine, maxMine = max(N-len(y), 0), min(N, len(x))
                        ind = self.list_all(minMine, maxMine, len(xy))
                        cluster_members = []
                        for ind0 in ind:
                            code = ['U']*len(unCells) # the binary code for unCells, 1 for mine, 0 for non-mine, U for unknown
                            unused_mark = [i0 for i0, j0 in enumerate(mark) if j0==1]
                            for ind1 in ind0:
#                                pdb.set_trace()
                                if mark[ind1] == 1:
                                    unused_mark.remove(ind1)
                                    code[unCells.index(xy[ind1])]='1'
                            for unused_ind in unused_mark:
#                                print '\n',code, unCells, x, unused_ind
                                code[unCells.index(xy[unused_ind])]='0'
                            cluster_members.append(code)
                        k += 1
                        clusters[k] = cluster_members

#----------------------go through a list of filters----------------------------
                all_moves = clusters.get(1, [])
                level = 2
                while level <= k:
                    if all_moves == []:
                        break
                    new_moves = []
                    for i in all_moves:
                        for j in clusters[level]:
                            intersect_tmp = self.intersect(i,j)
                            if intersect_tmp != None:
                                new_moves.append(intersect_tmp)
                    all_moves = copy.copy(new_moves)
                    level += 1

#---------------------combine the final result-------------------------
                if all_moves != []:
                    current_move = all_moves[0]
                    for i in all_moves:
                        if current_move == None:
                            break
                        current_move = self.common_elem(current_move, i)
                    if current_move != None:
                        for i in range(len(current_move)):
                            if current_move[i] == '1':
                                moves.append(('flag',unCells[i]))
                            elif current_move[i] == '0':
                                moves.append(('open',unCells[i]))
        return moves
                        
                
    def list_all(self, minMine, maxMine, n):
        state = []
        for i in range(minMine, maxMine+1):
            for a in itertools.combinations(range(n),i):
                state.append(a)
        return state
                
    def intersect(self, code1, code2):
        code = ['U']*len(code1)
        for i in range(len(code1)):
            if code1[i]=='U':
                code[i] = code2[i]
            elif code2[i]=='U':
                code[i] = code1[i]
            elif code1[i]!=code2[i]:
                return None
            else:
                code[i] = code1[i]
        return code

    def common_elem(self, code1, code2):
        code = ['2']*len(code1)
        for i in range(len(code1)):
            if (code1[i]!='U') and (code1[i]==code2[i]):
                code[i] = code2[i]
        if code != ['2']*len(code1):
            return code
        else:
            return None

    def convert_board(self):
        cboard = np.zeros((self.rows,self.cols), dtype='c')
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[c][r][1] == 0:# unknown cell
                    cboard[r,c] = '#'
                elif self.board[c][r][1] == -1:
                    if self.board[c][r][0]>=0:
                        cboard[r,c] = str(self.board[c][r][0])# a successful open
                    else:
                        cboard[r,c] = '*'# a mine is opened
                elif self.board[c][r][1] == 1:# flag
                    cboard[r,c] = 'F'
        return cboard

    def count_neighbors(self, r, c):
        rows = self.rows
        cols = self.cols
        flagCount = 0 # count the cells which are marked as Flagged
        unopenedCount = 0 # count the cells which are unexplored
        if (r>=1):
            flagCount += (self.cboard[r-1,c]=='F')
            unopenedCount += (self.cboard[r-1,c]=='#')
            if (c>=1):
                flagCount += (self.cboard[r-1,c-1]=='F')
                unopenedCount += (self.cboard[r-1,c-1]=='#')
            if (c<=cols-2):
                flagCount += (self.cboard[r-1,c+1]=='F')
                unopenedCount += (self.cboard[r-1,c+1]=='#')
        if (r<=rows-2):
            flagCount += (self.cboard[r+1,c]=='F')
            unopenedCount += (self.cboard[r+1,c]=='#')
            if (c>=1):
                flagCount += (self.cboard[r+1,c-1]=='F')
                unopenedCount += (self.cboard[r+1,c-1]=='#')
            if (c<=cols-2):   
                flagCount += (self.cboard[r+1,c+1]=='F')
                unopenedCount += (self.cboard[r+1,c+1]=='#')
        if (c>=1):
            flagCount += (self.cboard[r,c-1]=='F')
            unopenedCount += (self.cboard[r,c-1]=='#')
        if (c<=cols-2):
            flagCount += (self.cboard[r,c+1]=='F')
            unopenedCount += (self.cboard[r,c+1]=='#')
        return flagCount, unopenedCount           

    def get_unopened_cells(self, r, c):
        rows = self.rows
        cols = self.cols
        unopened_coords = [] # a list of two-tuples
        if self.cboard[r,c]=='#':
            unopened_coords.append((r,c)) 
        if (r>=1):
            if (self.cboard[r-1,c]=='#'): unopened_coords.append((r-1,c)) 
            if (c>=1):
                if (self.cboard[r-1,c-1]=='#'): unopened_coords.append((r-1,c-1)) 
            if (c<=cols-2):
                if (self.cboard[r-1,c+1]=='#'): unopened_coords.append((r-1,c+1)) 
        if (r<=rows-2):
            if (self.cboard[r+1,c]=='#'): unopened_coords.append((r+1,c)) 
            if (c>=1):
                if (self.cboard[r+1,c-1]=='#'): unopened_coords.append((r+1,c-1)) 
            if (c<=cols-2):   
                if (self.cboard[r+1,c+1]=='#'): unopened_coords.append((r+1,c+1)) 
        if (c>=1):
            if (self.cboard[r,c-1]=='#'): unopened_coords.append((r,c-1)) 
        if (c<=cols-2):
            if (self.cboard[r,c+1]=='#'): unopened_coords.append((r,c+1)) 
        return unopened_coords
        


    def get_input(self):
        result = []
        print '\nCount =', tmp_count
        print "\nRemaining mines:",(self.mines-sum(sum(self.cboard=='F')))
        print '\nRemaining cells:',(sum(sum(self.cboard=='#')))
        var = raw_input("\nEnter Next Move: ")
        if var == 'quit':
            return [(var, None)]
        if var == 'reset':
            return [(var, None)]
        act, num1, num2 = var.split()
        num1, num2 = int(num1), int(num2)
        result.append((act, (num1, num2)))
        return result
    

def run_game(game_opts):
    """Run the game with the given options and interface.

    This function runs the main game loop with the given options and
    interface.  It exits only when the player quits.

    game_opts is a dictionary of game options, as provided by get_options().
    ui is the interface to use for the game.
    """
    no_quit = 1

    while no_quit:
        field = Field(game_opts['rows'], game_opts['cols'], game_opts['mines'])
        player = Player(field.board, field.rows, field.cols, field.mines)
        field.display_board()
        no_reset = 1
        while no_reset:
            in_put = player.move()     
            output = []
            # This loop generates the list of commands for ui.update().
            # It is a list of 2-tuples.  The first value of each tuple is a
            # string, indicating the type of output; the second value is an
            # n-sequence of parameters, which varies from one action to
            # another.
            #
            # This loop also takes other internal actions as necessary
            # given the input -- for example, prints debugging information.
            for act, pos in in_put:
                if act == 'quit':
                    print "Quit the game!"
                    return 0
                if act == 'reset':
                    no_reset = 0
                    break
                pos = (pos[1], pos[0])
                if act in ('open', 'sweep'):
                    if act == 'open':
                        opened = field.open_cell(pos[0], pos[1])
                    else:
                        opened = field.open_adjacent(pos[0], pos[1])

                    for result in opened:
                        if result[1] == -1:
                            field.display_board()
                            print "\nSorry, you lost!"
                            return 0
                elif act == 'flag':
                    result = field.flag(pos[0], pos[1])
            field.display_board()
            player.update_board(field.board)
            if field.won()==1 or (field.flags + sum(sum(player.cboard=='#'))==field.mines):               
                print "\nCongratulations! You won!"
                return 0

if __name__ == '__main__':
    game_opts = get_options()
    run_game(game_opts)
