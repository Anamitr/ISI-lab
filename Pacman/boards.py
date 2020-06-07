board0 = ["*   g",
         " www ",
         " w*  ",
         " www ",
         "p    "]

board1 = ["wwwwwww",
         "w     w",
         "w www w",
         "w w*  w",
         "w www w",
         "wp    w",
         "wwwwwww"]

board2 = ["wwwwwww",   # - works
         "wp   *w",
         "wwwwwww"]

board3 = ["wwwwwww",
         "wp    w",
         "wwwww*w",
         "wwwwwww"]

board4 = ["wwwwwww",
         "wp    w",
         "ww w  w",
         "w   * w",
         "wwwwwww"]
# frozenLake 4x4
board5 = [             # - works
        "p   ",
        " w w",
        "   w",
        "w  *"
    ]
# frozenLake 8x8
board6 = [             # - works
        "p       ",
        "        ",
        "   w    ",
        "     w  ",
        "   w    ",
        " ww   w ",
        " w  w w ",
        "   w   *"
    ]
board7 = ["     ",       # - works 6m
         " www ",
         " w*  ",
         " www ",
         "p    "]
board8 = ["*    ",       # - record mean reward 33.6, episode 32
         " www ",
         " w*  ",
         " www ",
         "p    "]
board9 = ["*    ",
         " www ",
         " w*  ",
         " www ",
         "p    "]
board10 = ["wwwwwww",   # - works
           "wg p *w",
           "wwwwwww"]
board11 = ["g p  ",     # - works
           "wwww ",
           "ww*  "]
board12 = ["    *",     # max mean reward = 489, works good
           " www ",
           "p   g"]


board = board12