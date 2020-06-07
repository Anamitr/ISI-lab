board_dict = {
    "0": ["*   g",
          " www ",
          " w*  ",
          " www ",
          "p    "],
    "1": ["wwwwwww",
          "w     w",
          "w www w",
          "w w*  w",
          "w www w",
          "wp    w",
          "wwwwwww"],
    "2": ["wwwwwww",  # - works
          "wp   *w",
          "wwwwwww"],
    "3": ["wwwwwww",
          "wp    w",
          "wwwww*w",
          "wwwwwww"],
    "4": ["wwwwwww",
          "wp    w",
          "ww w  w",
          "w   * w",
          "wwwwwww"],
    # frozenLake 4x4
    "5": ["p   ",  # - works
          " w w",
          "   w",
          "w  *"],
    # frozenLake 8x8
    "6": ["p       ",  # - works
          "        ",
          "   w    ",
          "     w  ",
          "   w    ",
          " ww   w ",
          " w  w w ",
          "   w   *"],
    "7": ["     ",  # - works 6m
          " www ",
          " w*  ",
          " www ",
          "p    "],
    "8": ["*    ",  # - record mean reward 33.6, episode 32
          " www ",
          " w*  ",
          " www ",
          "p    "],
    "10": ["wwwwwww",  # - works
           "wg p *w",
           "wwwwwww"],
    "11": ["g p  ",  # - works
           "wwww ",
           "ww*  "],
    "12": ["    *",  # max mean reward = 489, works very good
           " www ",
           "p   g"],
    "13": ["    g", # max mean reward = 483, works very good, ~15 min
           " www ",
           " w*  ",
           " www ",
           "p    "],
    "14": ["*   g", # max mean reward = 341, Total victory ratio: 788 / 1000, 300 epochs, ~1h
           " www ", # 909, 906, 912
           " w*  ",
           " www ",
           "p    "]
}

board = board_dict["14"]
