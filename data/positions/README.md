# Variation Positions

These files contain the Wall of Variations starting positions from Abalone Online.
Each `.fen` file is a single Abalone FEN record in the format used by
`engine/src/board.rs`:

```text
<9 row-compressed board rows> 0 0 <B|W> <white ejected> <black ejected>
```

The source order is the 60-image Wall of Variations page:
https://abaloneonline.wordpress.com/variations/wall-of-variations/

Filenames follow the image link slug from that page, so a source image like
`belgian-daisy.png` is stored as `belgian-daisy.fen`.

`snakes.fen` is the plain Snakes position, and `snakes-variation.fen` is the
Snakes variante position.
