import chess, chess.pgn, sys, os
from io import StringIO as S

# archaic names
P = {1: "pawne", 2: "Knight", 3: "Biſhop", 4: "Rooke", 5: "Queen", 6: "King"}
F = {
    i: n
    for i, n in enumerate(
        "Kings Rookes,Kings Knights,Kings Biſhops,Kings,Queens,Queens Biſhops,Queens Knights,Queens Rookes".split(
            ","
        )
    )
}
R = "firſt ſecond third fourth fifth ſixth ſeventh eighth".split()
H = "one two three four five ſix ſeven".split()
C = ("Black", "White")  # colour strings


def sq(q):  # absolute square description
    return F[chess.square_file(q)] + " " + R[chess.square_rank(q)] + " houſe"


def houses(n):  # relative distance
    return (H[n - 1] if n < 8 else str(n)) + " houſe" + ("s" * (n > 1))


def desc(b, m):  # move → text
    p = b.piece_at(m.from_square)
    col = C[p.color]
    f0, r0 = chess.square_file(m.from_square), chess.square_rank(m.from_square)
    f1, r1 = chess.square_file(m.to_square), chess.square_rank(m.to_square)
    fd, rd = abs(f1 - f0), abs(r1 - r0)

    if b.is_castling(m):
        return f"{col} {'Kings' if f1 > f0 else 'Queens'} Caſtle."

    if p.piece_type == 1:  # pawn
        s = f"{col} {F[f0]} {P[1]}"
        if b.is_capture(m):  # capture (incl. en passant)
            cap = b.piece_at(m.to_square)
            if not cap:  # en passant capture square holds none
                cap = b.piece_at(chess.square(f1, r0))
            s += f" takes {C[cap.color]} {P[cap.piece_type]}"
        s += " " + (houses(rd) if fd == 0 else "to " + sq(m.to_square))
        if m.promotion:
            s += f", becomes a {P[m.promotion]}"
    else:  # other pieces
        s = f"{col} {P[p.piece_type]}"
        if b.is_capture(m):
            cap = b.piece_at(m.to_square)
            s += f" takes {C[cap.color]} {P[cap.piece_type]}"
        elif fd and rd:
            s += " to " + sq(m.to_square)
        elif fd:
            s += f" {'right' if f1 > f0 else 'left'} " + houses(fd)
        else:
            s += f" {'up' if r1 > r0 else 'down'} " + houses(rd)

    b.push(m)
    s += (
        " gives Mate" if b.is_checkmate() else " gives Checke" if b.is_check() else ""
    ) + "."
    b.pop()
    return s


def convert(path):  # PGN → full antiquated text
    out, pgn = [], S(open(path).read())
    while g := chess.pgn.read_game(pgn):
        b = g.board()
        moves = [
            f"A game betwixt {g.headers.get('White','Unknown')} with ye White pieces "
            f"and {g.headers.get('Black','Unknown')} with ye Black pieces."
        ]
        for n in g.mainline():
            moves.append(desc(b, n.move))
            b.push(n.move)

        o = b.outcome()
        if o:
            term = o.termination.name.lower().replace("_", " ")
            if o.winner is None:
                moves.append(f"Game drawn by {term}")
            elif o.termination != chess.Termination.CHECKMATE:
                moves.append(f"{C[o.winner]} wins by {term}")

        moves.append("— Beale, The Royall Game of Cheſſe-Play")
        out.append("\n".join(moves))
    return "\n\n".join(out)


if __name__ == "__main__":
    p = sys.argv[1]
    txt = convert(p)
    out = f"royall_game_{os.path.basename(p)}.txt"
    open(out, "w").write(txt)
    print("\n".join(txt.split("\n")[:10]), "…", sep="\n")
