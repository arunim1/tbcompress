//! ≤ 5-piece random legal position generator (shakmaty 0.27).

use rand::{
    prelude::{SliceRandom, SmallRng},
    Rng,
};
use shakmaty::{
    bitboard::Bitboard, board::Board, CastlingMode, Color, FromSetup, Role, Setup,
};
use std::num::NonZeroU32;

pub fn random_board(rng: &mut SmallRng) -> shakmaty::Chess {
    loop {
        // —— place pieces ————————————————————————————————————————————
        let mut board = Board::empty();
        let mut sqs: Vec<_> = Bitboard::FULL.into_iter().collect();
        sqs.shuffle(rng);

        board.set_piece_at(sqs.pop().unwrap(), Role::King.of(Color::White));
        board.set_piece_at(sqs.pop().unwrap(), Role::King.of(Color::Black));

        const ROLES: &[Role] =
            &[Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
        for _ in 0..rng.gen_range(0..=3) {
            let role = *ROLES.choose(rng).unwrap();
            let col  = if rng.gen_bool(0.5) { Color::White } else { Color::Black };
            board.set_piece_at(sqs.pop().unwrap(), role.of(col));
        }

        // —— build Setup ————————————————————————————————————————————
        let stm = if rng.gen_bool(0.5) { Color::White } else { Color::Black };

        let setup = Setup {
            board,
            turn: stm,
            castling_rights: Bitboard::EMPTY,
            ep_square: None,
            fullmoves: NonZeroU32::new(1).unwrap(),
            halfmoves: 0,
            promoted: Bitboard::EMPTY,
            pockets: None,
            remaining_checks: None,
        };

        if let Ok(pos) = shakmaty::Chess::from_setup(setup, CastlingMode::Standard) {
            return pos;
        }
    }
}
