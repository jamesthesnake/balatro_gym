use serde::{Deserialize, Serialize};
use crate::balatro_enum;
use crate::balatro::play::HandCard;


#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Card {
   // pub edition: Option<Edition>,
  //  pub enhancement: Option<Enhancement>,
    pub rank: Rank,
    pub suit: Suit,
   // pub seal: Option<Seal>
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum Suit {
    Spades,
    Hearts,
    Clubs,
    Diamonds
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum Rank {
    #[serde(rename = "Ace")]
    Ace,
    #[serde(rename = "2")]
    Two,
    #[serde(rename = "3")]
    Three,
    #[serde(rename = "4")]
    Four,
    #[serde(rename = "5")]
    Five,
    #[serde(rename = "6")]
    Six,
    #[serde(rename = "7")]
    Seven,
    #[serde(rename = "8")]
    Eight,
    #[serde(rename = "9")]
    Nine,
    #[serde(rename = "10")]
    Ten,
    Jack,
    Queen,
    King
}

balatro_enum!(Enhancement {
    Wild = "m_wild",
    Glass = "m_glass",
    Bonus = "m_bonus",
    Mult = "m_mult",
    Lucky = "m_lucky",
    Steel = "m_steel",
    Stone = "m_stone",
    Gold = "m_gold"
});

balatro_enum!(Seal {
    Blue = "blue_seal",
    Red = "red_seal",
    Purple = "purple_seal",
    Gold = "gold_seal"
});

balatro_enum!(Edition {
    None = "e_base",
    Foil = "e_foil",
    Holographic = "e_holo",
    Polychrome = "e_polychrome"
});
impl Card {
    /// Return a *static* 52-card deck in fixed order
    /// (Spades→Clubs→Diamonds→Hearts, 2→A inside each suit).
    pub fn full_deck() -> [Card; 52] {
        use Rank::*;
        use Suit::*;

        const RANKS: [Rank; 13] = [
            Two, Three, Four, Five, Six, Seven, Eight,
            Nine, Ten, Jack, Queen, King, Ace,
        ];
        const SUITS: [Suit; 4] = [Spades, Clubs, Diamonds, Hearts];

        let mut deck = [Card { rank: Two, suit: Spades }; 52];
        let mut k = 0;
        for &s in &SUITS {
            for &r in &RANKS {
                deck[k] = Card { rank: r, suit: s };
                k += 1;
            }
        }
        deck
    }
}

impl HandCard {
    pub fn card(&self) -> &Card {
        self.card
            .as_ref()
            .expect("HandCard.card was None when we expected a Card")
    }
}
