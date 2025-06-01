// src/bin/llm_agent.rs

use log::{error, info};
use remotro::{
    balatro::{
        menu::{Deck, Stake},
        play::{DiscardResult, HandCard, PlayResult},
        deck::{Card, Rank, Suit},
        Screen
    }, 
    Remotro
};
use tokio::time::{sleep, Duration};
use serde::{Serialize, Deserialize};
use serde_json::json;
use reqwest::Client;
use std::collections::HashMap;

// ===== Data Structures for LLM Communication =====

#[derive(Serialize, Debug)]
struct GameContext {
    hand: Vec<CardInfo>,
    hands_left: u8,
    discards_left: u8,
    current_score: f64,
    blind_target: f64,
    money: u32,
    ante: u32,
    phase: String,
}

#[derive(Serialize, Debug)]
struct CardInfo {
    index: usize,
    rank: String,
    suit: String,
    rank_value: u8,
}

#[derive(Deserialize, Debug)]
struct LLMResponse {
    function_call: FunctionCall,
    reasoning: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "name")]
enum FunctionCall {
    #[serde(rename = "analyze_hand")]
    AnalyzeHand { 
        hand_indices: Vec<usize> 
    },
    #[serde(rename = "evaluate_all_combinations")]
    EvaluateAllCombinations,
    #[serde(rename = "select_discard_cards")]
    SelectDiscardCards { 
        strategy: String,
        threshold_rank: Option<u8> 
    },
    #[serde(rename = "play_best_hand")]
    PlayBestHand,
    #[serde(rename = "skip_action")]
    SkipAction { 
        reason: String 
    },
}

// ===== Function Implementations =====

fn rank_to_value(rank: &Rank) -> u8 {
    match rank {
        Rank::Two => 0,
        Rank::Three => 1,
        Rank::Four => 2,
        Rank::Five => 3,
        Rank::Six => 4,
        Rank::Seven => 5,
        Rank::Eight => 6,
        Rank::Nine => 7,
        Rank::Ten => 8,
        Rank::Jack => 9,
        Rank::Queen => 10,
        Rank::King => 11,
        Rank::Ace => 12,
    }
}

fn get_chip_value(rank: &Rank) -> u32 {
    match rank {
        Rank::Two => 2,
        Rank::Three => 3,
        Rank::Four => 4,
        Rank::Five => 5,
        Rank::Six => 6,
        Rank::Seven => 7,
        Rank::Eight => 8,
        Rank::Nine => 9,
        Rank::Ten => 10,
        Rank::Jack => 10,
        Rank::Queen => 10,
        Rank::King => 10,
        Rank::Ace => 11,
    }
}

// Evaluate a specific 5-card hand
fn evaluate_hand(cards: &[&Card]) -> (u32, String) {
    if cards.len() != 5 {
        return (0, "Invalid hand size".to_string());
    }

    let mut chips = 0u32;
    let mut mult = 0u32;
    let mut hand_name = String::new();

    // Check for flush
    let flush = cards.iter().all(|c| c.suit == cards[0].suit);
    
    // Check for straight
    let mut sorted_ranks: Vec<u8> = cards.iter().map(|c| rank_to_value(&c.rank)).collect();
    sorted_ranks.sort();
    
    let mut straight = cards.len() == 5;
    if sorted_ranks != vec![0, 1, 2, 3, 12] {
        for i in 0..sorted_ranks.len() - 1 {
            if sorted_ranks[i] + 1 != sorted_ranks[i + 1] {
                straight = false;
                break;
            }
        }
    }

    // Count ranks
    let mut rank_counts: HashMap<&Rank, Vec<&Card>> = HashMap::new();
    for card in cards {
        rank_counts.entry(&card.rank).or_insert(Vec::new()).push(card);
    }

    let mut counts: Vec<(&Rank, Vec<&Card>)> = rank_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    
    let primary_hand = &counts.get(0).map(|(_, c)| c).unwrap_or(&vec![]);
    let secondary_hand = &counts.get(1).map(|(_, c)| c).unwrap_or(&vec![]);

    // Determine hand type
    if flush && primary_hand.len() == 5 {
        chips = 160; mult = 16; hand_name = "Flush Five".to_string();
    } else if flush && primary_hand.len() == 3 && secondary_hand.len() == 2 {
        chips = 140; mult = 14; hand_name = "Flush House".to_string();
    } else if primary_hand.len() == 5 {
        chips = 120; mult = 12; hand_name = "Five of a Kind".to_string();
    } else if straight && flush {
        chips = 100; mult = 8; hand_name = "Straight Flush".to_string();
    } else if primary_hand.len() == 4 {
        chips = 60; mult = 7; hand_name = "Four of a Kind".to_string();
    } else if primary_hand.len() == 3 && secondary_hand.len() == 2 {
        chips = 40; mult = 4; hand_name = "Full House".to_string();
    } else if flush {
        chips = 35; mult = 4; hand_name = "Flush".to_string();
    } else if straight {
        chips = 30; mult = 4; hand_name = "Straight".to_string();
    } else if primary_hand.len() == 3 {
        chips = 30; mult = 3; hand_name = "Three of a Kind".to_string();
    } else if primary_hand.len() == 2 && secondary_hand.len() == 2 {
        chips = 20; mult = 2; hand_name = "Two Pair".to_string();
    } else if primary_hand.len() == 2 {
        chips = 10; mult = 2; hand_name = "Pair".to_string();
    } else {
        chips = 5; mult = 1; hand_name = "High Card".to_string();
    }

    // Add scoring cards chip values
    let scoring_cards: Vec<&Card> = match hand_name.as_str() {
        "Four of a Kind" | "Three of a Kind" | "Pair" => primary_hand.clone(),
        "Two Pair" => {
            let mut combined = primary_hand.clone();
            combined.extend(secondary_hand);
            combined
        },
        "High Card" => {
            vec![*cards.iter().max_by_key(|c| rank_to_value(&c.rank)).unwrap()]
        },
        _ => cards.to_vec(),
    };

    for card in scoring_cards {
        chips += get_chip_value(&card.rank);
    }

    (chips * mult, hand_name)
}

// Evaluate all possible 5-card combinations
fn find_all_combinations(hand: &[HandCard]) -> Vec<(Vec<usize>, u32, String)> {
    let mut results = Vec::new();
    let n = hand.len();
    
    // Generate all C(n,5) combinations
    fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut combo = vec![0; k];
        
        fn helper(result: &mut Vec<Vec<usize>>, combo: &mut Vec<usize>, start: usize, idx: usize, n: usize, k: usize) {
            if idx == k {
                result.push(combo.clone());
                return;
            }
            
            for i in start..=n-k+idx {
                combo[idx] = i;
                helper(result, combo, i + 1, idx + 1, n, k);
            }
        }
        
        helper(&mut result, &mut combo, 0, 0, n, k);
        result
    }
    
    if n >= 5 {
        let combinations = generate_combinations(n, 5);
        
        for combo in combinations {
            let cards: Vec<&Card> = combo.iter()
                .map(|&idx| hand[idx].card())
                .collect();
            
            let (score, hand_type) = evaluate_hand(&cards);
            results.push((combo, score, hand_type));
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.cmp(&a.1));
    }
    
    results
}

// Select cards to discard based on strategy
fn select_discard_cards(hand: &[HandCard], strategy: &str, threshold: Option<u8>) -> Vec<u32> {
    match strategy {
        "threshold" => {
            let threshold = threshold.unwrap_or(8);
            let mut indices = Vec::new();
            
            for (i, hand_card) in hand.iter().enumerate() {
                if rank_to_value(&hand_card.card().rank) < threshold {
                    indices.push(i as u32);
                }
            }
            
            // Limit to 5 discards max
            if indices.len() > 5 {
                indices.truncate(5);
            }
            
            indices
        },
        "keep_pairs" => {
            // Keep pairs and high cards, discard others
            let mut rank_positions: HashMap<u8, Vec<usize>> = HashMap::new();
            
            for (i, hand_card) in hand.iter().enumerate() {
                let rank_val = rank_to_value(&hand_card.card().rank);
                rank_positions.entry(rank_val).or_insert(Vec::new()).push(i);
            }
            
            let mut keep_indices = Vec::new();
            
            // Keep all pairs/trips/quads
            for (_, positions) in rank_positions.iter() {
                if positions.len() >= 2 {
                    keep_indices.extend(positions);
                }
            }
            
            // Keep high cards if we don't have 5 cards yet
            let mut all_indices: Vec<(usize, u8)> = hand.iter().enumerate()
                .map(|(i, hc)| (i, rank_to_value(&hc.card().rank)))
                .collect();
            all_indices.sort_by(|a, b| b.1.cmp(&a.1));
            
            for (idx, _) in all_indices {
                if keep_indices.len() >= 5 {
                    break;
                }
                if !keep_indices.contains(&idx) {
                    keep_indices.push(idx);
                }
            }
            
            // Discard everything not in keep_indices
            let mut discard_indices = Vec::new();
            for i in 0..hand.len() {
                if !keep_indices.contains(&i) {
                    discard_indices.push(i as u32);
                }
            }
            
            discard_indices
        },
        _ => Vec::new(),
    }
}

// ===== LLM Integration =====

struct LLMAgent {
    client: Client,
    api_key: String,
    model: String,
}

impl LLMAgent {
    fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: "gpt-4".to_string(),
        }
    }

    async fn get_decision(&self, context: &GameContext) -> Result<LLMResponse, Box<dyn std::error::Error>> {
        let system_prompt = r#"You are an expert Balatro player with access to analysis functions.

Available functions:
1. analyze_hand(hand_indices) - Analyze specific cards
2. evaluate_all_combinations() - Find best 5-card combination from all possibilities
3. select_discard_cards(strategy, threshold_rank) - Choose cards to discard
   - Strategies: "threshold" (discard below rank), "keep_pairs" (keep pairs/high cards)
4. play_best_hand() - Play the optimal 5-card combination
5. skip_action(reason) - Skip current action

Decision guidelines:
- If discards_left > 0 and you have low cards (< 10), use select_discard_cards
- Always use evaluate_all_combinations before playing
- Consider blind_target vs current_score when deciding strategy
- Preserve high cards (10,J,Q,K,A) and pairs when possible

Respond with ONE function call in JSON:
{
  "function_call": {
    "name": "function_name",
    "parameter1": value1,
    ...
  },
  "reasoning": "Brief explanation"
}"#;

        let user_prompt = format!(
            "Game state:\n{}\n\nWhat should I do?",
            serde_json::to_string_pretty(context)?
        );

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "response_format": { "type": "json_object" }
            }))
            .send()
            .await?;

        let resp_json: serde_json::Value = response.json().await?;
        let content = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("No content in response")?;
        
        Ok(serde_json::from_str(content)?)
    }
}

// ===== Main Game Loop =====

async fn run_llm_agent() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");
    
    let agent = LLMAgent::new(api_key);
    
    let mut remotro = Remotro::host("127.0.0.1", 34143).await?;
    info!("LLM Agent ready on 127.0.0.1:34143");

    loop {
        let mut balatro = match remotro.accept().await {
            Ok(b) => {
                info!("Connected to Balatro!");
                b
            }
            Err(e) => {
                error!("Connection failed: {e}");
                continue;
            }
        };

        loop {
            let screen = match balatro.screen().await {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to get screen: {e}");
                    break;
                }
            };

            match screen {
                Screen::Menu(menu) => {
                    menu.new_run(Deck::Red, Stake::White, None).await?;
                }

                Screen::SelectBlind(blinds) => {
                    blinds.select().await?;
                }

                Screen::Play(mut play) => {
                    // Build context for LLM
                    let hand = play.hand();
                    let card_infos: Vec<CardInfo> = hand.iter().enumerate()
                        .map(|(i, hc)| {
                            let card = hc.card();
                            CardInfo {
                                index: i,
                                rank: format!("{:?}", card.rank),
                                suit: format!("{:?}", card.suit),
                                rank_value: rank_to_value(&card.rank),
                            }
                        })
                        .collect();

                    let context = GameContext {
                        hand: card_infos,
                        hands_left: *play.hands(),
                        discards_left: *play.discards(),
                        current_score: *play.score(),
                        blind_target: 300.0, // You'd get this from play.blind()
                        money: *play.money(),
                        ante: 1, // You'd track this
                        phase: if play.discards() > &0 { "discard".to_string() } else { "play".to_string() },
                    };

                    // Get LLM decision
                    match agent.get_decision(&context).await {
                        Ok(decision) => {
                            println!("\n🤖 LLM Decision: {:?}", decision.function_call);
                            println!("💭 Reasoning: {}", decision.reasoning);

                            match decision.function_call {
                                FunctionCall::AnalyzeHand { hand_indices } => {
                                    // Analyze specific cards
                                    let cards: Vec<&Card> = hand_indices.iter()
                                        .filter_map(|&i| hand.get(i).map(|hc| hc.card()))
                                        .collect();
                                    
                                    if cards.len() == 5 {
                                        let (score, hand_type) = evaluate_hand(&cards);
                                        println!("Analysis: {} - {} points", hand_type, score);
                                    }
                                }

                                FunctionCall::EvaluateAllCombinations => {
                                    let combinations = find_all_combinations(hand);
                                    println!("\n📊 Top 5 possible hands:");
                                    for (indices, score, hand_type) in combinations.iter().take(5) {
                                        println!("  {} - {} points (cards: {:?})", hand_type, score, indices);
                                    }
                                    
                                    // Play the best hand
                                    if let Some((best_indices, _, _)) = combinations.first() {
                                        let indices_u32: Vec<u32> = best_indices.iter().map(|&i| i as u32).collect();
                                        play = play.click(&indices_u32).await?;
                                        
                                        match play.play().await? {
                                            PlayResult::Again(next_play) => continue,
                                            PlayResult::RoundOver(overview) => {
                                                println!("Round complete! Earned: ${}", overview.total_earned());
                                                sleep(Duration::from_secs(2)).await;
                                                overview.cash_out().await?;
                                            }
                                            PlayResult::GameOver(_) => break,
                                        }
                                    }
                                }

                                FunctionCall::SelectDiscardCards { strategy, threshold_rank } => {
                                    let discard_indices = select_discard_cards(hand, &strategy, threshold_rank);
                                    
                                    if !discard_indices.is_empty() {
                                        println!("Discarding {} cards using {} strategy", discard_indices.len(), strategy);
                                        play = play.click(&discard_indices).await?;
                                        
                                        match play.discard().await? {
                                            DiscardResult::Again(next_play) => {
                                                play = next_play;
                                                continue;
                                            }
                                            DiscardResult::GameOver(_) => break,
                                        }
                                    }
                                }

                                FunctionCall::PlayBestHand => {
                                    let combinations = find_all_combinations(hand);
                                    
                                    if let Some((best_indices, score, hand_type)) = combinations.first() {
                                        println!("Playing best hand: {} - {} points", hand_type, score);
                                        let indices_u32: Vec<u32> = best_indices.iter().map(|&i| i as u32).collect();
                                        play = play.click(&indices_u32).await?;
                                        
                                        match play.play().await? {
                                            PlayResult::Again(next_play) => continue,
                                            PlayResult::RoundOver(overview) => {
                                                println!("Round complete! Earned: ${}", overview.total_earned());
                                                sleep(Duration::from_secs(2)).await;
                                                overview.cash_out().await?;
                                            }
                                            PlayResult::GameOver(_) => break,
                                        }
                                    }
                                }

                                FunctionCall::SkipAction { reason } => {
                                    println!("Skipping: {}", reason);
                                    // Continue to next screen update
                                }
                            }
                        }
                        Err(e) => {
                            error!("LLM decision failed: {e}");
                            // Fallback to basic strategy
                            if play.discards() > &0 {
                                let discard_indices = select_discard_cards(hand, "threshold", Some(8));
                                if !discard_indices.is_empty() {
                                    play = play.click(&discard_indices).await?;
                                    play.discard().await?;
                                }
                            } else {
                                let combinations = find_all_combinations(hand);
                                if let Some((best_indices, _, _)) = combinations.first() {
                                    let indices_u32: Vec<u32> = best_indices.iter().map(|&i| i as u32).collect();
                                    play = play.click(&indices_u32).await?;
                                    play.play().await?;
                                }
                            }
                        }
                    }
                }

                Screen::Shop(shop) => {
                    shop.leave().await?;
                }
            }
            
            sleep(Duration::from_millis(100)).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_llm_agent().await
}
