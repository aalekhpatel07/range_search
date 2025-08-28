use fst::Automaton;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Cost {
    L2(f32),
    Hamming(i64),
}

impl Cost {
    pub fn is_non_negative(&self) -> bool {
        match self {
            Cost::Hamming(d) => *d >= 0,
            Cost::L2(d) => d.is_sign_positive(),
        }
    }

    pub fn step(&self, from: u8, to: u8) -> Cost {
        match self {
            Cost::Hamming(remaining) => {
                let step_by = i64::from((from ^ to).count_ones());
                Cost::Hamming((*remaining).saturating_sub(step_by))
            }
            Cost::L2(remaining) => {
                let from = from as f32;
                let to = to as f32;
                let step_by = (from - to) * (from - to);
                Cost::L2(*remaining - step_by)
            }
        }
    }
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cost::Hamming(v) => {
                write!(f, "{}", *v)
            }
            Cost::L2(v) => {
                write!(f, "{:.3}", *v)
            }
        }
    }
}

#[derive(Debug)]
pub struct RangeSearch<'a, const N: usize> {
    query: &'a [u8],
    max_distance: Cost,
}

impl<'a, const N: usize> RangeSearch<'a, N> {
    pub fn new(query: &'a [u8], max_distance: Cost) -> Self {
        Self {
            query,
            max_distance,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangeSearchState {
    budget: Cost,
    position: usize,
    query_is_bad_size: bool,
}

impl std::fmt::Display for RangeSearchState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(budget: {}, position: {})", self.budget, self.position)
    }
}

impl<const N: usize> Automaton for RangeSearch<'_, N> {
    type State = RangeSearchState;

    fn start(&self) -> Self::State {
        RangeSearchState {
            budget: self.max_distance,
            position: 0,
            query_is_bad_size: self.query.len() != { N },
        }
    }
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        RangeSearchState {
            // try to take a step, consuming some of our budget.
            budget: state.budget.step(self.query[state.position], byte),
            position: state.position + 1,
            query_is_bad_size: false,
        }
    }

    fn is_match(&self, state: &Self::State) -> bool {
        if state.query_is_bad_size {
            return false;
        }
        state.position == { N } && state.budget.is_non_negative()
    }

    fn can_match(&self, state: &Self::State) -> bool {
        // we already know the query vector is of an incompatible
        // size.
        if state.query_is_bad_size {
            return false;
        }
        // our scanned vector is larger than what we're expecting.
        if state.position > { N } {
            return false;
        }
        // can only match if there is any budget left at all.
        state.budget.is_non_negative()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use fst::{IntoStreamer, MapBuilder, Streamer};
    use fst::set::SetBuilder;
    use rand::{prelude::*, rng};
    use arbitrary::{Arbitrary, Unstructured};

    fn generate_data<const SIZE: usize>(count: usize) -> Vec<[u8; SIZE]> {
        let mut rng = rng();
        let mut buffer=  [0u8; 1024];
        let mut vectors = vec![];
        for _ in 0..count {
            rng.fill_bytes(&mut buffer);
            let data: [u8; SIZE] = Arbitrary::arbitrary(&mut Unstructured::new(&buffer)).unwrap();
            vectors.push(data);
        }
        vectors
    }

    #[test]
    fn automaton_works() {
        let mut builder = SetBuilder::memory();
        let mut data = vec![[0u8, 255, 255, 255], [255, 255, 255, 255], [1, 1, 1, 1]];
        data.sort();

        for vector in data {
            builder.insert(vector).unwrap();
        }

        let set = builder.into_set();

        let query = [0, 0, 0, 0];

        let aut = RangeSearch::<4>::new(&query, Cost::Hamming(5));
        let stream = set.search(aut).into_stream();
        let observed_matches = stream.into_bytes();
        assert_eq!(observed_matches, vec![vec![1, 1, 1, 1]]);
    }

    pub fn naive_distance_l2(v1: &[u8], v2: &[u8]) -> f32 {
        let mut dist: f32 = 0.;
        for (&b1, &b2) in v1.iter().zip(v2) {
            dist += ((b1 as f32) - (b2 as f32)) * ((b1 as f32) - (b2 as f32))
        }
        dist
    }

    #[test]
    fn automaton_correctness_l2() {
        // generate 1M vectors.
        let mut data: Vec<[u8; 144]> = generate_data::<144>(1_000_000);
        data.sort();
        eprintln!("finished sorting...");
        let data_with_ids: Vec<_> = data.into_iter().enumerate().collect();

        let mut builder = MapBuilder::memory();
        for (idx, vector) in data_with_ids.iter() {
            builder.insert(*vector, *idx as u64).unwrap();
        }
        let map = builder.into_map();
        eprintln!("finished generating map. begin searches.");
        let query = generate_data::<144>(1)[0];

        let max_distance: f32 = 1_500_000.0;
        let aut = RangeSearch::<144>::new(&query, Cost::L2(max_distance));
        let mut stream = map.search(aut).into_stream();
        let mut hit_indices = HashSet::new();

        while let Some((key, idx)) = stream.next() {
            let distance=  naive_distance_l2(&query, key);
            assert!(distance <= max_distance);
            hit_indices.insert(idx);
        }
        eprintln!("hits: {}", hit_indices.len());
        let non_hits: Vec<_> = 
            data_with_ids
            .iter()
            .filter_map(|(idx, vector)| {
                match hit_indices.contains(&(*idx as u64)) {
                    false => Some(*vector),
                    true => None
                }
            })
            .collect();
    
        eprintln!("non hits: {}", non_hits.len());
        for non_hit in non_hits {
            let distance=  naive_distance_l2(&query, &non_hit);
            assert!(distance > max_distance);
        }
    }

    #[test]
    fn get_single_hit() {
        let count: usize = 100_000;
        const SIZE: usize = 144;
        // generate vectors.
        let mut data: Vec<[u8; SIZE]> = generate_data::<SIZE>(count);
        eprintln!("generated {count} vectors of dimension {SIZE}...");
        data.sort();
        eprintln!("sorted {count} vectors of dimension {SIZE}...");

        let mut builder = SetBuilder::memory();
        for vector in data.iter() {
            builder.insert(*vector).unwrap();
        }
        let set = builder.into_set();
        eprintln!("finished generating set. begin searches.");
        
        let num_searches: usize = 1_000;
        // generate search queries.
        let queries: Vec<[u8; SIZE]> = generate_data::<SIZE>(num_searches);
        eprintln!("generated {num_searches} query vectors of dimension {SIZE}...");

        let max_distance: f32 = 1_000_000.0;
        let mut hits = 0;
        let mut search_times = Vec::with_capacity(num_searches);
        let mut seen = 0;
        for query in queries {
            let aut = RangeSearch::<144>::new(&query, Cost::L2(max_distance));
            let mut stream = set.search_with_state(aut).into_stream();
            let search_time = std::time::Instant::now();
            if let Some((_hit, state)) = stream.next() {
                eprintln!("hit state: {state}");
                // found some hit.
                hits += 1;
            }
            seen += 1;
            let elapsed = search_time.elapsed();
            eprintln!("seen so far: {}", seen);
            search_times.push(elapsed.as_nanos());
        }
        eprintln!(
            "search times (ns): max={} min={} mean={} total={} hits={}", 
            search_times.iter().max().unwrap(),
            search_times.iter().min().unwrap(),
            search_times.iter().copied().sum::<u128>() / search_times.len() as u128,
            num_searches,
            hits,
        );
        eprintln!("total hits: {hits} at max_distance={max_distance:.4} out of {num_searches} queries against {count} vectors.");

    }
}
