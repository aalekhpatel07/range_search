# range_search

An [automaton](https://docs.rs/fst/latest/fst/automaton/trait.Automaton.html) to query for nearest neighbours of a byte vector (within a given distance) against an [Fst](https://docs.rs/fst/latest/fst/) that contains a population of byte vectors.

## Installation

```sh
cargo add range_search
```

## Usage

- An automaton that accepts only (144,)-u8 vectors if they are within a 
squared L2-distance of 100_000.0 of a given query vector.

```rust
use range_search::RangeSearch;
use fst::{Automaton, set::{Set, SetBuilder}, IntoStreamer, Streamer};

fn main() {

    // the vector we want to find neighbors for.
    let query = [0x01; 144];
    let aut = RangeSearch::<144>::new_l2(&query, 100_000.0).unwrap();

    // Suppose you have some fst::Set of vectors.
    let set = SetBuilder::memory().into_set();

    // Iterate over the set of vectors close enough to the query vector.
    let mut stream = set.search(aut).into_stream();
    while let Some(hit) = stream.next() {
        eprintln!("found a vector within the given range of the query vector {hit:#?}");
    }
}
```

- An automaton that accepts only (32,)-u8 vectors (i.e. 256-dimensional binary vectors) if they are within a hamming distance of 5 of a given query vector.

```rust
use range_search::RangeSearch;
use fst::{Automaton, set::{Set, SetBuilder}, IntoStreamer, Streamer};

fn main() {

    // the 256-dimensional binary vector (packed into single bit per component) that we want to find neighbors for.
    let query = [0x01; 32];
    let aut = RangeSearch::<32>::new_hamming(&query, 5).unwrap();

    // Suppose you have some fst::Set of vectors.
    let set = SetBuilder::memory().into_set();

    // Iterate over the set of vectors close enough to the query vector.
    let mut stream = set.search(aut).into_stream();
    while let Some(hit) = stream.next() {
        eprintln!("found a vector within the given range of the query vector {hit:#?}");
    }
}
```
