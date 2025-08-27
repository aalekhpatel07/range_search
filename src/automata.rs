use fst::Automaton;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Budget {
    L2(f32),
    Hamming(i64),
}

impl Budget {
    pub fn is_non_negative(&self) -> bool {
        match self {
            Budget::Hamming(d) => *d >= 0,
            Budget::L2(d) => *d >= 0.,
        }
    }

    pub fn step(&self, from: u8, to: u8) -> Budget {
        match self {
            Budget::Hamming(remaining) => {
                let step_by = i64::from((from ^ to).count_ones());
                Budget::Hamming((*remaining).saturating_sub(step_by))
            }
            Budget::L2(remaining) => {
                let from = from as f32;
                let to = to as f32;
                let step_by = (from - to) * (from - to);
                Budget::L2(*remaining - step_by)
            }
        }
    }
}

impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Budget::Hamming(v) => {
                write!(f, "{}", *v)
            }
            Budget::L2(v) => {
                write!(f, "{:.3}", *v)
            }
        }
    }
}

#[derive(Debug)]
pub struct VectorSetAutomata<'a, const N: usize> {
    query: &'a [u8],
    max_distance: Budget,
}

impl<'a, const N: usize> VectorSetAutomata<'a, N> {
    pub fn new(query: &'a [u8], max_distance: Budget) -> Self {
        Self {
            query,
            max_distance,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VectorSetAutomataState {
    budget: Budget,
    position: usize,
    query_is_bad_size: bool,
}

impl std::fmt::Display for VectorSetAutomataState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(budget: {}, position: {})", self.budget, self.position)
    }
}

impl<const N: usize> Automaton for VectorSetAutomata<'_, N> {
    type State = VectorSetAutomataState;

    fn start(&self) -> Self::State {
        VectorSetAutomataState {
            budget: self.max_distance,
            position: 0,
            query_is_bad_size: self.query.len() != { N },
        }
    }
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        VectorSetAutomataState {
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
