use anyhow::Result;
use webgraph_algo::{
    prelude::*,
    utils::{HyperLogLogCounterArray, HyperLogLogCounterArrayBuilder},
};
use xxhash_rust::xxh3::Xxh3Builder;

/// The number of trials to run to ensure a bad seed does not
/// fail the test
const NUM_TRIALS: u64 = 100;
/// The requires number of successes required for the test to pass
const REQUIRED_TRIALS: u64 = 90;

#[test]
fn test_single() -> Result<()> {
    let sizes = [1, 10, 100, 1000, 100_000];
    let log2ms = [4, 6, 8, 12];

    for size in sizes {
        for log2m in log2ms {
            let rsd = HyperLogLogCounterArray::relative_standard_deviation(log2m);
            let mut correct = 0;

            for trial in 0..NUM_TRIALS {
                let counters = HyperLogLogCounterArrayBuilder::new()
                    .word_type::<u16>()
                    .log_2_num_registers(log2m)
                    .num_elements_upper_bound(size)
                    .hasher_builder(Xxh3Builder::new().with_seed(trial))
                    .build(1)?;
                let mut counter = counters.get_counter(0);
                let incr = (1 << 32) / size as i64;
                let mut x = i64::MIN;
                for _ in 0..size {
                    counter.add(x);
                    x += incr;
                }

                let float_size = size as f64;

                if (float_size - counter.estimate_count()).abs() / float_size < 2.0 * rsd {
                    correct += 1;
                }
            }

            assert!(
                correct >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct = {} < {}",
                size,
                log2m,
                correct,
                REQUIRED_TRIALS
            );
        }
    }

    Ok(())
}

#[test]
fn test_double() -> Result<()> {
    let sizes = [1, 10, 100, 1000, 100_000];
    let log2ms = [4, 6, 8, 12];

    for size in sizes {
        for log2m in log2ms {
            let rsd = HyperLogLogCounterArray::relative_standard_deviation(log2m);
            let mut correct_0 = 0;
            let mut correct_1 = 0;

            for trial in 0..NUM_TRIALS {
                let counters = HyperLogLogCounterArrayBuilder::new()
                    .word_type::<u16>()
                    .log_2_num_registers(log2m)
                    .num_elements_upper_bound(size)
                    .hasher_builder(Xxh3Builder::new().with_seed(trial))
                    .build(2)?;
                let incr = (1 << 32) / size as i64;
                let mut x = i64::MIN;
                for _ in 0..size {
                    counters.get_counter(0).add(x);
                    counters.get_counter(1).add(x);
                    x += incr;
                }

                let float_size = size as f64;

                if (float_size - counters.get_counter(0).estimate_count()).abs() / float_size
                    < 2.0 * rsd
                {
                    correct_0 += 1;
                }
                if (float_size - counters.get_counter(1).estimate_count()).abs() / float_size
                    < 2.0 * rsd
                {
                    correct_1 += 1;
                }
            }

            assert!(
                correct_0 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_0 = {} < {}",
                size,
                log2m,
                correct_0,
                REQUIRED_TRIALS
            );
            assert!(
                correct_1 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_1 = {} < {}",
                size,
                log2m,
                correct_1,
                REQUIRED_TRIALS
            );
        }
    }

    Ok(())
}

#[test]
fn test_merge_safe() -> Result<()> {
    let sizes = [1, 10, 100, 1000, 100_000];
    let log2ms = [4, 6, 8, 12];

    for size in sizes {
        for log2m in log2ms {
            let rsd = HyperLogLogCounterArray::relative_standard_deviation(log2m);
            let mut correct_0 = 0;
            let mut correct_1 = 0;

            for trial in 0..NUM_TRIALS {
                let counters = HyperLogLogCounterArrayBuilder::new()
                    .word_type::<u16>()
                    .log_2_num_registers(log2m)
                    .num_elements_upper_bound(size)
                    .hasher_builder(Xxh3Builder::new().with_seed(trial))
                    .build(2)?;
                let incr = (1 << 32) / (size * 2) as i64;
                let mut x = i64::MIN;
                for _ in 0..size {
                    counters.get_counter(0).add(x);
                    x += incr;
                    counters.get_counter(1).add(x);
                    x += incr;
                }

                counters.get_counter(0).merge(&counters.get_counter(1));

                let float_size = size as f64;

                if (float_size * 2.0 - counters.get_counter(0).estimate_count()).abs()
                    / (float_size * 2.0)
                    < 2.0 * rsd
                {
                    correct_0 += 1;
                }
                if (float_size - counters.get_counter(1).estimate_count()).abs()
                    / (float_size * 2.0)
                    < 2.0 * rsd
                {
                    correct_1 += 1;
                }
            }

            assert!(
                correct_0 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_0 = {} < {}",
                size,
                log2m,
                correct_0,
                REQUIRED_TRIALS
            );
            assert!(
                correct_1 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_1 = {} < {}",
                size,
                log2m,
                correct_1,
                REQUIRED_TRIALS
            );
        }
    }

    Ok(())
}

#[test]
fn test_merge_unsafe() -> Result<()> {
    let sizes = [1, 10, 100, 1000, 100_000];
    let log2ms = [4, 6, 8, 12];

    for size in sizes {
        for log2m in log2ms {
            let rsd = HyperLogLogCounterArray::relative_standard_deviation(log2m);
            let mut correct_0 = 0;
            let mut correct_1 = 0;

            for trial in 0..NUM_TRIALS {
                let counters = HyperLogLogCounterArrayBuilder::new()
                    .word_type::<u16>()
                    .log_2_num_registers(log2m)
                    .num_elements_upper_bound(size)
                    .hasher_builder(Xxh3Builder::new().with_seed(trial))
                    .build(2)?;
                let incr = (1 << 32) / (size * 2) as i64;
                let mut x = i64::MIN;
                for _ in 0..size {
                    counters.get_counter(0).add(x);
                    x += incr;
                    counters.get_counter(1).add(x);
                    x += incr;
                }

                unsafe {
                    counters
                        .get_counter(0)
                        .merge_unsafe(&counters.get_counter(1));
                }

                let float_size = size as f64;

                if (float_size * 2.0 - counters.get_counter(0).estimate_count()).abs()
                    / (float_size * 2.0)
                    < 2.0 * rsd
                {
                    correct_0 += 1;
                }
                if (float_size - counters.get_counter(1).estimate_count()).abs()
                    / (float_size * 2.0)
                    < 2.0 * rsd
                {
                    correct_1 += 1;
                }
            }

            assert!(
                correct_0 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_0 = {} < {}",
                size,
                log2m,
                correct_0,
                REQUIRED_TRIALS
            );
            assert!(
                correct_1 >= REQUIRED_TRIALS,
                "assertion failed for size {} and log2m {}: correct_1 = {} < {}",
                size,
                log2m,
                correct_1,
                REQUIRED_TRIALS
            );
        }
    }

    Ok(())
}
