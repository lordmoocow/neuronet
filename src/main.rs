use clap::Parser;

use crate::cli::{Cli, Commands};

mod activation;
mod cli;
mod data;
mod export;
mod layer;
mod loss;
mod matrix;
mod model;
mod network;
mod test;
mod train;
mod visualisation;

fn main() {
    let args = Cli::parse();

    match args.command {
        Commands::Train(train_args) => train::handle_train(train_args),
        Commands::Test(test_args) => test::handle_test(test_args),
        Commands::Visualise(vis_args) => visualisation::handle_visualisation(vis_args),
    }
}
