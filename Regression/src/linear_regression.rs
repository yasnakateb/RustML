use rand;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::error::Error;
use rusty_machine;
use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::analysis::score::neg_mean_squared_error;
use rusty_machine::learning::SupModel;
use utils::datasets::get_records_from_file;

// For regression
pub fn r_squared_score(y_test: &[f64], y_preds: &[f64]) -> f64 {
    let model_variance: f64 = y_test.iter().zip(y_preds.iter()).fold(
        0., |v, (y_i, y_i_hat)| {
            v + (y_i - y_i_hat).powi(2)
        }
    );

    // Gets the mean for the actual values to be used later
    let y_test_mean = y_test.iter().sum::<f64>() as f64
        / y_test.len() as f64;

    // Finds the variance
    let variance =  y_test.iter().fold(
        0., |v, &x| {v + (x - y_test_mean).powi(2)}
    );
    let r2_calculated: f64 = 1.0 - (model_variance / variance);
    r2_calculated
}

pub fn run() -> Result<(), Box<dyn Error>> {
    // Get all the data
    let filename = "../datasets/Housing.csv";
    let mut data = get_records_from_file(&filename);

    // Shuffling data serves the purpose of reducing 
    // variance and making sure that models remain general and overfit less. 
    data.shuffle(&mut thread_rng());

    // Split the data into 80% train and 20% tests, 
    // and convert them into f64 vectors.
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // Differentiates the features and the targets
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();
    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    /*
    //neg_mean_squared_error expects the inputs to be in matrix format.
    */

    // col 13 ==> medv(Price)

    //** We will try to predict the price of the house 
    //** based on the predictors available.

    // Convert the data into matrices for rusty machine
    let boston_x_train = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train = Vector::new(boston_y_train);
    let boston_x_test = Matrix::new(test_size, 13, boston_x_test);
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);
    
    // Create a linear regression model
    let mut lin_model = LinRegressor::default();
    println!("{:?}", lin_model);

    // Train the model
    let _result = lin_model.train(&boston_x_train, &boston_y_train);
    match _result {
        Ok(_result)=> {
            println!("Status: Trained");
        },
        Err(e)=> {
            println!("Status: Error\n{:?}", e);  
        }
    }

    // Predict
    let predictions = lin_model.predict(&boston_x_test).unwrap();
    let predictions = Matrix::new(test_size, 1, predictions);
    let acc = neg_mean_squared_error(&predictions, &boston_y_test);
    println!("linear regression error: {:?}", acc);
    // Coefficient of determination also called as R2 score is 
    // used to evaluate the performance of a linear regression model. 
    println!("linear regression R2 score: {:?}", r_squared_score(
        &boston_y_test.data(), &predictions.data()));

    Ok(())
}