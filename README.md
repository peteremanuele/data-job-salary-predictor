
# Salary Predictor for Remote Work

This Python script uses machine learning algorithms to predict the salary for remote work based on job title, experience level, company location, and company size.

## Getting Started

1.  Install Python 3.7 or higher.
    
2.  Install the required Python packages if needed
    
3.  Run the script by executing the following command in your terminal:
    
    Copy code
    
    `python ds_salary_predictor.py` 
    
4.  Follow the on-screen instructions to input the job title, experience level, company location, company size, and model type (linear regression, decision tree regression, or random forest regression).
    

## Input Format

The script takes the following inputs:

-   Job Title: a string containing the job title.
-   Experience Level: a string containing the experience level (MI, SE, EN, or EX).
-   Company Location: a string containing the company location (ISO 3166-1 alpha-2 country code).
-   Company Size: an integer value for number of employees 
-   Model Type: a string containing the model type (linear regression, decision tree regression, or random forest regression).

## Output Format

The script outputs the predicted salary in USD.

## Model Training

The machine learning models used in this script were trained on the [Data Science Jobs Analysis dataset](https://www.kaggle.com/niyalthakkar/data-science-jobs-analysis) using scikit-learn.

## License

This project is licensed under the MIT License. See the [LICENSE](https://chat.openai.com/LICENSE) file for details.

## Acknowledgments

-   [Niyal Thakkar](https://www.kaggle.com/niyalthakkar) for providing the [Data Science Jobs Analysis dataset](https://www.kaggle.com/niyalthakkar/data-science-jobs-analysis).
