import pandas as pd
import joblib

# Load the trained model
def load_model():
    try:
        model = joblib.load('model25.pkl')
        return model
    except FileNotFoundError:
        raise Exception("Model file not found. Please check the path and file name.")
    
def process_input35(input_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([input_data])

    # Encoding categorical variables
    df['ChestPainType_ASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    df['ChestPainType_ATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    df['ExerciseAngina_Y'] = (df['ExerciseAngina'] == 'Y').astype(int)
    df['ST_Slope_Flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    df['ST_Slope_Up'] = (df['ST_Slope'] == 'Up').astype(int)

    # Drop the original categorical columns
    df.drop(columns=['ChestPainType', 'ExerciseAngina', 'ST_Slope'], inplace=True)

    # Define the expected columns as per the specific model requirements
    expected_columns = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'MaxHR',
        'Oldpeak',
        'ChestPainType_ASY',
        'ChestPainType_ATA',
        'ExerciseAngina_Y',
        'ST_Slope_Flat',
        'ST_Slope_Up'
    ]

    # Reorder columns to match the specific model requirements
    df = df.reindex(columns=expected_columns)

    return df

def process_input25(input_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([input_data])

    # Encoding categorical variables
    df['Sex_M'] = (df['Sex'] == 'M').astype(int)
    df['ChestPainType_ASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    df['ChestPainType_ATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    df['ExerciseAngina_Y'] = (df['ExerciseAngina'] == 'Y').astype(int)
    df['ST_Slope_Flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    df['ST_Slope_Up'] = (df['ST_Slope'] == 'Up').astype(int)

    # Drop the original categorical columns
    df.drop(columns=['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope'], inplace=True)

    # Define the expected columns as per the specific model requirements
    expected_columns = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'MaxHR',
        'Oldpeak',
        'Sex_M',
        'ChestPainType_ASY',
        'ChestPainType_ATA',
        'ExerciseAngina_Y',
        'ST_Slope_Flat',
        'ST_Slope_Up'
    ]

    # Reorder columns to match the specific model requirements
    df = df.reindex(columns=expected_columns)

    return df

def process_input15(input_data):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([input_data])

    # Encoding categorical variables
    df['Sex_M'] = (df['Sex'] == 'M').astype(int)
    df['ChestPainType_ASY'] = (df['ChestPainType'] == 'ASY').astype(int)
    df['ChestPainType_ATA'] = (df['ChestPainType'] == 'ATA').astype(int)
    df['ChestPainType_NAP'] = (df['ChestPainType'] == 'NAP').astype(int)
    df['ExerciseAngina_Y'] = (df['ExerciseAngina'] == 'Y').astype(int)
    df['ST_Slope_Flat'] = (df['ST_Slope'] == 'Flat').astype(int)
    df['ST_Slope_Up'] = (df['ST_Slope'] == 'Up').astype(int)

    # Drop the original categorical columns
    df.drop(columns=['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope'], inplace=True)

    # Define the expected columns as per the trained model
    expected_columns = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'FastingBS',
        'MaxHR',
        'Oldpeak',
        'Sex_M',
        'ChestPainType_ASY',
        'ChestPainType_ATA',
        'ChestPainType_NAP',
        'ExerciseAngina_Y',
        'ST_Slope_Flat',
        'ST_Slope_Up'
    ]

    # Reorder columns to match the training data
    df = df.reindex(columns=expected_columns)

    return df

model = load_model()

# Function to make a prediction
def predict(input_data):
    try:
        processed_data = process_input25(input_data)

        prediction = model.predict(processed_data)
        return prediction[0]  # Adjust as necessary based on your model's output format
    except Exception as e:
        # Log the exception details here for debugging
        # For example, you can print the error or log to a file
        print(f"Error during prediction: {str(e)}")
        # Optionally, you can re-raise the exception or handle it as appropriate
        raise