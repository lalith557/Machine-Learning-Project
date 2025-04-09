
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict
from datetime import datetime, timedelta

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_excel(filepath)

    # Clean column names
    data.columns = data.columns.str.strip().str.replace(' ', '_')

    # Handle missing values
    fill_values = {
        'blood_pressure': data['blood_pressure'].mode()[0],
        'blood_sugar_levels': pd.to_numeric(data['blood_sugar_levels'], errors='coerce').mean(),
        'lifestyle_interventions': "None",
        'preventive_measures': "None"
    }
    data = data.fillna(fill_values)

    # Convert last_checkup_date to datetime
    if 'last_checkup_date' in data.columns:
        data['last_checkup_date'] = pd.to_datetime(data['last_checkup_date'], errors='coerce')
    else:
        print("Column 'last_checkup_date' not found in the dataset.")

    # Convert numerical columns
    num_cols = ['Hours_Required', 'age']
    for col in num_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(data[col].median())

    # Encode categorical variables
    le = LabelEncoder()
    cat_cols = ['gender', 'chronic_disease', 'Department', 'Shift_Type', 'Nurse_ID', 'Acuity_Level']
    for col in cat_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))

    # Calculate priority score
    data['priority_score'] = (
        (data['Acuity_Level'].astype(float) * 0.5) +
        (data['Hours_Required'].astype(float) * 0.3) +
        (data['age'].astype(float) / 100 * 0.2)
    )

    return data

def train_model(data, k=5):
    features = ['age', 'gender', 'chronic_disease', 'Acuity_Level', 'Department',
               'Shift_Type', 'Hours_Required', 'priority_score']
    X = data[features]
    y = data['Nurse_ID']

    model = RandomForestClassifier(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=k, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

    # Fit the model on the entire dataset with the best parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)

    return best_model

def get_date_input():
    while True:
        date_str = input("Enter date for schedule (DD-MM-YYYY): ")
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except ValueError:
            print("Invalid format. Please use DD-MM-YYYY.")

def optimize_schedule(data, model, target_date, max_hours=40):
    # Convert target_date to datetime if it isn't already
    target_date = pd.to_datetime(target_date)

    # Find exact matches for the target date
    daily_data = data[data['last_checkup_date'].dt.date == target_date.date()].copy()

    # If no appointments found, create synthetic data
    if daily_data.empty:
        print(f"\nNo appointments found for {target_date.date()}. Generating synthetic data.")

        # Create synthetic data for the target date
        synthetic_data = {
            'age': np.random.randint(20, 80, size=5),  # Example ages
            'gender': np.random.choice(['Male', 'Female'], size=5),
            'chronic_disease': np.random.choice(['Yes', 'No'], size=5),
            'Acuity_Level': np.random.randint(1, 5, size=5),
            'Department': np.random.choice(['Cardiology', 'Neurology', 'Pediatrics'], size=5),
            'Shift_Type': np.random.choice(['Day', 'Night'], size=5),
            'Hours_Required': np.random.randint(1, 8, size=5),
            'Nurse_ID': np.random.choice(data['Nurse_ID'].unique(), size=5)  # Randomly assign existing Nurse IDs
        }

        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df['last_checkup_date'] = target_date

        # Calculate priority score for synthetic data
        synthetic_df['priority_score'] = (
            (synthetic_df['Acuity_Level'].astype(float) * 0.5) +
            (synthetic_df['Hours_Required'].astype(float) * 0.3) +
            (synthetic_df['age'].astype(float) / 100 * 0.2)
        )

        # Encode categorical variables for synthetic data
        le = LabelEncoder()
        synthetic_df['gender'] = le.fit_transform(synthetic_df['gender'].astype(str))
        synthetic_df['chronic_disease'] = le.fit_transform(synthetic_df['chronic_disease'].astype(str))
        synthetic_df['Department'] = le.fit_transform(synthetic_df['Department'].astype(str))
        synthetic_df['Shift_Type'] = le.fit_transform(synthetic_df['Shift_Type'].astype(str))

        daily_data = synthetic_df

    # Make predictions
    features = ['age', 'gender', 'chronic_disease', 'Acuity_Level', 'Department',
               'Shift_Type', 'Hours_Required', 'priority_score']
    daily_data['predicted_nurse'] = model.predict(daily_data[features])

    # Initialize tracking
    nurse_hours = defaultdict(float)
    schedule = {}

    # Process by shift and priority
    for shift, shift_group in daily_data.groupby('Shift_Type'):
        shift_assignments = {}

        # Sort by priority (highest first)
        for _, row in shift_group.sort_values('priority_score', ascending=False).iterrows():
            patient = f"Patient_{row['patient_id']}" if 'patient_id' in row else f"Patient_{row.name}"
            hours = row['Hours_Required']

            # Find available nurses
            available_nurses = [n for n in set(data['Nurse_ID'])
                              if nurse_hours[n] + hours <= max_hours]

            if available_nurses:
                # Assign to nurse with least current hours
                assigned_nurse = min(available_nurses, key=lambda x: nurse_hours[x])
                shift_assignments[patient] = f"Nurse_{int(assigned_nurse)}"
                nurse_hours[assigned_nurse] += hours
            else:
                shift_assignments[patient] = "UNASSIGNED"

        schedule[f"Shift_{int(shift)}"] = shift_assignments

    return schedule, nurse_hours, target_date.date()

def main():
    print("Nurse Scheduling System")
    print("-----------------------")

    # Load data
    try:
        data = load_and_preprocess_data("Final Dataset.xlsx")
        print(f"\nData loaded successfully. Date range in system:")
        print(f"Earliest last check-up date: {data['last_checkup_date'].min().date()}")
        print(f"Latest last check-up date: {data['last_checkup_date'].max().date()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Train model
    model = train_model(data)

    while True:
        # Get date
        schedule_date = get_date_input()

        # Generate schedule
        schedule, nurse_hours, actual_date = optimize_schedule(data, model, schedule_date)

        if schedule:
            print(f"\nSchedule for {actual_date}:")
            print("=========================")
            for shift, assignments in schedule.items():
                print(f"\n{shift}:")
                for patient, nurse in assignments.items():
                    print(f"  {patient} -> {nurse}")

            print("\nNurse Workload Summary:")
            print("----------------------")
            for nurse, hours in nurse_hours.items():
                print(f"Nurse_{int(nurse)}: {hours:.1f} hours")

        # Ask if user wants to check another date
        another = input("\nCheck another date? (y/n): ").lower()
        if another != 'y':
            break

if __name__ == "__main__":
    main()