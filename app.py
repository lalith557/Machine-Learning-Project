from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import json
import os
import io
from datetime import datetime, timedelta
import model as scheduler
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)
app.secret_key = 'nurse_scheduling_secret_key'

# Global variables to store model and data
global_data = None
global_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if global_data is None:
        flash('Please load data first', 'warning')
        return redirect(url_for('load_data'))
    
    # Get key metrics
    total_patients = len(global_data)
    total_nurses = global_data['Nurse_ID'].nunique()
    total_departments = global_data['Department'].nunique()
    avg_hours = round(global_data['Hours_Required'].mean(), 1)
    
    # Department distribution
    dept_counts = global_data['Department'].value_counts().reset_index()
    dept_counts.columns = ['Department', 'Count']
    dept_fig = px.pie(dept_counts, values='Count', names='Department', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    dept_chart = json.dumps(dept_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Acuity levels
    acuity_counts = global_data['Acuity_Level'].value_counts().sort_index().reset_index()
    acuity_counts.columns = ['Acuity Level', 'Count']
    acuity_fig = px.bar(acuity_counts, x='Acuity Level', y='Count',
                       color='Count', color_continuous_scale='Viridis')
    acuity_chart = json.dumps(acuity_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Age distribution
    age_fig = px.histogram(global_data, x='age', nbins=20, 
                         color_discrete_sequence=['#0083B8'])
    age_fig.update_layout(bargap=0.1)
    age_chart = json.dumps(age_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Date range
    min_date = global_data['last_checkup_date'].min().date()
    max_date = global_data['last_checkup_date'].max().date()
    
    return render_template(
        'dashboard.html',
        total_patients=total_patients,
        total_nurses=total_nurses,
        total_departments=total_departments,
        avg_hours=avg_hours,
        dept_chart=dept_chart,
        acuity_chart=acuity_chart,
        age_chart=age_chart,
        min_date=min_date,
        max_date=max_date
    )

@app.route('/load_data', methods=['GET', 'POST'])
def load_data():
    global global_data
    
    if request.method == 'POST':
        if 'data_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['data_file']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            try:
                # Save the file temporarily
                file_path = 'temp_dataset.xlsx'
                file.save(file_path)
                
                # Load and preprocess the data
                global_data = scheduler.load_and_preprocess_data(file_path)
                
                flash('Data loaded successfully!', 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                flash(f'Error loading data: {str(e)}', 'danger')
                return redirect(request.url)
    
    return render_template('load_data.html')

@app.route('/generate_schedule', methods=['GET', 'POST'])
def generate_schedule():
    global global_data, global_model
    
    if global_data is None:
        flash('Please load data first', 'warning')
        return redirect(url_for('load_data'))
    
    min_date = global_data['last_checkup_date'].min().date()
    max_date = global_data['last_checkup_date'].max().date()
    
    if request.method == 'POST':
        try:
            # Get form data
            schedule_date = request.form.get('schedule_date')
            max_hours = int(request.form.get('max_hours', 40))
            
            # Convert string date to datetime
            schedule_date = datetime.strptime(schedule_date, '%Y-%m-%d')
            
            # Train model if not already trained
            if global_model is None:
                flash('Training model... This may take a moment.', 'info')
                global_model = scheduler.train_model(global_data)
            
            # Generate schedule
            schedule, nurse_hours, actual_date = scheduler.optimize_schedule(
                global_data, global_model, schedule_date, max_hours=max_hours
            )
            
            # Prepare data for templates
            shifts = list(schedule.keys())
            
            # Prepare nurse workload data for chart
            workload_data = []
            for nurse, hours in nurse_hours.items():
                workload_data.append({
                    'nurse': f"Nurse_{int(nurse)}",
                    'hours': hours
                })
            
            # Sort by hours
            workload_data = sorted(workload_data, key=lambda x: x['hours'], reverse=True)
            
            # Create workload chart
            workload_df = pd.DataFrame(workload_data)
            workload_fig = px.bar(
                workload_df, 
                x='nurse', 
                y='hours',
                color='hours',
                color_continuous_scale='Viridis',
                labels={'hours': 'Assigned Hours', 'nurse': 'Nurse ID'}
            )
            
            # Add max hours line
            workload_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=max_hours,
                x1=len(workload_df)-0.5,
                y1=max_hours,
                line=dict(color="red", width=2, dash="dash")
            )
            
            workload_fig.add_annotation(
                x=len(workload_df)-1, 
                y=max_hours,
                text=f"Max Hours: {max_hours}",
                showarrow=False,
                yshift=10
            )
            
            workload_chart = json.dumps(workload_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template(
                'schedule_results.html',
                schedule=schedule,
                shifts=shifts,
                nurse_hours=nurse_hours,
                actual_date=actual_date,
                workload_chart=workload_chart,
                max_hours=max_hours
            )
            
        except Exception as e:
            flash(f'Error generating schedule: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template(
        'generate_schedule.html',
        min_date=min_date,
        max_date=max_date
    )

@app.route('/view_data')
def view_data():
    if global_data is None:
        flash('Please load data first', 'warning')
        return redirect(url_for('load_data'))
    
    # Convert data to HTML table
    data_html = global_data.head(100).to_html(classes='table table-striped table-hover', index=False)
    
    return render_template('view_data.html', data_html=data_html)

@app.route('/download_schedule', methods=['POST'])
def download_schedule():
    try:
        # Get schedule data from form
        schedule_data = request.form.get('schedule_data')
        date = request.form.get('date')
        
        # Convert JSON string to dict
        schedule = json.loads(schedule_data)
        
        # Create a DataFrame from the schedule
        rows = []
        for shift, assignments in schedule.items():
            for patient, nurse in assignments.items():
                rows.append({
                    'Shift': shift,
                    'Patient': patient,
                    'Assigned Nurse': nurse
                })
        
        df = pd.DataFrame(rows)
        
        # Create a string buffer
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'nurse_schedule_{date}.csv'
        )
        
    except Exception as e:
        flash(f'Error downloading schedule: {str(e)}', 'danger')
        return redirect(url_for('generate_schedule'))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)