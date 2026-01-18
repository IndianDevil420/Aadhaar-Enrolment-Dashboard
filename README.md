# Aadhaar-Enrolment-Dashboard
End-to-end analytics dashboard for Aadhaar enrolment data with automated data cleaning, state/district normalization, anomaly detection, and interactive visualizations using Streamlit &amp; Plotly.

## ✨ Features

### 📊 Data Processing
- **Smart Data Loading**: Upload ZIP files containing multiple CSV files
- **Intelligent Column Mapping**: Automatically recognizes various column name formats
- **Advanced Data Cleaning**: 
  - Aggressive state and district name normalization
  - Handles spelling variations and inconsistencies
  - Deduplication of records
  - Date parsing with multiple format support
  
### 📈 Visualizations
- **Time Series Analysis**: Track enrolment trends over time with area-filled charts
- **Geographic Distribution**: State and district-level enrolment heatmaps
- **Age Group Analysis**: Stacked bar charts showing demographic breakdown
- **Intensity Heatmaps**: Visual representation of enrolment density
- **Comparative Analysis**: Side-by-side region comparison tools

### 🎯 Key Performance Indicators (KPIs)
- Total enrolments (overall and by age group)
- Month-over-month/Year-over-year growth rates
- Highest and lowest performing regions
- Age group distribution percentages

### 💡 AI-Powered Insights
- Automatic detection of low-performing regions
- Identification of slow growth patterns
- Spike and anomaly detection
- Trend analysis across time periods

### 🔍 Advanced Filtering
- Multi-state selection
- District-level filtering
- PIN code search
- Custom date range selection
- Age group segmentation

### 📥 Export Capabilities
- Download filtered datasets as CSV
- Export visualizations
- Generate custom reports

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aadhaar-enrolment-dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Upload your data**
   - Prepare a ZIP file containing CSV files with enrolment data
   - Use the sidebar file uploader to load your data
   - The dashboard will automatically process and display your data

## 📁 Data Format

### Expected CSV Structure

Your CSV files should contain the following columns (column names are flexible):

| Column | Possible Names | Type | Description |
|--------|---------------|------|-------------|
| Date | `date`, `enrolment_date`, `date_of_enrolment` | Date | Enrolment date |
| State | `state`, `state_name` | String | State name |
| District | `district`, `district_name` | String | District name |
| PIN Code | `pincode`, `pin`, `pin_code` | String | Postal code |
| Age 0-5 | `0_5`, `age_0_5`, `age_0to5` | Integer | Count of 0-5 years |
| Age 5-17 | `5_17`, `age_5_17`, `age_5to17` | Integer | Count of 5-17 years |
| Age 18+ | `18_plus`, `age_18_plus`, `age_18+` | Integer | Count of 18+ years |

### Sample Data

```csv
date,state,district,pincode,age_0_5,age_5_17,age_18_plus
2024-01-15,West Bengal,Kolkata,700001,150,300,500
2024-01-16,Maharashtra,Mumbai,400001,200,400,800
```

## 🛠️ Technical Architecture

### Core Technologies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Python Standard Library**: ZIP handling, date processing

### Key Components

#### Data Processing Pipeline
1. **Loading**: ZIP file extraction and CSV parsing
2. **Standardization**: Column name normalization
3. **Cleaning**: String trimming, date parsing, numeric coercion
4. **Consolidation**: State/district name unification using fuzzy matching
5. **Deduplication**: Removal of duplicate records
6. **Aggregation**: Calculation of derived metrics

#### Visualization Engine
- Dynamic chart generation based on filtered data
- Responsive layouts with Plotly Express and Graph Objects
- Custom color palettes for brand consistency
- Interactive tooltips and hover information

#### Insight Generation
- Rule-based analysis for pattern detection
- Statistical anomaly identification
- Trend calculation and forecasting
- Comparative metrics computation

## 🎨 Customization

### Color Palette
Modify the `COLOR_PALETTE` variable in `app.py`:
```python
COLOR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
```

### Page Configuration
Adjust settings in the `st.set_page_config()` call:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎯",
    layout="wide",
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: CSV files not recognized in ZIP
- **Solution**: Ensure all files have `.csv` extension (lowercase)

**Issue**: Date parsing errors
- **Solution**: Check that dates are in standard formats (YYYY-MM-DD, DD/MM/YYYY, etc.)

**Issue**: Missing columns error
- **Solution**: Verify your CSV contains at minimum: date, state, district, and age group columns

**Issue**: State/district names not consolidating
- **Solution**: Check the debug info in the sidebar to see normalized names

## 📊 Performance Optimization

For large datasets (>1 million rows):
- The application uses `@st.cache_data` for efficient data loading
- Filters are applied progressively to minimize memory usage
- Visualizations are limited to top 20-30 entries for clarity

## 🤝 Contributing

Created by **Ankan and Sudip** ❤️

## 📄 License

This project is provided as-is for educational and analytical purposes.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 📞 Support

For issues, questions, or suggestions:
1. Check the debug info in the sidebar
2. Review the troubleshooting section
3. Examine your data format against the expected structure

---

**Note**: This dashboard is designed for analytical purposes. Ensure compliance with data privacy regulations when handling personal information.
