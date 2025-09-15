# 🚀 Streamlit Cloud Deployment Guide

## Enhanced Marine Data Analytics Platform

This guide will help you deploy your enhanced Streamlit app to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Data Files**: Ensure all data files are included in your repository

## 🔧 Repository Structure

Your repository should have this structure:
```
SIH_25/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Dependencies
├── etl_phase1.py            # ETL script 1
├── etl_phase2_integration.py # ETL script 2
├── data/
│   ├── samples.csv
│   ├── asv_table.tsv
│   ├── env_data.nc
│   ├── images/
│   └── processed/
└── DEPLOYMENT_GUIDE.md
```

## 🚀 Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Enhanced Streamlit app with advanced features"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select the repository and branch
5. Set the main file path to `streamlit_app.py`
6. Click "Deploy!"

### 3. Configure App Settings

In the Streamlit Cloud dashboard:
- **App URL**: Your app will be available at `https://your-app-name.streamlit.app`
- **Repository**: Your GitHub repository
- **Branch**: Usually `main` or `master`
- **Main file**: `streamlit_app.py`

## 🔧 Environment Configuration

The app will automatically install dependencies from `requirements.txt`. 

### ⚠️ Version Compatibility Fix

If you encounter version conflicts during deployment, you have three options:

1. **Use the main requirements.txt** (recommended):
   ```txt
   streamlit>=1.28.0
   pandas>=1.5.0
   numpy>=1.24.0
   plotly>=5.15.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   xarray>=2023.1.0
   h5netcdf>=1.1.0
   openpyxl>=3.1.0
   pillow>=9.5.0
   pyyaml>=6.0
   ```

2. **Use minimal requirements** (if main fails):
   - Rename `requirements_minimal.txt` to `requirements.txt`

3. **Use conservative requirements** (for maximum compatibility):
   - Rename `requirements_conservative.txt` to `requirements.txt`

## 📊 Features Available After Deployment

### 🎯 Enhanced Dashboard
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover
- **Real-time Filtering**: Multi-dimensional data filtering
- **Data Quality Metrics**: Automated data quality assessment
- **Export Capabilities**: CSV, Excel, and JSON export

### 🔍 Data Explorer
- **Advanced Table View**: Sortable, filterable data tables
- **Column Selection**: Choose which columns to display
- **Pagination**: Handle large datasets efficiently
- **Search Functionality**: Search across sample IDs and species

### 🗺️ Geographic View
- **Interactive Maps**: Plotly-based maps with customization
- **Color Coding**: Color points by species, depth, or other variables
- **Size Mapping**: Size points by data values
- **Geographic Statistics**: Automatic calculation of spatial metrics

### 🔬 Sample Analysis
- **Detailed Sample View**: Comprehensive sample information
- **ASV Analysis**: Molecular data visualization and statistics
- **Environmental Context**: NetCDF data integration
- **Statistical Metrics**: Diversity indices and abundance analysis

### 📈 Advanced Analytics
- **Statistical Analysis**: Correlation matrices and summary statistics
- **Diversity Analysis**: Shannon and Simpson diversity indices
- **Environmental Analysis**: NetCDF data exploration
- **Trend Analysis**: Time series and spatial analysis

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **Data Not Loading**: Check that data files are in the repository
3. **ETL Scripts Not Found**: Ensure ETL scripts are in the root directory
4. **Memory Issues**: For large datasets, consider data sampling

### Performance Tips

1. **Data Caching**: The app uses Streamlit's caching for better performance
2. **Pagination**: Large datasets are paginated for better user experience
3. **Lazy Loading**: Images and heavy computations are loaded on demand

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs in the dashboard
2. Verify all files are in the repository
3. Ensure data files are accessible
4. Check the requirements.txt for missing dependencies

## 🎉 Success!

Once deployed, your enhanced Marine Data Analytics Platform will be available at your Streamlit Cloud URL with all the advanced features:

- ✅ Modern, responsive UI
- ✅ Interactive visualizations
- ✅ Advanced filtering and search
- ✅ Geographic mapping
- ✅ Statistical analysis
- ✅ Data export capabilities
- ✅ Comprehensive error handling
- ✅ Performance optimizations

Enjoy your enhanced marine data analytics platform! 🐟📊
