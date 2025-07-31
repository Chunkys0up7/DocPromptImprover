# Document Extraction Evaluation Service - Demos

This directory contains comprehensive demonstrations of the Document Extraction Evaluation Service, showcasing feedback collection, analysis, and visualization capabilities.

## 🚀 Available Demos

### 1. Feedback API Demo with Visualization Dashboard
**File:** `feedback_api_demo.py`

A comprehensive web-based dashboard built with Streamlit that provides:
- **Interactive Visualizations**: Charts and graphs showing feedback trends, field performance, and accuracy rates
- **Real-time Statistics**: Live metrics for feedback records, accuracy rates, and alerts
- **Sample Data Generation**: Generate realistic feedback data for testing
- **API Documentation**: Built-in documentation of all available endpoints

**Features:**
- 📊 Field-level performance analysis
- 📈 Trend visualization over time
- 🚨 Real-time alerts and recommendations
- 📋 Raw data exploration
- 🔄 Interactive data generation

**To run:**
```bash
# Install required packages
pip install streamlit plotly

# Run the dashboard
python demos/feedback_api_demo.py
```

The dashboard will be available at `http://localhost:8501`

### 2. API Test Demo
**File:** `api_test_demo.py`

A programmatic demonstration of all API endpoints, including:
- **Feedback Submission**: Single and batch feedback submission
- **Statistical Analysis**: Retrieving and analyzing feedback statistics
- **Trend Analysis**: Getting feedback trends over time
- **Alert Monitoring**: Checking for active alerts and recommendations
- **Error Handling**: Demonstrating validation and error responses

**Features:**
- 🔄 Async API client with proper error handling
- 📊 Comprehensive statistical analysis
- 🚨 Alert and recommendation monitoring
- ⚠️ Error handling and validation testing
- 📝 Sample data generation

**To run:**
```bash
# Make sure the API server is running first
python -m src.api.evaluation_service

# In another terminal, run the API test demo
python demos/api_test_demo.py
```

### 3. Feedback Demo (Original)
**File:** `feedback_demo.py`

The original feedback demonstration showing:
- **Feedback Collection**: How user feedback is collected and validated
- **Analysis**: Field-level aggregation and trend analysis
- **Alerts**: Automated alert generation based on feedback patterns
- **Recommendations**: Optimization recommendations based on feedback

**To run:**
```bash
python demos/feedback_demo.py
```

### 4. Comprehensive Demo
**File:** `comprehensive_demo.py`

A complete demonstration of the entire evaluation pipeline, including:
- **Document Evaluation**: Complete document evaluation workflow
- **Field Analysis**: Individual field evaluation and scoring
- **Statistical Aggregation**: Document-level and system-level statistics
- **Performance Metrics**: Comprehensive performance analysis

**To run:**
```bash
python demos/comprehensive_demo.py
```

### 5. Evaluation Demo
**File:** `evaluation_demo.py`

Focused demonstration of the evaluation capabilities:
- **Field Evaluation**: Individual field extraction evaluation
- **Confidence Analysis**: Confidence score correlation analysis
- **Error Detection**: Pattern detection and error analysis
- **Optimization**: Performance optimization recommendations

**To run:**
```bash
python demos/evaluation_demo.py
```

### 6. Simple Demo
**File:** `simple_demo.py`

A basic demonstration for quick testing:
- **Quick Start**: Minimal setup required
- **Basic Functionality**: Core evaluation features
- **Sample Data**: Built-in sample data for immediate testing

**To run:**
```bash
python demos/simple_demo.py
```

## 🛠️ Setup and Requirements

### Prerequisites
```bash
# Install all required packages
pip install -r requirements.txt
pip install -r requirements-dev.txt

# For the dashboard demo, also install:
pip install streamlit plotly
```

### API Server Setup
Before running the API demos, start the evaluation service:

```bash
# Start the API server
python -m src.api.evaluation_service

# The server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 📊 Demo Data

All demos include realistic sample data generation with:
- **Multiple Document Types**: Invoices, receipts, contracts, forms, letters
- **Various Fields**: Invoice numbers, dates, amounts, vendor names, etc.
- **Different Feedback Statuses**: Correct, incorrect, partial
- **Reason Codes**: Bad OCR, ambiguous prompts, wrong formats, etc.
- **Confidence Scores**: Realistic confidence scores for each field
- **Time-based Data**: Historical data for trend analysis

## 🔗 API Endpoints Demonstrated

### Feedback Endpoints
- `POST /feedback` - Submit user feedback
- `GET /feedback/stats` - Get overall feedback statistics
- `GET /feedback/aggregation` - Get field-level aggregation
- `GET /feedback/trends` - Get feedback trends over time
- `GET /feedback/alerts` - Get active alerts
- `GET /feedback/recommendations` - Get optimization recommendations
- `POST /feedback/reset` - Reset feedback data

### Evaluation Endpoints
- `POST /evaluate` - Evaluate document extraction results
- `GET /stats` - Get evaluation statistics
- `GET /performance-metrics` - Get performance metrics
- `GET /field-performance` - Get field performance analysis
- `GET /document-type-performance` - Get document type performance
- `GET /error-patterns` - Get detected error patterns
- `POST /optimize` - Generate optimization recommendations

### Utility Endpoints
- `GET /health` - Health check
- `GET /` - Root endpoint with service information
- `GET /config` - Get current configuration
- `POST /config` - Update configuration
- `POST /reset` - Reset evaluation statistics

## 📈 Key Features Demonstrated

### 1. Feedback Collection
- **User Feedback**: Collect feedback on extracted fields
- **Validation**: Automatic validation of feedback data
- **Corrections**: User-provided corrections for incorrect extractions
- **Comments**: User comments and reasoning

### 2. Statistical Analysis
- **Accuracy Rates**: Field-level and overall accuracy calculations
- **Trend Analysis**: Performance trends over time
- **Error Patterns**: Detection of common error patterns
- **Confidence Correlation**: Analysis of confidence score accuracy

### 3. Alert System
- **High Error Rates**: Alerts for fields with high error rates
- **Accuracy Drops**: Alerts for significant accuracy decreases
- **Feedback Spikes**: Alerts for unusual feedback volume increases
- **Severity Levels**: Different alert severity levels

### 4. Optimization Recommendations
- **Prompt Clarification**: Recommendations for ambiguous prompts
- **Format Standardization**: Recommendations for format issues
- **Validation Improvement**: Recommendations for validation rules
- **Impact Assessment**: Expected impact of recommendations

### 5. Visualization
- **Interactive Charts**: Bar charts, line charts, and trend visualizations
- **Real-time Updates**: Live updates as new data is added
- **Filtering**: Filter data by field, document type, time period
- **Export**: Export data and charts for reporting

## 🎯 Use Cases

### 1. Document Processing Quality Assurance
- Monitor extraction accuracy across different document types
- Identify problematic fields and prompt versions
- Track performance improvements over time

### 2. Prompt Engineering
- Use feedback to identify ambiguous prompts
- Optimize prompts based on user corrections
- A/B test different prompt versions

### 3. System Monitoring
- Set up alerts for quality degradation
- Monitor system performance in real-time
- Generate reports for stakeholders

### 4. User Experience Improvement
- Understand common user corrections
- Identify fields that need better validation
- Improve confidence score calibration

## 🔧 Customization

### Adding New Document Types
1. Update the `document_types` list in the data generators
2. Add appropriate fields for the new document type
3. Update validation rules if needed

### Customizing Alerts
1. Modify alert thresholds in the feedback collector
2. Add new alert types for specific use cases
3. Customize alert severity levels

### Extending Visualizations
1. Add new chart types using Plotly
2. Create custom metrics and KPIs
3. Implement new filtering options

## 🚨 Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```
   Error: Connection refused
   Solution: Start the API server with `python -m src.api.evaluation_service`
   ```

2. **Missing Dependencies**
   ```
   ImportError: No module named 'streamlit'
   Solution: Install with `pip install streamlit plotly`
   ```

3. **Port Already in Use**
   ```
   Error: Port 8000 is already in use
   Solution: Change the port or stop the existing service
   ```

4. **Validation Errors**
   ```
   ValidationError: Correction must be provided when status is incorrect
   Solution: Ensure all incorrect feedback includes correction values
   ```

### Getting Help

- Check the API documentation at `http://localhost:8000/docs`
- Review the test files for examples
- Check the logs for detailed error messages
- Ensure all dependencies are installed

## 📚 Next Steps

After running the demos:

1. **Integrate with Your Application**: Use the API client to integrate feedback collection
2. **Set Up Monitoring**: Configure alerts and monitoring for your use case
3. **Customize Analysis**: Adapt the analysis for your specific document types
4. **Scale Up**: Deploy the service for production use
5. **Continuous Improvement**: Use feedback to continuously improve your prompts

## 🤝 Contributing

To add new demos or improve existing ones:

1. Follow the existing code structure and patterns
2. Include comprehensive documentation
3. Add appropriate error handling
4. Test with various data scenarios
5. Update this README with new information

---

**Happy Demo-ing! 🎉** 