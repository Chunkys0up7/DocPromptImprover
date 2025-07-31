# 🚀 Demo Guide - Document Extraction Evaluation Service

This guide will walk you through running the comprehensive feedback API demo with visualization and statistical analysis.

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **All dependencies** installed (see requirements.txt)
3. **Git** for version control

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd DocPromptImprover

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install additional packages for demos
pip install streamlit plotly
```

## 🎯 Quick Start Demo

### Step 0: Kill Any Existing Sessions
```bash
# Find and kill processes using ports 8000 and 8501
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

### Step 1: Start the API Server
**Open a PowerShell window** and run:
```bash
python -m uvicorn src.api.evaluation_service:app --host 127.0.0.1 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Step 2: Test the API Server
**Open a NEW PowerShell window** and run:
```bash
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -Method GET
```

**Expected Output:** JSON response with status "healthy"

### Step 3: Run the API Test Demo
**In the same PowerShell window**, run:
```bash
python demos/api_test_demo.py
```

**What You'll See:**
- ✅ Feedback submission (single and batch)
- 📊 Statistical analysis with accuracy rates
- 📈 Field-level aggregation
- 🚨 Active alerts for problematic fields
- 🔧 Optimization recommendations
- 🌐 All API endpoints tested

### Step 4: Start the Visualization Dashboard
**Open another PowerShell window** and run:
```bash
python -m streamlit run demos/feedback_api_demo.py
```

**What You'll See:**
- 🌐 Dashboard opens automatically in your browser at `http://localhost:8501`
- 📊 Interactive charts and visualizations
- 🔄 "Generate Sample Data" button in the sidebar
- 📈 Multiple tabs: Field Performance, Trends, Alerts & Recommendations, Raw Data

## 📊 Demo Features

### ✅ Feedback Collection
- **Single and batch feedback submission**
- **Field-level feedback with corrections**
- **User comments and reasoning**
- **Real-time processing**

### ✅ Statistical Analysis
- **Overall accuracy rates** (currently showing ~46%)
- **Field-level performance breakdown**
- **Most problematic fields identification**
- **Trend analysis over time**

### ✅ Alert System
- **High error rate alerts** (4 active alerts shown)
- **Severity levels** (high, medium)
- **Threshold-based monitoring**
- **Real-time alert generation**

### ✅ Optimization Recommendations
- **Data-driven recommendations**
- **Impact assessment**
- **Priority levels**
- **Implementation tracking**

### ✅ API Endpoints
- **All 15+ endpoints tested successfully**
- **Proper error handling demonstrated**
- **Validation and data integrity**
- **RESTful API design**

## 🎯 Interactive Dashboard Features

### 📈 Field Performance Tab
- **Bar charts** showing accuracy rates by field
- **Color-coded** performance indicators
- **Drill-down** capabilities for detailed analysis

### 📊 Trends Tab
- **Line charts** showing performance over time
- **Trend direction indicators** (improving, declining, stable)
- **Time period filtering** (7d, 30d, 90d)

### 🚨 Alerts & Recommendations Tab
- **Alert cards** with severity indicators
- **Recommendation details** with expected impact
- **Action items** for optimization

### 📋 Raw Data Tab
- **Interactive data table** with all feedback records
- **Filtering and sorting** capabilities
- **Export functionality**

## 🔧 API Documentation

Visit the interactive API documentation at:
```
http://127.0.0.1:8000/docs
```

**Available Endpoints:**
- `POST /feedback` - Submit user feedback
- `GET /feedback/stats` - Get overall statistics
- `GET /feedback/aggregation` - Get field-level aggregation
- `GET /feedback/trends` - Get feedback trends
- `GET /feedback/alerts` - Get active alerts
- `GET /feedback/recommendations` - Get optimization recommendations

## 🚨 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing processes
   netstat -ano | findstr :8000
   taskkill /PID <PID_NUMBER> /F
   ```

2. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   pip install streamlit plotly
   ```

3. **Connection Refused**
   - Ensure API server is running on `http://127.0.0.1:8000`
   - Check firewall settings
   - Verify no other services are using the port

4. **Dashboard Not Loading**
   - Check if Streamlit is installed: `pip install streamlit`
   - Verify port 8501 is available
   - Check browser console for errors

### Getting Help

- **API Documentation:** `http://127.0.0.1:8000/docs`
- **Test Coverage:** Run `python -m pytest tests/ -v --cov=src`
- **Logs:** Check server logs for detailed error messages

## 📈 Sample Data

The demo includes realistic sample data with:
- **Multiple document types:** Invoices, receipts, contracts, forms, letters
- **Various fields:** Invoice numbers, dates, amounts, vendor names, etc.
- **Different feedback statuses:** Correct, incorrect, partial
- **Reason codes:** Bad OCR, ambiguous prompts, wrong formats, etc.
- **Confidence scores:** Realistic confidence scores for each field
- **Time-based data:** Historical data for trend analysis

## 🎯 Success Indicators

You'll know everything is working when you see:
- ✅ API server logs showing "Uvicorn running on http://127.0.0.1:8000"
- ✅ API test demo showing successful feedback submissions and analysis
- ✅ Dashboard loading in your browser with interactive charts
- ✅ Sample data generation working in the dashboard
- ✅ All 43 tests passing with 48% code coverage

## 🚀 Next Steps

After running the demo:

1. **Integrate with Your Application**
   - Use the API client code as a reference
   - Implement feedback collection in your app
   - Set up monitoring and alerts

2. **Customize for Your Use Case**
   - Modify field definitions
   - Adjust alert thresholds
   - Customize visualization charts

3. **Scale Up**
   - Deploy the service for production use
   - Set up database persistence
   - Configure monitoring and logging

4. **Continuous Improvement**
   - Use feedback to optimize prompts
   - Monitor performance trends
   - Implement A/B testing

## 📚 Additional Resources

- **Project Documentation:** See `README.md` for project overview
- **API Reference:** `http://127.0.0.1:8000/docs`
- **Test Suite:** Run `python -m pytest tests/` for comprehensive testing
- **Code Coverage:** Run `python -m pytest tests/ --cov=src --cov-report=html`

---

**Happy Demo-ing! 🎉**

For questions or issues, please check the troubleshooting section or refer to the project documentation. 