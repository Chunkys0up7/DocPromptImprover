"""
Feedback API Demo with Visualization Dashboard

This demo shows how to use the feedback API endpoints and provides
a web-based dashboard to visualize feedback data, trends, and insights.
"""

import json
import time
import asyncio
import httpx
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

# Mock API server for demo purposes
class MockFeedbackAPI:
    """Mock API server for demonstration purposes."""
    
    def __init__(self):
        self.feedback_data = []
        self.base_url = "http://localhost:8000"
        
    def generate_sample_feedback(self, num_records: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic sample feedback data."""
        
        document_types = ["invoice", "receipt", "contract", "form", "letter"]
        fields = ["invoice_number", "date", "amount", "vendor_name", "customer_name", 
                 "tax_amount", "total_amount", "payment_terms", "due_date", "po_number"]
        prompt_versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
        feedback_statuses = ["correct", "incorrect", "partial"]
        reason_codes = ["bad_ocr", "prompt_ambiguous", "irrelevant_text", "wrong_format", 
                       "missing_value", "extraneous_value", "confusing_context", "other"]
        
        feedback_records = []
        
        for i in range(num_records):
            # Generate random document
            doc_type = random.choice(document_types)
            prompt_version = random.choice(prompt_versions)
            
            # Generate 3-6 fields per document
            num_fields = random.randint(3, 6)
            field_feedback = []
            
            for j in range(num_fields):
                field_name = random.choice(fields)
                status = random.choice(feedback_statuses)
                
                field_data = {
                    "field_name": field_name,
                    "shown_value": f"Sample value {i}-{j}",
                    "feedback_status": status,
                    "correction": f"Corrected value {i}-{j}" if status == "incorrect" else None,
                    "comment": f"User comment for {field_name}",
                    "reason_code": random.choice(reason_codes) if status != "correct" else None,
                    "confidence_score": round(random.uniform(0.5, 0.95), 2)
                }
                field_feedback.append(field_data)
            
            # Create feedback record
            feedback_record = {
                "document_id": f"doc_{i:04d}",
                "user_id": f"user_{random.randint(1, 10):03d}",
                "session_id": f"session_{random.randint(1000, 9999)}",
                "prompt_version": prompt_version,
                "document_type": doc_type,
                "field_feedback": field_feedback,
                "overall_comment": f"Overall feedback for document {i}",
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
            }
            
            feedback_records.append(feedback_record)
        
        return feedback_records
    
    async def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit feedback to the API."""
        # Simulate API call
        await asyncio.sleep(0.1)
        
        # Add to local storage
        feedback_data["feedback_id"] = f"fb_{len(self.feedback_data):06d}"
        feedback_data["processing_time"] = round(random.uniform(0.05, 0.2), 3)
        self.feedback_data.append(feedback_data)
        
        return {
            "feedback_id": feedback_data["feedback_id"],
            "status": "success",
            "processing_time": feedback_data["processing_time"]
        }
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        await asyncio.sleep(0.1)
        
        if not self.feedback_data:
            return {
                "total_feedback_records": 0,
                "total_fields_evaluated": 0,
                "overall_accuracy_rate": 0.0,
                "most_problematic_fields": [],
                "feedback_trends": [],
                "active_alerts": [],
                "recent_optimizations": []
            }
        
        # Calculate statistics
        total_records = len(self.feedback_data)
        total_fields = sum(len(record["field_feedback"]) for record in self.feedback_data)
        
        # Calculate accuracy
        correct_fields = 0
        for record in self.feedback_data:
            for field in record["field_feedback"]:
                if field["feedback_status"] == "correct":
                    correct_fields += 1
        
        overall_accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
        
        # Get field statistics
        field_stats = {}
        for record in self.feedback_data:
            for field in record["field_feedback"]:
                field_name = field["field_name"]
                if field_name not in field_stats:
                    field_stats[field_name] = {"total": 0, "correct": 0, "incorrect": 0, "partial": 0}
                
                field_stats[field_name]["total"] += 1
                field_stats[field_name][field["feedback_status"]] += 1
        
        # Get most problematic fields
        problematic_fields = []
        for field_name, stats in field_stats.items():
            if stats["total"] >= 5:  # Minimum sample size
                error_rate = (stats["incorrect"] + stats["partial"]) / stats["total"]
                problematic_fields.append({
                    "field_name": field_name,
                    "error_rate": error_rate,
                    "total_feedback": stats["total"]
                })
        
        problematic_fields.sort(key=lambda x: x["error_rate"], reverse=True)
        
        return {
            "total_feedback_records": total_records,
            "total_fields_evaluated": total_fields,
            "overall_accuracy_rate": overall_accuracy,
            "most_problematic_fields": problematic_fields[:10],
            "feedback_trends": self._generate_trends(),
            "active_alerts": self._generate_alerts(),
            "recent_optimizations": self._generate_optimizations()
        }
    
    async def get_feedback_aggregation(self, field_name: str = None) -> List[Dict[str, Any]]:
        """Get feedback aggregation."""
        await asyncio.sleep(0.1)
        
        if not self.feedback_data:
            return []
        
        # Group by field
        field_groups = {}
        for record in self.feedback_data:
            for field in record["field_feedback"]:
                if field_name and field["field_name"] != field_name:
                    continue
                
                fname = field["field_name"]
                if fname not in field_groups:
                    field_groups[fname] = []
                field_groups[fname].append(field)
        
        aggregations = []
        for fname, fields in field_groups.items():
            total = len(fields)
            correct = sum(1 for f in fields if f["feedback_status"] == "correct")
            incorrect = sum(1 for f in fields if f["feedback_status"] == "incorrect")
            partial = sum(1 for f in fields if f["feedback_status"] == "partial")
            
            aggregation = {
                "field_name": fname,
                "total_feedback": total,
                "correct_count": correct,
                "incorrect_count": incorrect,
                "partial_count": partial,
                "accuracy_rate": correct / total if total > 0 else 0.0,
                "common_reasons": self._get_common_reasons(fields),
                "sample_comments": [f["comment"] for f in fields[:3] if f["comment"]],
                "prompt_versions": list(set(record["prompt_version"] for record in self.feedback_data)),
                "document_types": list(set(record["document_type"] for record in self.feedback_data))
            }
            aggregations.append(aggregation)
        
        return aggregations
    
    def _generate_trends(self) -> List[Dict[str, Any]]:
        """Generate trend data."""
        trends = []
        for i in range(7):  # Last 7 days
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            trends.append({
                "field_name": "overall",
                "time_period": "day",
                "date": date,
                "total_feedback": random.randint(5, 20),
                "accuracy_rate": round(random.uniform(0.6, 0.9), 2),
                "incorrect_count": random.randint(1, 5),
                "trend_direction": random.choice(["improving", "declining", "stable"])
            })
        return trends
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alert data."""
        if random.random() < 0.3:  # 30% chance of alerts
            return [{
                "alert_id": f"alert_{random.randint(1000, 9999)}",
                "field_name": random.choice(["invoice_number", "date", "amount"]),
                "alert_type": "high_error_rate",
                "severity": random.choice(["medium", "high"]),
                "description": "High error rate detected",
                "threshold_value": 0.3,
                "current_value": round(random.uniform(0.3, 0.6), 2),
                "prompt_version": "v2.0",
                "status": "active"
            }]
        return []
    
    def _generate_optimizations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        if random.random() < 0.5:  # 50% chance of recommendations
            return [{
                "recommendation_id": f"rec_{random.randint(1000, 9999)}",
                "field_name": random.choice(["invoice_number", "date", "amount"]),
                "recommendation_type": "prompt_clarification",
                "description": "Clarify prompt for better extraction",
                "suggested_actions": [
                    "Add more specific field definitions",
                    "Include validation examples",
                    "Specify expected format more clearly"
                ],
                "feedback_evidence": ["User feedback indicates confusion"],
                "expected_impact": round(random.uniform(0.1, 0.3), 2),
                "priority": random.choice(["medium", "high"]),
                "affected_prompt_versions": ["v2.0"],
                "status": "active"
            }]
        return []
    
    def _get_common_reasons(self, fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get common reason codes from fields."""
        reasons = {}
        for field in fields:
            if field.get("reason_code"):
                reason = field["reason_code"]
                reasons[reason] = reasons.get(reason, 0) + 1
        
        return [
            {"reason": reason, "count": count, "percentage": count / len(fields)}
            for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3]
        ]


# Streamlit Dashboard
def create_dashboard():
    """Create the Streamlit dashboard."""
    
    st.set_page_config(
        page_title="Feedback API Demo Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Feedback API Demo Dashboard")
    st.markdown("This dashboard demonstrates the feedback collection and analysis capabilities of the Document Extraction Evaluation Service.")
    
    # Initialize mock API
    if "mock_api" not in st.session_state:
        st.session_state.mock_api = MockFeedbackAPI()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Generate sample data
    if st.sidebar.button("🔄 Generate Sample Data"):
        with st.spinner("Generating sample feedback data..."):
            sample_data = st.session_state.mock_api.generate_sample_feedback(50)
            for record in sample_data:
                asyncio.run(st.session_state.mock_api.submit_feedback(record))
        st.success(f"Generated {len(sample_data)} sample feedback records!")
        st.rerun()
    
    # Clear data
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state.mock_api.feedback_data.clear()
        st.success("All data cleared!")
        st.rerun()
    
    # Get current statistics
    stats = asyncio.run(st.session_state.mock_api.get_feedback_stats())
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Feedback Records",
            value=stats["total_feedback_records"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Fields Evaluated",
            value=stats["total_fields_evaluated"],
            delta=None
        )
    
    with col3:
        accuracy_pct = stats["overall_accuracy_rate"] * 100
        st.metric(
            label="Overall Accuracy Rate",
            value=f"{accuracy_pct:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Active Alerts",
            value=len(stats["active_alerts"]),
            delta=None
        )
    
    # Charts section
    st.header("📈 Feedback Analytics")
    
    if stats["total_feedback_records"] > 0:
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Field Performance", "Trends", "Alerts & Recommendations", "Raw Data"])
        
        with tab1:
            st.subheader("Field-Level Performance")
            
            # Field accuracy chart
            if stats["most_problematic_fields"]:
                field_data = pd.DataFrame(stats["most_problematic_fields"])
                field_data["accuracy_rate"] = 1 - field_data["error_rate"]
                
                fig = px.bar(
                    field_data,
                    x="field_name",
                    y="accuracy_rate",
                    title="Field Accuracy Rates",
                    labels={"accuracy_rate": "Accuracy Rate", "field_name": "Field Name"},
                    color="accuracy_rate",
                    color_continuous_scale="RdYlGn"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Field statistics table
                st.subheader("Field Statistics")
                field_stats = asyncio.run(st.session_state.mock_api.get_feedback_aggregation())
                if field_stats:
                    df = pd.DataFrame(field_stats)
                    df["accuracy_rate"] = df["accuracy_rate"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader("Feedback Trends")
            
            # Trends chart
            if stats["feedback_trends"]:
                trends_data = pd.DataFrame(stats["feedback_trends"])
                trends_data["date"] = pd.to_datetime(trends_data["date"])
                
                fig = px.line(
                    trends_data,
                    x="date",
                    y="accuracy_rate",
                    title="Accuracy Rate Over Time",
                    labels={"accuracy_rate": "Accuracy Rate", "date": "Date"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig2 = px.bar(
                    trends_data,
                    x="date",
                    y="total_feedback",
                    title="Feedback Volume Over Time",
                    labels={"total_feedback": "Number of Feedback Records", "date": "Date"}
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Alerts & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚨 Active Alerts")
                if stats["active_alerts"]:
                    for alert in stats["active_alerts"]:
                        with st.container():
                            st.error(f"""
                            **{alert['alert_type'].replace('_', ' ').title()}**
                            - Field: {alert['field_name']}
                            - Severity: {alert['severity']}
                            - Current Value: {alert['current_value']:.1%}
                            - Threshold: {alert['threshold_value']:.1%}
                            """)
                else:
                    st.success("No active alerts")
            
            with col2:
                st.subheader("💡 Optimization Recommendations")
                if stats["recent_optimizations"]:
                    for rec in stats["recent_optimizations"]:
                        with st.container():
                            st.info(f"""
                            **{rec['recommendation_type'].replace('_', ' ').title()}**
                            - Field: {rec['field_name']}
                            - Priority: {rec['priority']}
                            - Expected Impact: {rec['expected_impact']:.1%}
                            - Status: {rec['status']}
                            """)
                else:
                    st.info("No optimization recommendations")
        
        with tab4:
            st.subheader("Raw Feedback Data")
            
            # Show recent feedback records
            if st.session_state.mock_api.feedback_data:
                recent_data = st.session_state.mock_api.feedback_data[-10:]  # Last 10 records
                
                for record in recent_data:
                    with st.expander(f"Document {record['document_id']} - {record['document_type']}"):
                        st.json(record)
    
    else:
        st.info("No feedback data available. Click 'Generate Sample Data' to create demo data.")
    
    # API Documentation
    st.header("🔗 API Endpoints")
    
    st.markdown("""
    ### Available Endpoints:
    
    **POST /feedback** - Submit user feedback
    ```json
    {
        "document_id": "doc_123",
        "user_id": "user_001",
        "prompt_version": "v1.0",
        "document_type": "invoice",
        "field_feedback": [
            {
                "field_name": "invoice_number",
                "feedback_status": "correct",
                "comment": "Correctly extracted"
            }
        ]
    }
    ```
    
    **GET /feedback/stats** - Get overall feedback statistics
    
    **GET /feedback/aggregation** - Get field-level aggregation
    
    **GET /feedback/trends** - Get feedback trends over time
    
    **GET /feedback/alerts** - Get active alerts
    
    **GET /feedback/recommendations** - Get optimization recommendations
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Document Extraction Evaluation Service - Feedback API Demo</p>
        <p>This demo shows how user feedback can be collected, analyzed, and used to improve document extraction prompts.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Run the feedback API demo."""
    print("🚀 Starting Feedback API Demo Dashboard...")
    print("📊 Opening Streamlit dashboard...")
    print("🌐 The dashboard will be available at http://localhost:8501")
    print("📝 Use the sidebar controls to generate sample data and explore the features")
    
    # Run the Streamlit app
    import subprocess
    import sys
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            __file__, "--server.port", "8501", "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")


if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        create_dashboard()
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        import sys
        
        # Install required packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        
        print("✅ Packages installed successfully!")
        print("🔄 Please run the script again to start the dashboard.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have the required packages installed:")
        print("   pip install streamlit plotly") 