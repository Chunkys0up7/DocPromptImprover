"""
API Test Demo - Demonstrates feedback API functionality

This demo shows how to interact with the feedback API endpoints
programmatically, including submitting feedback, retrieving statistics,
and analyzing trends.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import httpx
import random

# Sample data generator
class FeedbackDataGenerator:
    """Generate realistic feedback data for testing."""
    
    def __init__(self):
        self.document_types = ["invoice", "receipt", "contract", "form", "letter"]
        self.fields = ["invoice_number", "date", "amount", "vendor_name", "customer_name", 
                      "tax_amount", "total_amount", "payment_terms", "due_date", "po_number"]
        self.prompt_versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
        self.feedback_statuses = ["correct", "incorrect", "partial"]
        self.reason_codes = ["bad_ocr", "prompt_ambiguous", "irrelevant_text", "wrong_format", 
                            "missing_value", "extraneous_value", "confusing_context", "other"]
    
    def generate_feedback_record(self, record_id: int) -> Dict[str, Any]:
        """Generate a single feedback record."""
        
        # Generate random document
        doc_type = random.choice(self.document_types)
        prompt_version = random.choice(self.prompt_versions)
        
        # Generate 3-6 fields per document
        num_fields = random.randint(3, 6)
        field_feedback = []
        
        for j in range(num_fields):
            field_name = random.choice(self.fields)
            status = random.choice(self.feedback_statuses)
            
            field_data = {
                "field_name": field_name,
                "shown_value": f"Sample value {record_id}-{j}",
                "feedback_status": status,
                "correction": f"Corrected value {record_id}-{j}" if status == "incorrect" else None,
                "comment": f"User comment for {field_name}",
                "reason_code": random.choice(self.reason_codes) if status != "correct" else None,
                "confidence_score": round(random.uniform(0.5, 0.95), 2)
            }
            field_feedback.append(field_data)
        
        # Create feedback record
        feedback_record = {
            "document_id": f"doc_{record_id:04d}",
            "user_id": f"user_{random.randint(1, 10):03d}",
            "session_id": f"session_{random.randint(1000, 9999)}",
            "prompt_version": prompt_version,
            "document_type": doc_type,
            "field_feedback": field_feedback,
            "overall_comment": f"Overall feedback for document {record_id}"
        }
        
        return feedback_record
    
    def generate_batch(self, num_records: int) -> List[Dict[str, Any]]:
        """Generate a batch of feedback records."""
        return [self.generate_feedback_record(i) for i in range(num_records)]


class FeedbackAPIClient:
    """Client for interacting with the feedback API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit feedback to the API."""
        try:
            response = await self.client.post(
                f"{self.base_url}/feedback",
                json=feedback_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            return {"error": str(e)}
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        try:
            response = await self.client.get(f"{self.base_url}/feedback/stats")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    async def get_feedback_aggregation(self, field_name: str = None) -> List[Dict[str, Any]]:
        """Get feedback aggregation."""
        try:
            params = {}
            if field_name:
                params["field_name"] = field_name
            
            response = await self.client.get(
                f"{self.base_url}/feedback/aggregation",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error getting aggregation: {e}")
            return []
    
    async def get_feedback_trends(self, field_name: str = None, time_period: str = "7d") -> List[Dict[str, Any]]:
        """Get feedback trends."""
        try:
            params = {"time_period": time_period}
            if field_name:
                params["field_name"] = field_name
            
            response = await self.client.get(
                f"{self.base_url}/feedback/trends",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error getting trends: {e}")
            return []
    
    async def get_feedback_alerts(self) -> List[Dict[str, Any]]:
        """Get feedback alerts."""
        try:
            response = await self.client.get(f"{self.base_url}/feedback/alerts")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error getting alerts: {e}")
            return []
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        try:
            response = await self.client.get(f"{self.base_url}/feedback/recommendations")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    async def reset_feedback_data(self) -> Dict[str, Any]:
        """Reset feedback data."""
        try:
            response = await self.client.post(f"{self.base_url}/feedback/reset")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error resetting data: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


async def demo_feedback_submission(api_client: FeedbackAPIClient, generator: FeedbackDataGenerator):
    """Demonstrate feedback submission."""
    print_header("Feedback Submission Demo")
    
    print_section("1. Single Feedback Submission")
    
    # Generate and submit a single feedback record
    feedback_record = generator.generate_feedback_record(1)
    
    print("Submitting feedback record:")
    print(f"  Document ID: {feedback_record['document_id']}")
    print(f"  Document Type: {feedback_record['document_type']}")
    print(f"  Prompt Version: {feedback_record['prompt_version']}")
    print(f"  Fields: {len(feedback_record['field_feedback'])}")
    
    result = await api_client.submit_feedback(feedback_record)
    
    if "error" not in result:
        print(f"✅ Feedback submitted successfully!")
        print(f"  Feedback ID: {result.get('feedback_id', 'N/A')}")
        print(f"  Processing Time: {result.get('processing_time', 'N/A')}s")
    else:
        print(f"❌ Failed to submit feedback: {result['error']}")
    
    print_section("2. Batch Feedback Submission")
    
    # Generate and submit multiple feedback records
    batch_size = 10
    feedback_batch = generator.generate_batch(batch_size)
    
    print(f"Submitting {batch_size} feedback records...")
    
    successful_submissions = 0
    for i, record in enumerate(feedback_batch, 1):
        result = await api_client.submit_feedback(record)
        if "error" not in result:
            successful_submissions += 1
        print(f"  Record {i}: {'✅' if 'error' not in result else '❌'}")
    
    print(f"\nBatch submission complete: {successful_submissions}/{batch_size} successful")


async def demo_feedback_analysis(api_client: FeedbackAPIClient):
    """Demonstrate feedback analysis."""
    print_header("Feedback Analysis Demo")
    
    print_section("1. Overall Statistics")
    
    stats = await api_client.get_feedback_stats()
    
    if "error" not in stats:
        print(f"Total Feedback Records: {stats.get('total_feedback_records', 0)}")
        print(f"Total Fields Evaluated: {stats.get('total_fields_evaluated', 0)}")
        print(f"Overall Accuracy Rate: {stats.get('overall_accuracy_rate', 0):.1%}")
        
        problematic_fields = stats.get('most_problematic_fields', [])
        if problematic_fields:
            print(f"\nMost Problematic Fields:")
            for field in problematic_fields[:5]:
                print(f"  {field['field_name']}: {field['error_rate']:.1%} error rate")
    else:
        print(f"❌ Failed to get statistics: {stats['error']}")
    
    print_section("2. Field-Level Aggregation")
    
    aggregations = await api_client.get_feedback_aggregation()
    
    if aggregations and 'aggregations' in aggregations:
        agg_list = aggregations['aggregations']
        print(f"Field Aggregations: {len(agg_list)} fields")
        for agg in agg_list[:5]:
            print(f"\nField: {agg['field_name']}")
            print(f"  Total Feedback: {agg['total_feedback']}")
            print(f"  Accuracy Rate: {agg['accuracy_rate']:.1%}")
            print(f"  Breakdown: {agg['correct_count']} correct, {agg['incorrect_count']} incorrect, {agg['partial_count']} partial")
    else:
        print("No aggregation data available")
    
    print_section("3. Feedback Trends")
    
    trends = await api_client.get_feedback_trends(time_period="7d")
    
    if trends and 'trends' in trends:
        trend_list = trends['trends']
        print(f"Trend Data Points: {len(trend_list)}")
        for trend in trend_list[:3]:
            print(f"  {trend['date']}: {trend['accuracy_rate']:.1%} accuracy ({trend['total_feedback']} feedback) - {trend['trend_direction']}")
    else:
        print("No trend data available")


async def demo_alerts_and_recommendations(api_client: FeedbackAPIClient):
    """Demonstrate alerts and recommendations."""
    print_header("Alerts and Recommendations Demo")
    
    print_section("1. Active Alerts")
    
    alerts = await api_client.get_feedback_alerts()
    
    if alerts and 'alerts' in alerts:
        alert_list = alerts['alerts']
        print(f"Active Alerts: {len(alert_list)}")
        for alert in alert_list:
            print(f"\nAlert ID: {alert['alert_id']}")
            print(f"Field: {alert['field_name']}")
            print(f"Type: {alert['alert_type']}")
            print(f"Severity: {alert['severity']}")
            print(f"Description: {alert['description']}")
            print(f"Current Value: {alert['current_value']:.1%}")
            print(f"Threshold: {alert['threshold_value']:.1%}")
    else:
        print("No active alerts")
    
    print_section("2. Optimization Recommendations")
    
    recommendations = await api_client.get_optimization_recommendations()
    
    if recommendations and 'recommendations' in recommendations:
        rec_list = recommendations['recommendations']
        print(f"Optimization Recommendations: {len(rec_list)}")
        for rec in rec_list:
            print(f"\nRecommendation ID: {rec['recommendation_id']}")
            print(f"Field: {rec['field_name']}")
            print(f"Type: {rec['recommendation_type']}")
            print(f"Priority: {rec['priority']}")
            print(f"Description: {rec['description']}")
            print(f"Expected Impact: {rec['expected_impact']:.1%}")
            print(f"Status: {rec['status']}")
    else:
        print("No optimization recommendations")


async def demo_api_endpoints(api_client: FeedbackAPIClient):
    """Demonstrate all API endpoints."""
    print_header("API Endpoints Demo")
    
    endpoints = [
        ("GET /health", "Health check"),
        ("GET /", "Root endpoint"),
        ("GET /config", "Configuration"),
        ("GET /stats", "Evaluation statistics"),
        ("GET /performance-metrics", "Performance metrics"),
        ("GET /field-performance", "Field performance"),
        ("GET /document-type-performance", "Document type performance"),
        ("GET /error-patterns", "Error patterns"),
        ("POST /optimize", "Generate optimization recommendations")
    ]
    
    for endpoint, description in endpoints:
        print_section(f"{endpoint} - {description}")
        
        try:
            if endpoint.startswith("GET"):
                response = await api_client.client.get(f"{api_client.base_url}{endpoint.split()[1]}")
            else:
                response = await api_client.client.post(f"{api_client.base_url}{endpoint.split()[1]}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"Response: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"❌ Status: {response.status_code}")
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_error_handling(api_client: FeedbackAPIClient, generator: FeedbackDataGenerator):
    """Demonstrate error handling."""
    print_header("Error Handling Demo")
    
    print_section("1. Invalid Feedback Data")
    
    # Test with invalid data
    invalid_feedback = {
        "document_id": "doc_invalid",
        # Missing required fields
    }
    
    result = await api_client.submit_feedback(invalid_feedback)
    print(f"Invalid data submission: {'❌' if 'error' in result else '✅'}")
    if "error" in result:
        print(f"Error: {result['error']}")
    
    print_section("2. Invalid Field Feedback")
    
    # Test with invalid field feedback
    invalid_field_feedback = {
        "document_id": "doc_invalid_field",
        "user_id": "user_001",
        "prompt_version": "v1.0",
        "document_type": "invoice",
        "field_feedback": [
            {
                "field_name": "test_field",
                "feedback_status": "incorrect",
                # Missing correction for incorrect status
            }
        ]
    }
    
    result = await api_client.submit_feedback(invalid_field_feedback)
    print(f"Invalid field feedback: {'❌' if 'error' in result else '✅'}")
    if "error" in result:
        print(f"Error: {result['error']}")


async def main():
    """Run the complete API demo."""
    print("🚀 Feedback API Demo")
    print("This demo shows how to interact with the feedback API endpoints")
    print("Make sure the API server is running on http://127.0.0.1:8000")
    
    # Initialize components
    api_client = FeedbackAPIClient()
    generator = FeedbackDataGenerator()
    
    try:
        # Demo 1: Feedback Submission
        await demo_feedback_submission(api_client, generator)
        
        # Demo 2: Feedback Analysis
        await demo_feedback_analysis(api_client)
        
        # Demo 3: Alerts and Recommendations
        await demo_alerts_and_recommendations(api_client)
        
        # Demo 4: API Endpoints
        await demo_api_endpoints(api_client)
        
        # Demo 5: Error Handling
        await demo_error_handling(api_client, generator)
        
        print_header("Demo Complete")
        print("✅ All API functionality demonstrated successfully!")
        
        print("\nKey Features Demonstrated:")
        print("  • Feedback submission (single and batch)")
        print("  • Statistical analysis and aggregation")
        print("  • Trend analysis over time")
        print("  • Alert generation and monitoring")
        print("  • Optimization recommendations")
        print("  • Error handling and validation")
        print("  • All API endpoints")
        
        print("\nNext Steps:")
        print("  1. Start the API server: python -m src.api.evaluation_service")
        print("  2. Run the dashboard: python demos/feedback_api_demo.py")
        print("  3. Integrate with your application")
        print("  4. Monitor feedback and optimize prompts")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        raise
    finally:
        await api_client.close()


if __name__ == "__main__":
    asyncio.run(main()) 