"""
Feedback Demo - Demonstrates user feedback collection and analysis.

This demo shows how the feedback-driven prompt evaluation system works,
including collecting user feedback, analyzing patterns, and generating
optimization recommendations.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any

from src.feedback.feedback_collector import FeedbackCollector
from data.feedback_data_generator import FeedbackDataGenerator


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def demo_feedback_collection():
    """Demonstrate feedback collection."""
    print_header("Feedback Collection Demo")
    
    # Initialize feedback collector
    collector = FeedbackCollector()
    generator = FeedbackDataGenerator()
    
    print_section("1. Collecting User Feedback")
    
    # Generate sample feedback data
    feedback_batch = generator.generate_feedback_batch(5)
    
    for i, feedback_data in enumerate(feedback_batch, 1):
        print(f"\nCollecting feedback {i}:")
        print(f"  Document ID: {feedback_data['document_id']}")
        print(f"  User ID: {feedback_data['user_id']}")
        print(f"  Document Type: {feedback_data['document_type']}")
        print(f"  Prompt Version: {feedback_data['prompt_version']}")
        
        # Collect feedback
        feedback_record = collector.collect_feedback(feedback_data)
        
        print(f"  Feedback ID: {feedback_record.feedback_id}")
        print(f"  Processing Time: {feedback_record.processing_time:.3f}s")
        print(f"  Fields Evaluated: {len(feedback_record.field_feedback)}")
        
        # Show field feedback summary
        correct_count = sum(1 for ff in feedback_record.field_feedback 
                          if ff.feedback_status.value == "correct")
        incorrect_count = sum(1 for ff in feedback_record.field_feedback 
                            if ff.feedback_status.value == "incorrect")
        partial_count = sum(1 for ff in feedback_record.field_feedback 
                          if ff.feedback_status.value == "partial")
        
        print(f"  Results: {correct_count} correct, {incorrect_count} incorrect, {partial_count} partial")
    
    return collector


def demo_feedback_analysis(collector: FeedbackCollector):
    """Demonstrate feedback analysis."""
    print_header("Feedback Analysis Demo")
    
    print_section("1. Overall Feedback Statistics")
    
    stats = collector.get_feedback_statistics()
    print(f"Total Feedback Records: {stats.total_feedback_records}")
    print(f"Total Fields Evaluated: {stats.total_fields_evaluated}")
    print(f"Overall Accuracy Rate: {stats.overall_accuracy_rate:.1%}")
    
    print_section("2. Field-Level Aggregation")
    
    aggregations = collector.get_feedback_aggregation()
    print(f"Fields with Feedback: {len(aggregations)}")
    
    for agg in aggregations[:5]:  # Show top 5 fields
        print(f"\nField: {agg.field_name}")
        print(f"  Total Feedback: {agg.total_feedback}")
        print(f"  Accuracy Rate: {agg.accuracy_rate:.1%}")
        print(f"  Breakdown: {agg.correct_count} correct, {agg.incorrect_count} incorrect, {agg.partial_count} partial")
        
        if agg.common_reasons:
            print(f"  Common Issues:")
            for reason in agg.common_reasons[:3]:
                print(f"    - {reason['reason']}: {reason['count']} times ({reason['percentage']:.1%})")
        
        if agg.sample_comments:
            print(f"  Sample Comments:")
            for comment in agg.sample_comments[:2]:
                print(f"    - \"{comment}\"")
    
    print_section("3. Feedback Trends")
    
    trends = collector.get_feedback_trends(time_period="7d")
    print(f"Trend Data Points: {len(trends)}")
    
    # Group trends by field
    field_trends = {}
    for trend in trends:
        if trend.field_name not in field_trends:
            field_trends[trend.field_name] = []
        field_trends[trend.field_name].append(trend)
    
    for field_name, field_trends_list in list(field_trends.items())[:3]:
        print(f"\nField: {field_name}")
        for trend in field_trends_list:
            print(f"  {trend.date}: {trend.accuracy_rate:.1%} accuracy ({trend.total_feedback} feedback) - {trend.trend_direction}")


def demo_alerts_and_recommendations(collector: FeedbackCollector):
    """Demonstrate alerts and recommendations."""
    print_header("Alerts and Recommendations Demo")
    
    print_section("1. Active Alerts")
    
    alerts = collector.get_active_alerts()
    if alerts:
        print(f"Active Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"\nAlert ID: {alert.alert_id}")
            print(f"Field: {alert.field_name}")
            print(f"Type: {alert.alert_type}")
            print(f"Severity: {alert.severity}")
            print(f"Description: {alert.description}")
            print(f"Current Value: {alert.current_value:.1%}")
            print(f"Threshold: {alert.threshold_value:.1%}")
    else:
        print("No active alerts at this time.")
    
    print_section("2. Optimization Recommendations")
    
    recommendations = collector.get_optimization_recommendations()
    if recommendations:
        print(f"Recent Recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"\nRecommendation ID: {rec.recommendation_id}")
            print(f"Field: {rec.field_name}")
            print(f"Type: {rec.recommendation_type}")
            print(f"Priority: {rec.priority}")
            print(f"Description: {rec.description}")
            print(f"Expected Impact: {rec.expected_impact:.1%}")
            print(f"Suggested Actions:")
            for action in rec.suggested_actions:
                print(f"  - {action}")
            
            if rec.feedback_evidence:
                print(f"Evidence from Feedback:")
                for evidence in rec.feedback_evidence[:2]:
                    print(f"  - \"{evidence}\"")
    else:
        print("No optimization recommendations at this time.")


def demo_historical_analysis():
    """Demonstrate historical feedback analysis."""
    print_header("Historical Feedback Analysis Demo")
    
    # Generate historical data
    generator = FeedbackDataGenerator()
    historical_feedback = generator.generate_historical_feedback(days_back=7, records_per_day=3)
    
    print_section("1. Historical Data Generation")
    print(f"Generated {len(historical_feedback)} historical feedback records")
    print(f"Time period: 7 days with 3 records per day")
    
    # Analyze historical patterns
    print_section("2. Historical Pattern Analysis")
    
    # Group by date
    daily_stats = {}
    for feedback_data in historical_feedback:
        # Extract date from timestamp (simplified)
        date_key = feedback_data.get('timestamp', datetime.now()).split('T')[0] if isinstance(feedback_data.get('timestamp'), str) else datetime.now().strftime('%Y-%m-%d')
        
        if date_key not in daily_stats:
            daily_stats[date_key] = {"total": 0, "correct": 0, "fields": {}}
        
        daily_stats[date_key]["total"] += 1
        
        for field in feedback_data["field_feedback"]:
            field_name = field["field_name"]
            if field_name not in daily_stats[date_key]["fields"]:
                daily_stats[date_key]["fields"][field_name] = {"total": 0, "correct": 0}
            
            daily_stats[date_key]["fields"][field_name]["total"] += 1
            if field["feedback_status"] == "correct":
                daily_stats[date_key]["fields"][field_name]["correct"] += 1
                daily_stats[date_key]["correct"] += 1
    
    print("Daily Feedback Summary:")
    for date, stats in sorted(daily_stats.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {date}: {stats['total']} records, {accuracy:.1%} accuracy")
        
        # Show field breakdown for this day
        for field_name, field_stats in stats["fields"].items():
            field_accuracy = field_stats["correct"] / field_stats["total"] if field_stats["total"] > 0 else 0
            print(f"    {field_name}: {field_accuracy:.1%} accuracy ({field_stats['correct']}/{field_stats['total']})")


def demo_api_integration():
    """Demonstrate API integration for feedback."""
    print_header("API Integration Demo")
    
    print_section("1. Sample API Request")
    
    # Sample feedback data for API
    sample_feedback = {
        "document_id": "doc_12345",
        "user_id": "user_001",
        "session_id": "session_abc123",
        "prompt_version": "v2.1",
        "document_type": "invoice",
        "field_feedback": [
            {
                "field_name": "invoice_number",
                "shown_value": "INV-2024-001",
                "feedback_status": "correct",
                "correction": None,
                "comment": "Correctly extracted",
                "reason_code": None,
                "confidence_score": 0.95
            },
            {
                "field_name": "date",
                "shown_value": "2024-01-15",
                "feedback_status": "incorrect",
                "correction": "2024-01-16",
                "comment": "Wrong date extracted",
                "reason_code": "wrong_format",
                "confidence_score": 0.85
            },
            {
                "field_name": "total_amount",
                "shown_value": "$1,250.00",
                "feedback_status": "correct",
                "correction": None,
                "comment": "Correct amount",
                "reason_code": None,
                "confidence_score": 0.92
            }
        ],
        "overall_comment": "Most fields correct, date field needs attention"
    }
    
    print("POST /feedback")
    print("Request Body:")
    print(json.dumps(sample_feedback, indent=2))
    
    print_section("2. Sample API Responses")
    
    print("GET /feedback/stats")
    print("Response would include:")
    print("  - Total feedback records")
    print("  - Overall accuracy rate")
    print("  - Most problematic fields")
    print("  - Active alerts")
    print("  - Recent recommendations")
    
    print("\nGET /feedback/aggregation?field_name=date&time_period=7d")
    print("Response would include:")
    print("  - Field-specific statistics")
    print("  - Common error reasons")
    print("  - Sample user comments")
    print("  - Accuracy trends")


def main():
    """Run the complete feedback demo."""
    print("🚀 Feedback-Driven Prompt Evaluation System Demo")
    print("This demo shows how user feedback is collected, analyzed, and used")
    print("to improve document extraction prompts.")
    
    try:
        # Demo 1: Feedback Collection
        collector = demo_feedback_collection()
        
        # Demo 2: Feedback Analysis
        demo_feedback_analysis(collector)
        
        # Demo 3: Alerts and Recommendations
        demo_alerts_and_recommendations(collector)
        
        # Demo 4: Historical Analysis
        demo_historical_analysis()
        
        # Demo 5: API Integration
        demo_api_integration()
        
        print_header("Demo Complete")
        print("✅ All feedback functionality demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("  • User feedback collection and validation")
        print("  • Field-level feedback aggregation")
        print("  • Trend analysis over time")
        print("  • Automated alert generation")
        print("  • Optimization recommendations")
        print("  • Historical pattern analysis")
        print("  • API integration examples")
        
        print("\nNext Steps:")
        print("  1. Run the API server: python -m src.api.evaluation_service")
        print("  2. Test feedback endpoints with real data")
        print("  3. Integrate with your document review UI")
        print("  4. Monitor alerts and recommendations")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 