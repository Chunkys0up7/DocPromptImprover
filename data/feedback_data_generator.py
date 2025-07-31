"""
Feedback data generator for testing and demos.

This module generates realistic user feedback data to test the feedback
collection and analysis system.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.models.feedback_models import (
    FeedbackStatus,
    FeedbackReason,
    FieldFeedback,
    UserFeedbackRecord
)


class FeedbackDataGenerator:
    """Generates realistic user feedback data for testing."""
    
    def __init__(self):
        """Initialize the feedback data generator."""
        self.field_names = [
            "invoice_number", "date", "total_amount", "vendor_name", 
            "customer_name", "tax_amount", "due_date", "po_number",
            "line_items", "payment_terms", "currency", "notes"
        ]
        
        self.document_types = [
            "invoice", "receipt", "form", "contract", "medical_record", "bank_statement"
        ]
        
        self.prompt_versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
        
        self.user_ids = [
            "user_001", "user_002", "user_003", "user_004", "user_005",
            "user_006", "user_007", "user_008", "user_009", "user_010"
        ]
        
        # Common feedback patterns
        self.feedback_patterns = {
            "invoice_number": {
                "correct_rate": 0.85,
                "common_errors": [
                    (FeedbackReason.WRONG_FORMAT, "Wrong format", "Should be INV-2024-001 format"),
                    (FeedbackReason.MISSING_VALUE, "Missing value", "Invoice number not found"),
                    (FeedbackReason.EXTRANEOUS_VALUE, "Extra characters", "Includes unnecessary text")
                ]
            },
            "date": {
                "correct_rate": 0.75,
                "common_errors": [
                    (FeedbackReason.WRONG_FORMAT, "Wrong format", "Should be YYYY-MM-DD format"),
                    (FeedbackReason.PROMPT_AMBIGUOUS, "Ambiguous date", "Multiple dates found, unclear which one"),
                    (FeedbackReason.BAD_OCR, "OCR error", "Date partially cut off or blurred")
                ]
            },
            "total_amount": {
                "correct_rate": 0.90,
                "common_errors": [
                    (FeedbackReason.WRONG_FORMAT, "Wrong format", "Missing currency symbol"),
                    (FeedbackReason.EXTRANEOUS_VALUE, "Extra text", "Includes tax or shipping info"),
                    (FeedbackReason.BAD_OCR, "OCR error", "Numbers misread")
                ]
            },
            "vendor_name": {
                "correct_rate": 0.80,
                "common_errors": [
                    (FeedbackReason.PROMPT_AMBIGUOUS, "Ambiguous name", "Multiple company names found"),
                    (FeedbackReason.IRRELEVANT_TEXT, "Wrong name", "Extracted customer name instead"),
                    (FeedbackReason.BAD_OCR, "OCR error", "Company name partially cut off")
                ]
            },
            "customer_name": {
                "correct_rate": 0.85,
                "common_errors": [
                    (FeedbackReason.PROMPT_AMBIGUOUS, "Ambiguous name", "Multiple names found"),
                    (FeedbackReason.IRRELEVANT_TEXT, "Wrong name", "Extracted vendor name instead"),
                    (FeedbackReason.BAD_OCR, "OCR error", "Name partially cut off")
                ]
            }
        }
    
    def generate_field_feedback(self, field_name: str, shown_value: str) -> FieldFeedback:
        """Generate realistic feedback for a single field."""
        
        pattern = self.feedback_patterns.get(field_name, {
            "correct_rate": 0.80,
            "common_errors": [
                (FeedbackReason.OTHER, "General error", "Field extraction issue")
            ]
        })
        
        # Determine feedback status based on pattern
        if random.random() < pattern["correct_rate"]:
            feedback_status = FeedbackStatus.CORRECT
            correction = None
            reason_code = None
            comment = "Correct extraction"
        else:
            # Choose a random error pattern
            error_pattern = random.choice(pattern["common_errors"])
            reason_code, error_type, error_desc = error_pattern
            
            if random.random() < 0.3:  # 30% chance of partial
                feedback_status = FeedbackStatus.PARTIAL
                correction = None
                comment = f"Partially correct: {error_desc}"
            else:
                feedback_status = FeedbackStatus.INCORRECT
                correction = self._generate_correction(field_name, shown_value)
                comment = f"Incorrect: {error_desc}"
        
        return FieldFeedback(
            field_name=field_name,
            shown_value=shown_value,
            feedback_status=feedback_status,
            correction=correction,
            comment=comment,
            reason_code=reason_code,
            confidence_score=random.uniform(0.5, 0.95)
        )
    
    def _generate_correction(self, field_name: str, shown_value: str) -> str:
        """Generate a realistic correction for a field."""
        
        if field_name == "invoice_number":
            return f"INV-2024-{random.randint(1, 999):03d}"
        elif field_name == "date":
            return f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        elif field_name == "total_amount":
            return f"${random.uniform(100, 10000):.2f}"
        elif field_name == "vendor_name":
            vendors = ["Acme Corp", "Tech Solutions Inc", "Global Industries", "Local Business LLC"]
            return random.choice(vendors)
        elif field_name == "customer_name":
            customers = ["John Smith", "Jane Doe", "ABC Company", "XYZ Corporation"]
            return random.choice(customers)
        else:
            return f"Corrected {field_name}"
    
    def generate_feedback_record(self, document_id: str = None) -> UserFeedbackRecord:
        """Generate a complete feedback record for a document."""
        
        if document_id is None:
            document_id = f"doc_{random.randint(1000, 9999)}"
        
        # Generate field feedback for 3-8 fields
        num_fields = random.randint(3, 8)
        selected_fields = random.sample(self.field_names, num_fields)
        
        field_feedback = []
        for field_name in selected_fields:
            # Generate a realistic shown value
            shown_value = self._generate_shown_value(field_name)
            field_feedback.append(self.generate_field_feedback(field_name, shown_value))
        
        # Generate overall comment (30% chance)
        overall_comment = None
        if random.random() < 0.3:
            comments = [
                "Most fields extracted correctly",
                "Several formatting issues",
                "Good overall extraction",
                "Needs improvement in date handling",
                "Vendor name extraction is problematic"
            ]
            overall_comment = random.choice(comments)
        
        return UserFeedbackRecord(
            feedback_id=f"feedback_{random.randint(10000, 99999)}",
            document_id=document_id,
            user_id=random.choice(self.user_ids),
            session_id=f"session_{random.randint(100, 999)}",
            prompt_version=random.choice(self.prompt_versions),
            document_type=random.choice(self.document_types),
            field_feedback=field_feedback,
            overall_comment=overall_comment
        )
    
    def _generate_shown_value(self, field_name: str) -> str:
        """Generate a realistic shown value for a field."""
        
        if field_name == "invoice_number":
            formats = [
                f"INV-2024-{random.randint(1, 999):03d}",
                f"INV{random.randint(1000, 9999)}",
                f"INVOICE-{random.randint(100, 999)}"
            ]
            return random.choice(formats)
        elif field_name == "date":
            formats = [
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                f"{random.randint(1, 12)}/{random.randint(1, 28)}/2024",
                f"{random.randint(1, 28)}-{random.randint(1, 12)}-2024"
            ]
            return random.choice(formats)
        elif field_name == "total_amount":
            formats = [
                f"${random.uniform(100, 10000):.2f}",
                f"${random.uniform(100, 10000):.2f} USD",
                f"{random.uniform(100, 10000):.2f}"
            ]
            return random.choice(formats)
        elif field_name == "vendor_name":
            vendors = [
                "Acme Corporation",
                "Tech Solutions Inc.",
                "Global Industries LLC",
                "Local Business Co.",
                "International Corp"
            ]
            return random.choice(vendors)
        elif field_name == "customer_name":
            customers = [
                "John Smith",
                "Jane Doe",
                "ABC Company Inc.",
                "XYZ Corporation",
                "Sample Customer LLC"
            ]
            return random.choice(customers)
        else:
            return f"Sample {field_name} value"
    
    def generate_feedback_batch(self, num_records: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of feedback records for API testing."""
        
        feedback_batch = []
        
        for i in range(num_records):
            feedback_record = self.generate_feedback_record()
            
            # Convert to dict format for API
            feedback_data = {
                "document_id": feedback_record.document_id,
                "user_id": feedback_record.user_id,
                "session_id": feedback_record.session_id,
                "prompt_version": feedback_record.prompt_version,
                "document_type": feedback_record.document_type,
                "field_feedback": [
                    {
                        "field_name": ff.field_name,
                        "shown_value": ff.shown_value,
                        "feedback_status": ff.feedback_status,
                        "correction": ff.correction,
                        "comment": ff.comment,
                        "reason_code": ff.reason_code,
                        "confidence_score": ff.confidence_score
                    }
                    for ff in feedback_record.field_feedback
                ],
                "overall_comment": feedback_record.overall_comment
            }
            
            feedback_batch.append(feedback_data)
        
        return feedback_batch
    
    def generate_historical_feedback(self, days_back: int = 30, records_per_day: int = 5) -> List[Dict[str, Any]]:
        """Generate historical feedback data over a time period."""
        
        all_feedback = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        for day in range(days_back):
            current_date = base_date + timedelta(days=day)
            
            # Generate records for this day
            for _ in range(records_per_day):
                feedback_record = self.generate_feedback_record()
                
                # Set timestamp to this day
                feedback_record.timestamp = current_date + timedelta(
                    hours=random.randint(9, 17),
                    minutes=random.randint(0, 59)
                )
                
                # Convert to dict format
                feedback_data = {
                    "document_id": feedback_record.document_id,
                    "user_id": feedback_record.user_id,
                    "session_id": feedback_record.session_id,
                    "prompt_version": feedback_record.prompt_version,
                    "document_type": feedback_record.document_type,
                    "field_feedback": [
                        {
                            "field_name": ff.field_name,
                            "shown_value": ff.shown_value,
                            "feedback_status": ff.feedback_status,
                            "correction": ff.correction,
                            "comment": ff.comment,
                            "reason_code": ff.reason_code,
                            "confidence_score": ff.confidence_score
                        }
                        for ff in feedback_record.field_feedback
                    ],
                    "overall_comment": feedback_record.overall_comment
                }
                
                all_feedback.append(feedback_data)
        
        return all_feedback


def main():
    """Generate sample feedback data and save to file."""
    
    generator = FeedbackDataGenerator()
    
    # Generate sample feedback batch
    print("Generating sample feedback data...")
    feedback_batch = generator.generate_feedback_batch(20)
    
    # Save to file
    output_file = "data/sample_feedback_data.json"
    with open(output_file, 'w') as f:
        json.dump(feedback_batch, f, indent=2, default=str)
    
    print(f"Generated {len(feedback_batch)} feedback records")
    print(f"Saved to {output_file}")
    
    # Generate historical data
    print("\nGenerating historical feedback data...")
    historical_feedback = generator.generate_historical_feedback(days_back=7, records_per_day=3)
    
    historical_file = "data/historical_feedback_data.json"
    with open(historical_file, 'w') as f:
        json.dump(historical_feedback, f, indent=2, default=str)
    
    print(f"Generated {len(historical_feedback)} historical feedback records")
    print(f"Saved to {historical_file}")
    
    # Print sample statistics
    print("\nSample Feedback Statistics:")
    correct_count = sum(
        1 for record in feedback_batch
        for field in record["field_feedback"]
        if field["feedback_status"] == "correct"
    )
    total_fields = sum(len(record["field_feedback"]) for record in feedback_batch)
    accuracy_rate = correct_count / total_fields if total_fields > 0 else 0
    
    print(f"Total feedback records: {len(feedback_batch)}")
    print(f"Total field evaluations: {total_fields}")
    print(f"Overall accuracy rate: {accuracy_rate:.1%}")
    
    # Show field breakdown
    field_stats = {}
    for record in feedback_batch:
        for field in record["field_feedback"]:
            field_name = field["field_name"]
            if field_name not in field_stats:
                field_stats[field_name] = {"correct": 0, "total": 0}
            
            field_stats[field_name]["total"] += 1
            if field["feedback_status"] == "correct":
                field_stats[field_name]["correct"] += 1
    
    print("\nField-level accuracy:")
    for field_name, stats in field_stats.items():
        accuracy = stats["correct"] / stats["total"]
        print(f"  {field_name}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main() 