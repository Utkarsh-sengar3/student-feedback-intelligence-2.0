"""Evidence-grounded recommendation engine for EduRABSA.

This module maps detected feedback categories to conservative action templates.
It does not invent evidence and does not claim recommendations are ground truth.
"""
TEMPLATES = {'Course - Workload': 'Review assignment volume, assessment spacing, and expected weekly workload; consider redistributing high-load tasks across the course schedule.', 'Course - Assessment': 'Review assessment difficulty, clarity, grading consistency, and assessment distribution; identify the most frequently criticized assessment points for targeted revision.', 'Staff - Teaching': 'Review teaching clarity, explanation quality, responsiveness, and instructional support; use recurring negative feedback to target teaching-development interventions.', 'Course - Learning activity': 'Review whether learning activities provide sufficient practice, engagement, and alignment with course objectives; revise activities with repeated negative feedback.', 'Course - Course materials': 'Audit course materials for completeness, clarity, accessibility, and alignment with assessments; prioritize materials repeatedly associated with negative feedback.', 'Course - Overall': 'Conduct a focused course review using the highest-frequency negative aspects and evidence examples to identify the main contributors to dissatisfaction.', 'University - Cost': 'Review the cost-related student concerns represented in the feedback and identify whether specific fees, value-for-money issues, or cost burdens recur across reviews.'}

def generate_recommendation(category: str) -> str:
    if category in TEMPLATES:
        return TEMPLATES[category]
    if "Staff" in category:
        return "Review the staff-related aspect identified in the feedback and target the recurring negative issue with focused support or training."
    if "Course" in category:
        return "Review the course-related aspect identified in the feedback and prioritize the recurring negative issue for course improvement."
    if "University" in category:
        return "Review the university-service issue identified in the feedback and investigate the recurring negative concern before selecting an intervention."
    return "Investigate the recurring negative concern and review the supporting feedback before selecting an intervention."
