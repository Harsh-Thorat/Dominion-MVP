"""
Prompt templates for exercise analysis
"""
from langchain.prompts import PromptTemplate


# Exercise-specific criteria
EXERCISE_CRITERIA = {
    "squat": [
        "knee alignment (knees should track over toes)",
        "depth (hips below parallel)",
        "back position (neutral spine)",
        "chest up and forward",
        "weight distribution (midfoot)"
    ],
    "push-up": [
        "body alignment (straight line from head to heels)",
        "elbow position (45-degree angle from body)",
        "depth (chest nearly touching ground)",
        "core engagement",
        "head position (neutral)"
    ],
    "pull-up": [
        "full range of motion (chin over bar)",
        "controlled movement (no swinging)",
        "shoulder engagement",
        "core stability",
        "grip width"
    ],
    "plank": [
        "body alignment (straight line)",
        "core engagement",
        "hip position (not sagging or raised)",
        "shoulder position (over elbows)",
        "head position (neutral)"
    ],
    "lunge": [
        "front knee alignment (over ankle)",
        "back knee position (nearly touching ground)",
        "torso upright",
        "weight distribution",
        "step length"
    ],
    "general": [
        "body alignment",
        "movement control",
        "range of motion",
        "posture",
        "stability"
    ]
}


def get_system_message() -> str:
    """Return the system message for the coaching persona"""
    return """You are an expert calisthenics and fitness coach analyzing exercise form in real-time. Your role is to:
1. Observe the movement in the video frames
2. Identify form issues that could lead to injury or reduce effectiveness
3. Provide clear, actionable feedback in 1-3 sentences
4. Rate the overall form quality

Be encouraging but honest. Focus on the most important issues first. Use simple language that anyone can understand."""


def get_few_shot_examples(exercise_type: str) -> str:
    """Return few-shot examples for the given exercise"""
    
    examples = {
        "squat": """Example 1:
Video: Person performing squat with knees tracking properly, good depth, neutral spine
Analysis: Excellent squat form! Your depth is good, knees are properly aligned, and your back stays neutral throughout. Good form.

Example 2:
Video: Person performing squat with knees caving inward, insufficient depth
Analysis: Your squat needs work. Your knees are caving inward - focus on pushing them out to align with your toes. Also, try to go deeper if mobility allows. Major issue.

Example 3:
Video: Person performing squat with good alignment but slightly high
Analysis: Good technique overall, but you're stopping a bit high. Try to get your hips just below parallel for full range of motion. Minor issue.""",
        
        "push-up": """Example 1:
Video: Person performing push-up with straight body, proper depth, elbows at 45 degrees
Analysis: Perfect push-up form! Your body stays in a straight line, elbows are at the right angle, and you're getting full depth. Good form.

Example 2:
Video: Person performing push-up with hips sagging, partial depth
Analysis: Your hips are sagging - engage your core to maintain a straight line from head to heels. Also, try to go deeper, bringing your chest closer to the ground. Major issue.""",
        
        "pull-up": """Example 1:
Video: Person performing pull-up with full ROM, chin over bar, controlled movement
Analysis: Excellent pull-up! You're achieving full range of motion, keeping control throughout, and your chin clears the bar. Good form.

Example 2:
Video: Person performing pull-up with swinging, partial ROM
Analysis: You're using too much momentum and swinging. Focus on controlled movement and pull until your chin is fully over the bar. Major issue."""
    }
    
    return examples.get(exercise_type, examples["squat"])


def create_analysis_prompt(exercise_type: str = "general") -> PromptTemplate:
    """
    Create the main analysis prompt template
    
    Args:
        exercise_type: Type of exercise being performed
    
    Returns:
        PromptTemplate for analysis
    """
    criteria = EXERCISE_CRITERIA.get(exercise_type, EXERCISE_CRITERIA["general"])
    criteria_str = "\n".join(f"- {c}" for c in criteria)
    
    template = f"""You are analyzing a {exercise_type} exercise. The video frames show the movement sequence.

Key criteria to evaluate for {exercise_type}:
{criteria_str}

{get_few_shot_examples(exercise_type)}

Now analyze the current video frames shown above. Provide specific, actionable feedback in 1-3 sentences. Focus on the most important form issues. End with one of these ratings: "Good form", "Minor issue", or "Major issue".

Your analysis:"""
    
    return PromptTemplate(
        template=template,
        input_variables=[]
    )


def create_quick_analysis_prompt() -> str:
    """
    Create a simplified prompt for faster analysis
    """
    return """Analyze the exercise form shown in these video frames. In 1-2 sentences, identify the most important issue (if any) and rate the form as: Good form, Minor issue, or Major issue.

Your analysis:"""
