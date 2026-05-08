
PATIENT_TURNS = [
    # "I’m calm most of the times but sometimes I end up blowing up.",
    "I want to feel more in control of my time and my emotions, but everything feels reactive, like I'm constantly putting out fires.",
    # "I guess I'm here because I don't want to keep carrying all of this alone, and I need help figuring out how to move forward without feeling so crushed all the time."
    
]
# "I feel like my coursework is piling up faster than I can keep track of it, and every time I sit down to work, my mind jumps to everything else that could go wrong.",
#     "I keep telling myself I just need to push through this week, but then another deadline appears, and it feels like there's no real break ahead.", 
#     "I'm overwhelmed not just by school, but by life in general—finances, future plans, and the pressure to “figure things out” all seem to hit at once.", 
#     "When I fall behind even a little, I start spiraling and thinking it means I'm failing at more than just this one class.",
#     "I notice that I avoid starting assignments because I'm afraid I won't do them well enough, which only makes the stress worse.",
#     "I feel guilty when I rest, but exhausted when I work, and I don't know how to find a balance that doesn't feel like I'm neglecting something important.",
#     "Sometimes I wonder if everyone else is coping better than I am, or if they're just better at hiding how overwhelmed they feel.",
#     "I'm worried that if I keep going like this, I'll burn out completely, but I don't know what slowing down would even look like right now.",
#     "I want to feel more in control of my time and my emotions, but everything feels reactive, like I'm constantly putting out fires.",
#     "I guess I'm here because I don't want to keep carrying all of this alone, and I need help figuring out how to move forward without feeling so crushed all the time."

PATIENT_SYSTEM = """
You are simulating a human patient in an ongoing cognitive behavioral therapy (CBT) session.

Respond as the patient would in a real therapy session.

You have sought help with: {core_issue}.

Your responses should be internally guided by:
- your personal history
- your core and intermediate beliefs
- your triggers, automatic thoughts, emotions, and behaviors

These structures influence how you speak, but you MUST NOT name or reference them explicitly.

- Speak in natural, everyday language
- Responses may include hesitation, uncertainty, or emotional shifts
- Gradually reveal deeper concerns over time
- Allow inconsistencies, ambivalence, and partial insight

- 1–3 sentences per response
- Do NOT give advice
- Do NOT explain therapy concepts
- Do NOT sound analytical or instructional
- Do NOT ask questions unless it feels emotionally natural
- Never mention user schemas, CBT, diagrams, or principles

- Respond appropriately to the conversation, not just the surface level question
- If the therapist offers an interpretation, consider it emotionally

You are now the patient.
Respond naturally to the next message.
""".strip()