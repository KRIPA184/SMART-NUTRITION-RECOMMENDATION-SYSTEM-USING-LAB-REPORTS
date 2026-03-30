"""
LLM Service – Gemini Integration
Receives ONLY structured JSON (never raw OCR text).
Generates personalized nutrition recommendations.
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

# ─── Configure Gemini ─────────────────────────────────────────────
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


def build_prompt(health_summary: dict) -> str:
    """
    Build a detailed prompt from the structured health summary.
    The LLM never sees raw OCR text — only validated, processed JSON.
    """
    patient = health_summary.get("patient_info", {})
    risk = health_summary.get("risk_assessment", {})
    abnormal = health_summary.get("abnormal_biomarkers", [])
    nutrient_targets = health_summary.get("nutrient_targets", {})
    top_features = health_summary.get("shap_top_features", [])
    med_interactions = health_summary.get("medication_interactions", [])
    supplement_recs = health_summary.get("supplement_recommendations", [])
    critical = health_summary.get("critical_alerts", [])

    prompt = f"""You are a certified clinical nutritionist AI assistant.
Based on the following VALIDATED and STRUCTURED health data, provide a comprehensive,
personalized nutrition recommendation. Use probabilistic language (e.g., "may benefit from",
"could consider") and NEVER provide a medical diagnosis.

═══════════════════════════════════════════
PATIENT PROFILE
═══════════════════════════════════════════
- Age: {patient.get('age', 'N/A')}
- Gender: {patient.get('gender', 'N/A')}
- Activity Level: {patient.get('activity_level', 'N/A')}
- Dietary Preference: {patient.get('dietary_preference', 'N/A')}
- Cuisine Preference: {patient.get('cuisine_preference', 'N/A')}
- Allergies: {', '.join(patient.get('allergies', [])) or 'None'}

═══════════════════════════════════════════
RISK ASSESSMENT
═══════════════════════════════════════════
- Overall Risk Score: {risk.get('risk_score', 'N/A')}%
- Category: {risk.get('risk_category', 'N/A')}

═══════════════════════════════════════════
ABNORMAL BIOMARKERS
═══════════════════════════════════════════
{json.dumps(abnormal, indent=2) if abnormal else "All biomarkers within normal range."}

═══════════════════════════════════════════
TOP INFLUENCING FACTORS (SHAP)
═══════════════════════════════════════════
{json.dumps(top_features, indent=2) if top_features else "N/A"}

═══════════════════════════════════════════
RECOMMENDED DAILY NUTRIENT TARGETS
═══════════════════════════════════════════
{json.dumps(nutrient_targets, indent=2)}

═══════════════════════════════════════════
MEDICATION-NUTRIENT INTERACTIONS
═══════════════════════════════════════════
{json.dumps(med_interactions, indent=2) if med_interactions else "None reported."}

═══════════════════════════════════════════
SUPPLEMENT RECOMMENDATIONS (Rule-Based)
═══════════════════════════════════════════
{json.dumps(supplement_recs, indent=2) if supplement_recs else "No supplements needed — all biomarkers normal."}

═══════════════════════════════════════════
CRITICAL ALERTS
═══════════════════════════════════════════
{json.dumps(critical, indent=2) if critical else "None."}

─────────────────────────────────────────
IMPORTANT CONSTRAINTS:
- Strictly respect the patient's Dietary Preference ({patient.get('dietary_preference', 'non-vegetarian')}).
  "vegetarian" = no meat/fish/eggs. "eggetarian" = vegetarian + eggs. "non-vegetarian" = all foods allowed.
- Cuisine Preference ({patient.get('cuisine_preference', 'indian')}):
  "indian" = only Indian dishes with local ingredient names.
  "global" = international dishes.
  "both" = a mix of Indian and global dishes.
- NEVER include any food the patient is allergic to: {', '.join(patient.get('allergies', [])) or 'None'}.

Please provide your response in the following JSON structure:
{{
    "personalized_diet_plan": "A 2-3 sentence overview of the recommended dietary strategy.",
    "foods_to_include": ["food1", "food2", "..."],
    "foods_to_avoid": ["food1", "food2", "..."],
    "weekly_meal_plan": {{
        "monday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "tuesday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "wednesday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "thursday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "friday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "saturday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }},
        "sunday": {{ "breakfast": "...", "lunch": "...", "dinner": "...", "snacks": "..." }}
    }},
    "recipes": [
        {{
            "name": "Recipe Name",
            "meal_type": "breakfast|lunch|dinner|snack",
            "prep_time": "15 mins",
            "servings": 2,
            "health_benefit": "Why this recipe helps the patient's specific condition",
            "ingredients": [
                {{ "item": "ingredient name", "quantity": "amount with unit" }},
                {{ "item": "...", "quantity": "..." }}
            ],
            "instructions": ["Step 1...", "Step 2...", "Step 3..."],
            "nutrition_per_serving": {{
                "calories": "250 kcal",
                "protein": "15g",
                "carbs": "30g",
                "fat": "8g",
                "fiber": "5g"
            }}
        }}
    ],
    "grocery_list": {{
        "fruits_and_vegetables": [{{ "item": "Spinach", "quantity": "500g" }}],
        "grains_and_cereals": [{{ "item": "Brown rice", "quantity": "1 kg" }}],
        "proteins": [{{ "item": "Lentils (masoor dal)", "quantity": "500g" }}],
        "dairy_and_alternatives": [{{ "item": "Low-fat yogurt", "quantity": "1 L" }}],
        "oils_and_condiments": [{{ "item": "Olive oil", "quantity": "500 ml" }}],
        "nuts_and_seeds": [{{ "item": "Almonds", "quantity": "200g" }}],
        "others": [{{ "item": "Green tea", "quantity": "1 box" }}]
    }},
    "recommended_supplements": [
        {{
            "name": "Supplement or tablet name",
            "dosage": "Recommended dosage",
            "form": "Tablet/Capsule/Softgel/etc.",
            "reason": "Why this supplement is recommended based on the patient's biomarker values",
            "precaution": "Any warnings or interactions to be aware of",
            "when_to_take": "e.g. With breakfast, On empty stomach, etc.",
            "duration": "e.g. 8-12 weeks then recheck, Ongoing, etc."
        }}
    ],
    "lifestyle_advice": ["advice1", "advice2", "..."],
    "medical_disclaimer": "This is AI-generated guidance, not medical advice. Consult your healthcare provider."
}}

IMPORTANT for recommended_supplements:
- Use the SUPPLEMENT RECOMMENDATIONS section above as a starting point, but enhance with your clinical nutrition knowledge.
- Include ALL supplements from the rule-based recommendations, plus any additional ones you deem beneficial.
- Provide practical "when_to_take" timing and realistic "duration" guidance for each.
- Flag any supplement-supplement or supplement-medication interactions.

Provide at least 5-7 detailed recipes that cover different meals and directly address the patient's abnormal biomarkers.
The grocery list should include ALL ingredients needed for the weekly meal plan and recipes, organized by category.
Return ONLY valid JSON. No markdown fences, no extra text.
"""
    return prompt


def get_nutrition_recommendation(health_summary: dict) -> dict:
    """
    Send structured health summary to Gemini and parse the response.
    Returns a dict with diet plan, meal plan, and lifestyle advice.
    """
    prompt = build_prompt(health_summary)

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # remove first line
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

        recommendation = json.loads(raw_text)
        return {"success": True, "recommendation": recommendation}

    except json.JSONDecodeError:
        # If Gemini doesn't return valid JSON, wrap text as-is
        return {
            "success": True,
            "recommendation": {
                "personalized_diet_plan": raw_text,
                "foods_to_include": [],
                "foods_to_avoid": [],
                "weekly_meal_plan": {},
                "recipes": [],
                "grocery_list": {},
                "recommended_supplements": [],
                "lifestyle_advice": [],
                "medical_disclaimer": "This is AI-generated guidance, not medical advice.",
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recommendation": None,
        }


# ══════════════════════════════════════════════════════════════════
#  CHATBOT – Conversational Nutrition Assistant
# ══════════════════════════════════════════════════════════════════

def _build_patient_context_block(patient_context: dict) -> str:
    """Build a concise patient context block for the chatbot system prompt."""
    if not patient_context:
        return "No lab report has been analyzed yet. Give general nutrition advice."

    pi = patient_context.get("patient_info", {})
    risk = patient_context.get("risk_result", {})
    validation = patient_context.get("validation", {})
    nutrients = patient_context.get("nutrient_targets", {})
    llm_rec = patient_context.get("llm_result", {})
    rec = llm_rec.get("recommendation", {}) if llm_rec.get("success") else {}

    # Abnormal biomarkers
    abnormal_lines = []
    for name, entry in validation.get("validated", {}).items():
        if entry.get("status") not in ("normal", "missing"):
            abnormal_lines.append(f"  - {name}: {entry.get('value')} {entry.get('unit')} ({entry.get('status')})")

    return f"""PATIENT CONTEXT (from analyzed lab report):
- Name: {pi.get('name', 'N/A')}
- Age: {pi.get('age', 'N/A')}, Gender: {pi.get('gender', 'N/A')}
- Activity Level: {pi.get('activity_level', 'N/A')}
- Dietary Preference: {pi.get('dietary_preference', 'N/A')}
- Cuisine Preference: {pi.get('cuisine_preference', 'N/A')}
- Allergies: {', '.join(pi.get('allergies', [])) or 'None'}
- Risk Score: {risk.get('risk_score', 'N/A')}% ({risk.get('risk_category', 'N/A')})

ABNORMAL BIOMARKERS:
{chr(10).join(abnormal_lines) if abnormal_lines else '  All within normal range.'}

RECOMMENDED NUTRIENT TARGETS:
{json.dumps(nutrients, indent=2) if nutrients else '  Not available.'}

EXISTING RECOMMENDATIONS (already shown to patient):
- Diet Plan: {rec.get('personalized_diet_plan', 'N/A')[:200] if rec else 'N/A'}
- Foods to Include: {', '.join(rec.get('foods_to_include', [])[:8]) if rec else 'N/A'}
- Foods to Avoid: {', '.join(rec.get('foods_to_avoid', [])[:8]) if rec else 'N/A'}
"""


def chat_with_nutritionist(user_message: str, patient_context: dict) -> str:
    """
    Conversational nutrition chatbot powered by Gemini.
    Uses patient's lab results + preferences as context.
    """
    context_block = _build_patient_context_block(patient_context)

    system_prompt = f"""You are a friendly Nutrition AI Assistant embedded in a Smart Nutrition Recommendation System.

YOUR ROLE:
• Help the patient understand their lab report results in simple, reassuring language
• Explain nutrient deficiencies and what they mean for health
• Suggest personalized recipes respecting dietary preference, cuisine preference, and allergies
• Provide grocery shopping lists when asked
• Give healthy diet and lifestyle tips
• Answer follow-up questions about the recommendations already shown

RULES:
1. Use simple, patient-friendly language — avoid medical jargon unless explaining it
2. STRICTLY respect dietary preference: vegetarian = no meat/fish/eggs; eggetarian = vegetarian + eggs; non-vegetarian = all foods
3. NEVER suggest foods the patient is allergic to
4. Match cuisine preference (Indian / Global / Both)
5. Use probabilistic language ("may help", "could benefit") — NEVER diagnose
6. Always end critical-health answers with "Please consult your doctor for personalized medical advice."
7. Keep responses concise but helpful (2-4 paragraphs max unless asked for detail)
8. Format lists with bullet points for readability
9. If asked about something outside nutrition/health, politely redirect

{context_block}"""

    try:
        full_prompt = system_prompt + "\n\n───────────────────────────────────────\nPATIENT'S QUESTION:\n" + user_message
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"I'm sorry, I couldn't process your question right now. Please try again shortly. (Error: {str(e)})"
